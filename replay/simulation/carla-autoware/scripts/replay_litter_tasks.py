"""Recompute recorded task plans on PC/J6; compare outputs without advancing a vehicle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

IMPORT_STARTED = time.perf_counter()
import numpy as np
import shapely

from console_session import brush_forward_reach
from coverage_boustrophedon import plan_boustrophedon
from litter_observations import observation_packet, task_targets
from litter_routes import RouteConfig, plan_litter_route
from tricycle_navigation import load_profile, wrap

IMPORT_SECONDS = time.perf_counter() - IMPORT_STARTED
SOURCE_NAMES = ('replay_litter_tasks.py', 'litter_routes.py', 'coverage_boustrophedon.py',
                'tricycle_navigation.py', 'console_session.py', 'litter_observations.py')
PROFILE_KEYS = ('wheel_base', 'max_steer_angle', 'max_speed_mps', 'reverse_speed_mps',
                'lateral_acceleration_mps2', 'steering_rate_rad_s', 'deceleration_mps2',
                'acceleration_mps2', 'front', 'rear', 'left', 'right')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_snapshots(run, profile):
    """Bind recorded inputs to their admitted packet; keep references outside planner inputs."""
    folder = run / 'litter-acceptance'
    cases = read(folder / 'cases.json')
    if not cases:
        raise ValueError('no recorded tasks')
    packets = {}
    for line in (folder / 'observations.jsonl').read_text().splitlines():
        row = json.loads(line)
        packet = row['packet']
        if not row['response']['accepted'] or observation_packet(packet['payload']) != packet:
            raise ValueError('observation record is not an admitted, intact packet')
        packets[packet['sha256']] = packet
    snapshots = []
    for case in cases:
        # The final task retains the route actually dispatched. A draft may have a different EKF start.
        tasks = [('dispatch', case['final'])]
        if case['initial']['route'] != case['final']['route']:
            tasks.append(('draft', case['initial']))
        for phase, task in tasks:
            route = task['route']
            if route['vehicleSource'] != profile['source_hashes']:
                raise ValueError('vehicle geometry differs from the recorded task')
            policy = task['cleaningPolicy']
            if (policy, route['planner']) not in {
                    ('full_coverage', 'boustrophedon'), ('on_demand', 'litter-boustrophedon-quintic')}:
                raise ValueError('unsupported recorded planner')
            start = np.asarray(route['start'], dtype=float)
            expected = np.asarray(route['poses'], dtype=float)
            if (start.shape != (3,) or expected.ndim != 2 or expected.shape[1] != 5
                    or len(expected) < 2 or not np.isfinite(start).all() or not np.isfinite(expected).all()):
                raise ValueError('missing finite start or reference route')
            targets = task.get('litterTargets', [])
            observation = task.get('litterObservation')
            if policy == 'on_demand':
                packet = packets[observation['sha256']]
                if (packet['payload']['mapVersionId'] != task['mapVersionId']
                        or packet['payload']['frameId'] != 'map'
                        or packet['payload']['sourceTime'] != observation['sourceTime']
                        or task_targets(packet, task['polygon'], task.get('targetIds')) != targets):
                    raise ValueError('recorded targets differ from their observation')
            brushes = route['brushes']
            roller = next(b for b in brushes if b['id'] == 'roller')
            inputs = dict(policy=policy, polygon=task['polygon'], start=start,
                allowed=route['allowed'], targets=targets,
                swath_m=roller['width']*.9 if policy == 'on_demand' else route['swathWidth'],
                reach_m=brush_forward_reach(brushes),
                turn_radius=profile['wheel_base']/math.tan(profile['max_steer_angle']))
            snapshots.append(dict(name=case['name']+'/'+phase, task_id=task['id'], inputs=inputs,
                observation=observation, expected=expected, expected_status=route['status']))
    if len({s['name'] for s in snapshots}) != len(snapshots):
        raise ValueError('duplicate task snapshot names')
    return snapshots


def replan(inputs, profile):
    if inputs['policy'] == 'full_coverage':
        route = plan_boustrophedon(inputs['polygon'], inputs['start'], inputs['swath_m'],
                                  turn_radius=inputs['turn_radius'])
        return np.asarray(route['poses'], dtype=float), 'planned'
    targets = inputs['targets']
    visible = [t for t in targets if t['visible']]
    regions = [np.asarray([t['point']] if t['kind'] == 'point' else t['polygon']) for t in visible]
    config = RouteConfig(profile, inputs['swath_m'], inputs['reach_m'])
    route = plan_litter_route(regions, inputs['start'], np.asarray(inputs['allowed']), config)
    status = ('waiting_for_observations' if not len(route.poses) else
              'partial_plan' if len(route.pending) or len(visible) != len(targets) else 'planned')
    return route.poses, status


def compare_route(actual, expected, status, expected_status, tolerance):
    same_shape = actual.shape == expected.shape
    finite = bool(np.isfinite(actual).all())
    metrics = dict(same_shape=same_shape, finite=finite, status_equal=status == expected_status,
                   position_max_m=None, yaw_max_rad=None, gear_and_brush_equal=False)
    if same_shape and finite and len(actual):
        metrics.update(position_max_m=float(np.linalg.norm(actual[:, :2]-expected[:, :2], axis=1).max()),
            yaw_max_rad=float(np.abs(wrap(actual[:, 2]-expected[:, 2])).max()),
            gear_and_brush_equal=bool(np.array_equal(actual[:, 3:], expected[:, 3:])))
    metrics['passed'] = bool(same_shape and finite and metrics['status_equal']
        and metrics['gear_and_brush_equal'] and metrics['position_max_m'] <= tolerance
        and metrics['yaw_max_rad'] <= tolerance)
    return metrics


def environment(target):
    machine = platform.machine()
    if target == 'j6' and machine.lower() not in ('aarch64', 'arm64'):
        raise ValueError('J6 execution requires an ARM64 host; this host is '+machine)
    model = Path('/proc/device-tree/model')
    return dict(requested_platform=target, machine=machine, hostname=platform.node(),
        os=platform.platform(), python=sys.version, numpy=np.__version__, shapely=shapely.__version__,
        device_tree_model=model.read_text().rstrip('\x00') if model.exists() else None,
        affinity=sorted(os.sched_getaffinity(0)), load_average=os.getloadavg(),
        thread_environment={k:os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def run_replay(run, output, *, target='pc', repetitions=3, tolerance=1e-6, profile_path=None):
    if repetitions < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive repetitions and finite tolerance required')
    host = environment(target)
    profile_path = profile_path or Path(__file__).resolve().parents[1]/'config/tricycle-navigation.json'
    original = read(run/'source-manifest.json')
    if digest(profile_path) != original['simulation/carla-autoware/config/tricycle-navigation.json']:
        raise ValueError('navigation configuration differs from the recorded input')
    profile = load_profile(profile_path)
    snapshots = load_snapshots(run, profile)
    numeric_profile = {k:profile[k] for k in PROFILE_KEYS}
    scripts = Path(__file__).resolve().parent
    files = [run/'litter-acceptance'/f for f in ('cases.json', 'observations.jsonl')]
    config = read(profile_path)
    files += [profile_path, *(profile_path.parent/config[k] for k in ('geometry_source', 'steering_source'))]
    source_identity = {name:dict(current=digest(scripts/name),
        recorded=original.get('simulation/carla-autoware/scripts/'+name)) for name in SOURCE_NAMES}
    inputs = {p.name:digest(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (output/'outputs.jsonl').open('x', encoding='utf-8') as stream:
        for repetition in range(repetitions):
            for snapshot in snapshots:
                row = dict(name=snapshot['name'], repetition=repetition, task_id=snapshot['task_id'],
                           observation=snapshot['observation'])
                begin = time.perf_counter()
                try:
                    poses, status = replan(snapshot['inputs'], numeric_profile)
                    row['compute_s'] = time.perf_counter()-begin
                    row['comparison'] = compare_route(poses, snapshot['expected'], status,
                                                     snapshot['expected_status'], tolerance)
                    row.update(status=status, poses=poses.tolist() if row['comparison']['finite'] else None, error=None)
                except Exception as error:
                    row.update(compute_s=time.perf_counter()-begin, error=repr(error),
                               comparison=dict(passed=False), poses=None)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    cases = []
    for snapshot in snapshots:
        selected = [r for r in rows if r['name'] == snapshot['name']]
        completed = [r for r in selected if r['error'] is None]
        times = [r['compute_s'] for r in completed]
        cases.append(dict(name=snapshot['name'], requested=len(selected), completed=len(completed),
            passed=all(r['comparison']['passed'] for r in selected),
            first_compute_s=selected[0]['compute_s'],
            compute_s=dict(p50=float(np.percentile(times, 50)), p95=float(np.percentile(times, 95)),
                           maximum=max(times)) if times else None,
            position_max_m=max((r['comparison']['position_max_m'] for r in completed
                               if r['comparison']['position_max_m'] is not None), default=None),
            yaw_max_rad=max((r['comparison']['yaw_max_rad'] for r in completed
                            if r['comparison']['yaw_max_rad'] is not None), default=None)))
    report = dict(schema='horizon-litter-open-loop-v1', created_at=datetime.now(timezone.utc).isoformat(),
        scope='recorded task inputs -> route; spatial samples compare to dispatched/draft routes',
        schedule='sequential event-driven task calls; no fixed-rate or control-latency claim',
        tolerance=dict(position_m=tolerance, yaw_rad=tolerance), environment=host,
        import_s=IMPORT_SECONDS, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sources=source_identity, inputs=inputs, requested=len(rows),
        completed=sum(r['error'] is None for r in rows), passed=all(c['passed'] for c in cases), cases=cases)
    (output/'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--platform', choices=('pc', 'j6'), default='pc')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--tolerance', type=float, default=1e-6, help='metres/radians, output regression only')
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    report = run_replay(args.run, args.output, target=args.platform, repetitions=args.repetitions,
                        tolerance=args.tolerance, profile_path=args.profile)
    print(json.dumps({k:report[k] for k in ('passed', 'requested', 'completed', 'cases')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
