"""Three-wheel references and actuator/action gates; no map or perception producer.

Coordinates: x forward, y left, rear axle ground reference. Acceleration at the
Autoware control output is pedal-like: positive accelerates in either gear.
Reference acceleration is signed in the body's longitudinal coordinate.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def fresh(now, stamp, limit):
    # CARLA float seconds -> ROS integer nanoseconds may round the same frame
    # one nanosecond differently. This tolerance is far below one sensor tick.
    return all(math.isfinite(v) for v in (now, stamp, limit)) and -1e-8 <= now-stamp <= limit+1e-6


def load_profile(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1]/'config/tricycle-navigation.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['schema_version'] != 1 or data['reference_point'] != 'rear_axle_ground':
        raise ValueError('unsupported navigation profile')
    geo_path = (path.parent/data['geometry_source']).resolve()
    steer_path = (path.parent/data['steering_source']).resolve()
    geo = json.loads(geo_path.read_text(encoding='utf-8'))
    steer = json.loads(steer_path.read_text(encoding='utf-8'))
    centers = geo['wheel_centers_m']
    f, l, r = (np.asarray(centers[n], dtype=float) for n in
               ('Wheel_Front_Left', 'Wheel_Rear_Left', 'Wheel_Rear_Right'))
    if not np.allclose([l[0], r[0], f[1], l[1]+r[1]], 0, atol=1e-6):
        raise ValueError('geometry is not a rear-axle-referenced tricycle')
    if not steer['passed'] or any(abs(a) > 1e-6 for row in steer['rows'] for a in row['wheel_angles_deg'][1:]):
        raise ValueError('rear wheels must not steer')
    angles = [row['wheel_angles_deg'][0] for row in steer['rows']]
    if min(angles) >= 0 or max(angles) <= 0:
        raise ValueError('both steering limits must be measured')
    lo, hi = np.asarray(geo['bounds_m'], dtype=float)
    data.update(wheel_base=float(f[0]), wheel_tread=float(l[1]-r[1]),
        front=float(hi[0]), rear=float(lo[0]), left=float(hi[1]), right=float(lo[1]),
        max_steer_angle=math.radians(min(-min(angles), max(angles))),
        vehicle_height=float(hi[2]), wheel_radius=float(f[2]),
        source_hashes={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in (geo_path, steer_path)})
    for key in ('wheel_base', 'max_steer_angle', 'max_speed_mps', 'reverse_speed_mps',
                'acceleration_mps2', 'deceleration_mps2', 'lateral_acceleration_mps2',
                'steering_rate_rad_s', 'steering_time_constant_s', 'feedback_max_age_s', 'stop_dwell_s',
                'command_source_age_s', 'command_wall_age_s'):
        if not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError('invalid navigation parameter: '+key)
    if not math.isfinite(data['command_delay_s']) or data['command_delay_s'] < 0:
        raise ValueError('invalid navigation parameter: command_delay_s')
    return data


def stopping_budget(profile, velocity_mps=None):
    """Command lease, interface delay, possible acceleration, then braking.

    These are bounds in the declared dynamics model. Wall-clock expiry is an
    independent watchdog; its seconds cannot be added to simulation seconds.
    The executable bridge reads the same source/wall lease values from this
    profile, so a consumer cannot omit the lifetime of an in-flight command.
    """
    from dg202604.safety.primitives import StoppingEnvelope
    for name in ('command_source_age_s', 'command_wall_age_s', 'command_delay_s'):
        value = profile[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name != 'command_delay_s' and value == 0)):
            raise ValueError('invalid navigation parameter: ' + name)
    velocity = profile['max_speed_mps'] if velocity_mps is None else velocity_mps
    latency = profile['command_source_age_s'] + profile['command_delay_s']
    values = dict(velocity_mps=velocity, end_to_end_latency_s=latency,
                  minimum_deceleration_mps2=profile['deceleration_mps2'],
                  maximum_acceleration_mps2=profile['acceleration_mps2'])
    return dict(response_time_s=StoppingEnvelope.required_time_s(**values),
                distance_m=StoppingEnvelope.required_distance_m(**values, safety_margin_m=0.),
                velocity_mps=abs(velocity), command_source_age_s=profile['command_source_age_s'],
                command_wall_age_s=profile['command_wall_age_s'],
                interface_delay_s=profile['command_delay_s'],
                maximum_acceleration_mps2=profile['acceleration_mps2'],
                minimum_deceleration_mps2=profile['deceleration_mps2'],
                dynamics_status=profile['dynamics_status'],
                dynamics_validated=profile['dynamics_status'] == 'simulation_identified_limits')


def controller_vehicle_parameters(profile):
    return dict(wheel_base=profile['wheel_base'], wheel_tread=profile['wheel_tread'],
        wheel_radius=profile['wheel_radius'], wheel_width=.12,
        front_overhang=profile['front']-profile['wheel_base'], rear_overhang=-profile['rear'],
        left_overhang=profile['left']-profile['wheel_tread']/2,
        right_overhang=-profile['right']-profile['wheel_tread']/2,
        vehicle_height=profile['vehicle_height'], max_steer_angle=profile['max_steer_angle'])


def _load_work_geometry(profile, work_geometry):
    """One source-identity check for the offline and straight-drive consumers."""
    path = Path(work_geometry)
    content = path.read_bytes()
    model = json.loads(content)
    if model.get('schema_version') != 1 or model.get('reference_point') != 'rear_axle_ground':
        raise ValueError('unsupported work geometry')
    if model.get('geometry_manifest_sha256') != profile['source_hashes']['geometry.json']:
        raise ValueError('work geometry differs from navigation asset')
    if not model.get('hard_parts') or not model.get('bristle_parts'):
        raise ValueError('work geometry requires hard and bristle envelopes')
    return model, hashlib.sha256(content).hexdigest()


def coverage_vehicle(profile, *, work_geometry=None):
    """Optional source-bound CAD collision layers; never infer brush contacts."""
    from dg202604.qloop.coverage_geometry import VehicleGeometry
    vehicle = VehicleGeometry(profile['wheel_base'], profile['front'], profile['rear'],
        profile['left'], profile['right'], (), profile['max_steer_angle'],
        profile['steering_rate_rad_s'], profile['steering_rate_rad_s']*4,
        profile['steering_time_constant_s'], profile['source_hashes'], 'front_steered_tricycle')
    if work_geometry is None:
        return vehicle
    from dataclasses import replace
    model, source_hash = _load_work_geometry(profile, work_geometry)
    return replace(vehicle, hard_parts=tuple(model['hard_parts']), bristle_parts=tuple(model['bristle_parts']),
                   source_hashes=dict(vehicle.source_hashes, work_geometry=source_hash))


def load_straight_collision_layers(profile, work_geometry):
    """Project source-bound component shapes onto Y/Z without a geometry library.

    Coordinates are base_link metres at the rear axle ground reference. These
    ranges cover straight translation; the caller retains its forward stop gap.
    """
    model, source_hash = _load_work_geometry(profile, work_geometry)
    layers = dict(reference_point='rear_axle_ground',
        source_hashes=dict(profile['source_hashes'], work_geometry=source_hash))
    ids = set()
    for name in ('hard_parts', 'bristle_parts'):
        layers[name] = []
        for part in model[name]:
            part_id = part.get('id')
            if not isinstance(part_id, str) or not part_id or part_id in ids:
                raise ValueError('collision part requires a unique id')
            ids.add(part_id)
            x, y, yaw = part['x'], part['y'], part.get('yaw', 0.)
            low, high = part['z_min_m'], part['z_max_m']
            if not np.isfinite([x, y, yaw, low, high]).all() or low > high:
                raise ValueError('finite collision part coordinates and height interval required')
            if part['shape'] == 'circle':
                radius = part['radius']
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError('positive finite circle radius required')
                y_min, y_max = y-radius, y+radius
            elif part['shape'] == 'rectangle':
                length, width = part['length'], part['width']
                if not np.isfinite([length, width]).all() or min(length, width) <= 0:
                    raise ValueError('positive finite rectangle dimensions required')
                half = (abs(math.sin(yaw))*length+abs(math.cos(yaw))*width)/2
                y_min, y_max = y-half, y+half
            elif part['shape'] == 'polygon':
                vertices = np.asarray(part['vertices'], dtype=float)
                if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
                        or not np.isfinite(vertices).all()):
                    raise ValueError('finite polygon vertices required')
                shifted = vertices-np.roll(vertices, 1, axis=0)
                area_twice = np.sum(shifted[:, 0]*(vertices[:, 1]+np.roll(vertices[:, 1], 1)))
                if area_twice == 0:
                    raise ValueError('nondegenerate polygon required')
                ys = math.sin(yaw)*vertices[:, 0]+math.cos(yaw)*vertices[:, 1]+y
                y_min, y_max = float(ys.min()), float(ys.max())
            else:
                raise ValueError('unsupported shape: '+str(part['shape']))
            layers[name].append(dict(id=part_id, y_min_m=float(y_min), y_max_m=float(y_max),
                                     z_min_m=float(low), z_max_m=float(high)))
    return layers


def layered_obstacle_points(points, layers, *, margin_m=.05, curb_contact=None):
    """Retain component collision candidates for the explicit straight CARLA mode.

    A caller-validated curb contact only exempts the named outer bristle tip.
    Hard parts and all other bristles retain priority. This does not establish
    free space, boundary validity, or permission to activate the brushes.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('finite Nx3 obstacle points required')
    if isinstance(margin_m, bool) or not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError('nonnegative finite collision margin required')
    contact_id = None
    if curb_contact is not None:
        if not isinstance(curb_contact, dict) or curb_contact.get('side') not in ('left', 'right'):
            raise ValueError('curb contact requires an explicit left or right side')
        depth, ground, height = (curb_contact[k] for k in ('max_depth_m', 'ground_z_m', 'max_height_m'))
        if (any(isinstance(v, bool) for v in (depth, ground, height))
                or not np.isfinite([depth, ground, height]).all() or depth < 0 or height <= 0):
            raise ValueError('finite curb contact depth, ground and positive height required')
        contact_id = 'outer_'+curb_contact['side']
    y, z = points[:, 1], points[:, 2]
    retained = np.zeros(len(points), dtype=bool)
    for name in ('hard_parts', 'bristle_parts'):
        for part in layers[name]:
            hit = ((y >= part['y_min_m']-margin_m) & (y <= part['y_max_m']+margin_m)
                   & (z >= part['z_min_m']-margin_m) & (z <= part['z_max_m']+margin_m))
            if name == 'bristle_parts' and part['id'] == contact_id:
                tip = y >= part['y_max_m']-depth if curb_contact['side'] == 'left' else y <= part['y_min_m']+depth
                permitted = tip & (z >= ground) & (z <= ground+height)
                hit &= ~permitted
            retained |= hit
    return points[retained]


@dataclass
class ReferencePath:
    """One direction only. Direction switches are separated by a confirmed stop."""
    poses: np.ndarray
    direction: int
    profile: dict

    def __post_init__(self):
        self.poses = np.asarray(self.poses, dtype=float).copy()
        if self.direction not in (-1, 1) or self.poses.ndim != 2 or self.poses.shape[1] != 3 or len(self.poses) < 3:
            raise ValueError('a reference needs >=3 body poses and a direction')
        if not np.isfinite(self.poses).all():
            raise ValueError('non-finite reference')
        delta = np.diff(self.poses[:, :2], axis=0)
        ds = np.linalg.norm(delta, axis=1)
        if np.any(ds < 1e-6) or np.any(ds > .25):
            raise ValueError('reference spacing must be in [1e-6, 0.25] m')
        yaw = np.unwrap(self.poses[:, 2])
        tangent = np.arctan2(delta[:, 1], delta[:, 0])
        expected = (yaw[:-1]+yaw[1:])/2+(math.pi if self.direction < 0 else 0)
        if np.max(abs((tangent-expected+math.pi) % (2*math.pi)-math.pi)) > .10:
            raise ValueError('reference moves sideways or disagrees with its gear')
        kappa = np.diff(yaw)/ds/self.direction
        self.curvature = np.r_[kappa, kappa[-1]]
        self.steering = np.arctan(self.curvature*self.profile['wheel_base'])
        if np.max(abs(self.steering)) > self.profile['max_steer_angle']+1e-4:
            raise ValueError('reference exceeds the three-wheel steering limit')
        self.s = np.r_[0., np.cumsum(ds)]
        limit = self.profile['max_speed_mps'] if self.direction > 0 else self.profile['reverse_speed_mps']
        speed = np.minimum(limit, np.sqrt(self.profile['lateral_acceleration_mps2']/np.maximum(abs(self.curvature), 1e-8)))
        rate_per_m = abs(np.diff(self.steering))/ds
        rate_speed = self.profile['steering_rate_rad_s']/np.maximum(rate_per_m, 1e-8)
        speed[:-1] = np.minimum(speed[:-1], rate_speed)
        speed[1:] = np.minimum(speed[1:], rate_speed)
        speed[-1] = 0.
        for i in range(len(speed)-2, -1, -1):
            speed[i] = min(speed[i], math.sqrt(speed[i+1]**2+2*self.profile['deceleration_mps2']*ds[i]))
        for i in range(1, len(speed)):
            speed[i] = min(speed[i], math.sqrt(speed[i-1]**2+2*self.profile['acceleration_mps2']*ds[i-1]))
        self.speed = speed*self.direction
        # Signed acceleration of the reference; control acceleration uses the
        # PID's documented pedal convention instead (handled by ActuationGate).
        self.acceleration = np.r_[self.direction*np.diff(speed**2)/(2*ds), 0.]

    def project(self, pose, previous_s=0., search_distance=2.):
        if not np.isfinite(pose).all() or not math.isfinite(previous_s):
            raise ValueError('invalid projection input')
        lo = max(0, int(np.searchsorted(self.s, previous_s-.2))-1)
        hi = min(len(self.s)-1, int(np.searchsorted(self.s, previous_s+search_distance))+1)
        a, b = self.poses[lo:hi, :2], self.poses[lo+1:hi+1, :2]
        ab = b-a
        f = np.clip(np.sum((np.asarray(pose[:2])-a)*ab, axis=1)/np.sum(ab*ab, axis=1), 0, 1)
        points = a+f[:, None]*ab
        index = int(np.argmin(np.linalg.norm(points-np.asarray(pose[:2]), axis=1)))
        s = self.s[lo+index]+f[index]*(self.s[lo+index+1]-self.s[lo+index])
        return float(s), float(np.linalg.norm(points[index]-pose[:2]))

    def goal_reached(self, pose):
        return (math.dist(pose[:2], self.poses[-1, :2]) <= self.profile['goal_position_tolerance_m']
                and abs(wrap(pose[2]-self.poses[-1, 2])) <= self.profile['goal_yaw_tolerance_rad'])

    def velocity_profile(self, speed_limit, stop=False):
        if not math.isfinite(speed_limit) or speed_limit < 0:
            raise ValueError('invalid speed limit')
        magnitude = np.minimum(abs(self.speed), speed_limit) if not stop else np.zeros(len(self.speed))
        acceleration = np.r_[self.direction*np.diff(magnitude**2)/(2*np.diff(self.s)), 0.]
        return magnitude*self.direction, acceleration


def curb_follow_reference(segments, current_pose, origin, profile, *, brush, side,
                          mission_distance):
    """Straight CARLA guide: fit the visible front curb and offset by the brush."""
    result = dict(path=None, reason='no_front_curb', source_ns=None,
                  valid_until_ns=None, segment_id=None, sigma_lateral_m=None)
    if abs(wrap(current_pose[2]-origin[2])) > math.radians(10):
        return dict(result, reason='current_heading_exceeds_10deg')
    tangent = np.array([math.cos(origin[2]), math.sin(origin[2])])
    normal = np.array([-tangent[1], tangent[0]])
    current_xy, origin_xy = np.asarray(current_pose[:2]), np.asarray(origin[:2])
    progress = float((current_xy-origin_xy) @ tangent)
    rotation = np.array([[math.cos(current_pose[2]), -math.sin(current_pose[2])],
                         [math.sin(current_pose[2]), math.cos(current_pose[2])]])
    best, best_span = None, 0.
    for segment in segments:
        points = np.asarray(segment['points_base_m'], dtype=float)
        sigma = np.asarray(segment['sigma_lateral_m'], dtype=float)
        xy = points[:, :2] @ rotation.T+current_xy-origin_xy
        along, lateral = xy @ tangent, xy @ normal
        span = float(np.ptp(along))
        if points[:, 0].max() < brush['x']+.1 or span < .15:
            continue
        center = along.mean()
        design = np.column_stack((np.ones(len(along)), along-center))
        weight = 1/np.maximum(sigma, .005)
        intercept, slope = np.linalg.lstsq(design*weight[:, None], lateral*weight, rcond=None)[0]
        intercept -= slope*center
        angle = math.atan(slope)
        if abs(angle) > math.radians(10):
            result['reason'] = 'curb_heading_exceeds_10deg'
            continue
        offset = (brush['y']+(1 if side == 'left' else -1)*brush['radius'])/math.cos(angle)
        lo = max(0., min(progress-.2, mission_distance-.2))
        hi = min(progress+3., mission_distance)
        sample = np.linspace(lo, hi, max(3, math.ceil((hi-lo)/math.cos(angle)/.05)+1))
        guide_xy = origin_xy+sample[:, None]*tangent+(intercept+slope*sample-offset)[:, None]*normal
        path = ReferencePath(np.column_stack((guide_xy, np.full(len(sample), origin[2]+angle))), 1, profile)
        if path.project(current_pose)[1] > .35:
            result['reason'] = 'current_lateral_correction_exceeds_0.35m'
            continue
        if best is None or (segment['source_ns'], span) > (best['source_ns'], best_span):
            best_span = span
            best = dict(path=path, reason='forward_curb_guide',
                source_ns=segment['source_ns'], valid_until_ns=segment['valid_until_ns'],
                segment_id=segment['segment_id'], sigma_lateral_m=np.full(len(sample), np.median(sigma)))
    return best if best is not None else result


def trajectory_message(path, stamp, speed_limit, *, stop=False, frame_id='map'):
    from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
    message = Trajectory()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    velocities, accelerations = path.velocity_profile(speed_limit, stop)
    for row, velocity, acceleration in zip(path.poses, velocities, accelerations):
        point = TrajectoryPoint()
        point.pose.position.x, point.pose.position.y = float(row[0]), float(row[1])
        point.pose.orientation.z = math.sin(row[2]/2)
        point.pose.orientation.w = math.cos(row[2]/2)
        point.longitudinal_velocity_mps = float(velocity)
        point.acceleration_mps2 = float(acceleration)
        message.points.append(point)
    return message


def connect_poses(start, goal, profile, *, direction=1, checker=None):
    """Bounded quintic connections between route poses, checked after sampling.

    No global search or implicit free-space claim. A caller must supply its
    permitted corridor via checker for operational use; unverified references
    are useful for isolated controller tests only.
    """
    if direction not in (-1, 1) or not np.isfinite([*start, *goal]).all():
        raise ValueError('invalid poses or direction')
    distance = math.dist(start[:2], goal[:2])
    if distance < .1:
        raise ValueError('connection too short; no in-place rotation for this vehicle')
    candidates = []
    for scale in (.75, 1., 1.5, 2.):
        p0, p1 = np.asarray(start[:2]), np.asarray(goal[:2])
        v0 = direction*scale*distance*np.array([math.cos(start[2]), math.sin(start[2])])
        v1 = direction*scale*distance*np.array([math.cos(goal[2]), math.sin(goal[2])])
        # p(0), p(1), first derivatives, and zero endpoint second derivatives.
        c0, c1 = p0, v0
        q, dv = p1-p0-v0, v1-v0
        c3, c4, c5 = 10*q-4*dv, -15*q+7*dv, 6*q-3*dv
        t = np.linspace(0., 1., max(101, math.ceil(distance*scale/.015)))[:, None]
        xy = c0+c1*t+c3*t**3+c4*t**4+c5*t**5
        derivative = c1+3*c3*t**2+4*c4*t**3+5*c5*t**4
        if np.min(np.linalg.norm(derivative, axis=1)) < .01:
            continue
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        sample = np.linspace(0., lengths[-1], max(3, math.ceil(lengths[-1]/.05)+1))
        yaw = np.unwrap(np.arctan2(derivative[:, 1], derivative[:, 0])+(math.pi if direction < 0 else 0.))
        poses = np.column_stack([np.interp(sample, lengths, xy[:, 0]),
                                 np.interp(sample, lengths, xy[:, 1]), np.interp(sample, lengths, yaw)])
        try:
            path = ReferencePath(poses, direction, profile)
        except ValueError:
            continue
        if checker is None or checker.path_reason(path.poses.tolist()) is None:
            candidates.append(path)
    if not candidates:
        raise ValueError('no feasible bounded connection; keep stopped or replan the route')
    return min(candidates, key=lambda path: path.s[-1])


class MotionSequence:
    """Feedback-driven segment execution; no ROS or actuator publisher."""
    def __init__(self, paths, profile):
        if not paths:
            raise ValueError('empty motion sequence')
        for a, b in zip(paths, paths[1:]):
            if math.dist(a.poses[-1, :2], b.poses[0, :2]) > .01 or abs(wrap(a.poses[-1, 2]-b.poses[0, 2])) > .01:
                raise ValueError('motion segments must meet at the same body pose')
        self.paths, self.profile = paths, profile
        self.index, self.progress = 0, 0.
        self.state, self.reason = 'ALIGNING', 'initial_steering'
        self.stopped_since = None
        self.last_stamp = None

    @property
    def path(self):
        return self.paths[self.index]

    def update(self, *, now, stamp, pose, speed, steering, clear, localization_ok, resume=False):
        good = (fresh(now, stamp, self.profile['feedback_max_age_s'])
                and np.isfinite([*pose, speed, steering]).all())
        if not good or not localization_ok:
            self.state, self.reason = 'FAULT', 'invalid_feedback_or_localization'
            self.stopped_since = None
            return self.state
        if self.last_stamp is not None and stamp <= self.last_stamp:
            if stamp < self.last_stamp:
                self.state, self.reason = 'FAULT', 'feedback_clock_reset'
                self.stopped_since = None
            elif not clear and self.state != 'FAULT':
                # A new obstacle can arrive between two vehicle samples.
                self.state, self.reason = 'WAITING', 'path_not_clear'
            return self.state
        self.last_stamp = stamp
        if abs(speed) < self.profile['stop_speed_mps']:
            if self.stopped_since is None:
                self.stopped_since = stamp
        else:
            self.stopped_since = None
        stopped = self.stopped_since is not None and stamp-self.stopped_since >= self.profile['stop_dwell_s']-1e-6
        if not clear:
            self.state, self.reason = 'WAITING', 'path_not_clear'
            return self.state
        if self.state in ('FAULT', 'WAITING'):
            if not resume or not stopped:
                return self.state
            self.state, self.reason = 'ALIGNING', 'explicit_resume'
        if self.state == 'COMPLETED':
            return self.state
        s, error = self.path.project(pose, self.progress)
        if error > .35:
            self.state, self.reason = 'FAULT', 'tracking_error'
            return self.state
        self.progress = max(self.progress, s)
        if self.state == 'ALIGNING':
            i = min(len(self.path.steering)-1, int(np.searchsorted(self.path.s, self.progress)))
            if stopped and abs(steering-self.path.steering[i]) <= self.profile['steering_convergence_rad']:
                self.state, self.reason = 'DRIVING', 'steering_ready'
            return self.state
        if self.path.goal_reached(pose) and stopped:
            if self.index == len(self.paths)-1:
                self.state, self.reason = 'COMPLETED', 'goal_pose_and_stable_stop'
            else:
                self.index += 1
                self.progress = 0.
                self.state, self.reason = 'ALIGNING', 'next_segment'
                self.stopped_since = None
        elif self.path.s[-1]-self.progress < .6:
            self.state, self.reason = 'STOPPING', 'segment_end'
        return self.state


class ActuationGate:
    """Preserve steering while braking; shift only after fresh, stable feedback."""
    def __init__(self, profile):
        self.profile = profile
        self.last_stamp = None
        self.stop_since = None
        self.last_angle = None
        self.last_now = None

    def command(self, *, now, feedback_stamp, speed, measured_angle, feedback_reverse,
                target_speed, angle, acceleration, emergency=False, hold=False):
        p = self.profile
        valid = (fresh(now, feedback_stamp, p['feedback_max_age_s'])
                 and np.isfinite([speed, measured_angle, target_speed, angle, acceleration]).all())
        result = dict(throttle=0., brake=1., steer=0., reverse=bool(feedback_reverse), state='EMERGENCY')
        if not valid:
            self.stop_since = None
            # Retain the last finite steering command, not a fabricated reading.
            if self.last_angle is not None:
                result['steer'] = -self.last_angle/p['max_steer_angle']
            return result
        if self.last_stamp is not None and feedback_stamp < self.last_stamp:
            self.stop_since = None
            emergency = True
        if self.last_stamp is None or feedback_stamp > self.last_stamp:
            self.last_stamp = feedback_stamp
            if abs(speed) < p['stop_speed_mps']:
                if self.stop_since is None:
                    self.stop_since = feedback_stamp
            else:
                self.stop_since = None
        dt = 0. if self.last_now is None else max(0., min(now-self.last_now, .2))
        self.last_now = now
        previous = measured_angle if self.last_angle is None else self.last_angle
        desired = measured_angle if emergency or hold else angle
        limited = float(np.clip(desired, -p['max_steer_angle'], p['max_steer_angle']))
        if not emergency:
            limited = float(np.clip(limited, previous-p['steering_rate_rad_s']*dt,
                                    previous+p['steering_rate_rad_s']*dt))
        self.last_angle = limited
        result['steer'] = -limited/p['max_steer_angle']
        if emergency:
            return result
        if hold:
            result['state'] = 'HOLD'
            return result
        requested_reverse = target_speed < 0 if abs(target_speed) > 1e-6 else feedback_reverse
        if requested_reverse != feedback_reverse:
            result['state'] = 'STOP_FOR_GEAR'
            if self.stop_since is not None and feedback_stamp-self.stop_since >= p['stop_dwell_s']-1e-6:
                result.update(reverse=bool(requested_reverse), state='WAIT_GEAR_FEEDBACK')
            return result
        if abs(target_speed) > (p['reverse_speed_mps'] if requested_reverse else p['max_speed_mps'])+1e-6:
            return result
        if abs(speed) < p['stop_speed_mps'] and abs(target_speed) < 1e-6:
            result['state'] = 'HOLD'
            return result
        if abs(target_speed) < 1e-6:
            acceleration = min(acceleration, -p['deceleration_mps2'])
        # Same CARLA pedal map as the existing bounded loop, now behind the
        # steering/gear gate. It remains an uncalibrated simulation adapter.
        result.update(brake=0., state='DRIVE' if acceleration >= 0 else 'BRAKING')
        if acceleration >= 0:
            result['throttle'] = float(np.clip(.04+.25*acceleration, 0, .30))
        else:
            result['brake'] = float(np.clip(-.25*acceleration, 0, 1))
        return result
