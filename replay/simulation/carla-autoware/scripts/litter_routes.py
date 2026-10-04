"""Tool-directed task routes over numeric target geometry."""
from dataclasses import dataclass
import math

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon

from coverage_boustrophedon import plan_boustrophedon
from tricycle_navigation import ReferencePath, connect_poses, wrap


@dataclass(frozen=True)
class RouteConfig:
    profile: dict
    swath_m: float
    tool_reach_m: float


@dataclass
class TaskRoute:
    poses: np.ndarray
    order: np.ndarray
    pending: np.ndarray


class _Corridor:
    def __init__(self, polygon, profile):
        self.polygon = Polygon(polygon).buffer(1e-7)
        self.body = np.asarray([[x, y] for x in (profile['rear'], profile['front'])
                                for y in (profile['right'], profile['left'])])[[0, 2, 3, 1]]

    def path_reason(self, poses):
        for x, y, yaw in poses:
            c, s = math.cos(yaw), math.sin(yaw)
            body = self.body @ np.asarray([[c, s], [-s, c]]) + [x, y]
            if not self.polygon.covers(Polygon(body)):
                return 'outside_task_corridor'
        return None


def _time_and_reference(poses, profile):
    directions = poses[:-1, 3]
    breaks = np.r_[0, np.flatnonzero(np.diff(directions)) + 1, len(directions)]
    seconds = 0.
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        part = poses[lo:hi + 1, :3]
        keep = np.r_[True, np.linalg.norm(np.diff(part[:, :2], axis=0), axis=1) > 1e-8]
        part = part[keep]
        if len(part) == 2:
            middle = (part[0] + part[1]) / 2
            middle[2] = part[0, 2] + wrap(part[1, 2] - part[0, 2]) / 2
            part = np.vstack([part[0], middle, part[1]])
        path = ReferencePath(part, int(directions[lo]), profile)
        seconds += float(np.sum(np.diff(path.s) / np.maximum(.05, np.abs(path.speed[:-1]))))
    return seconds + 2. * (len(breaks) - 2)


def _candidate(region, start, angle, corridor, config):
    shape = Point(region[0]).buffer(.05, cap_style=3) if len(region) == 1 else Polygon(region)
    c, s = math.cos(angle), math.sin(angle)
    lead = config.tool_reach_m
    # Extend both ends of each work row so forward-mounted tools reach the
    # target even when the next row is traversed in the opposite direction.
    extended = translate(shape, xoff=-lead*c, yoff=-lead*s).union(
        translate(shape, xoff=lead*c, yoff=lead*s)).convex_hull
    radius = config.profile['wheel_base'] / math.tan(config.profile['max_steer_angle'])
    raw = plan_boustrophedon(list(extended.exterior.coords)[:-1], start, config.swath_m,
                            turn_radius=radius, sweep_angle=angle)
    poses = np.asarray(raw['poses'], dtype=float)
    first = int(np.flatnonzero(poses[:, 4])[0])
    work = poses[first:]
    entry = work[0, :3]
    if np.linalg.norm(entry[:2] - start[:2]) < 1e-5 and abs(wrap(entry[2] - start[2])) < 1e-5:
        joined = work
    else:
        approach = connect_poses(start, entry, config.profile, checker=corridor).poses
        joined = np.vstack([np.column_stack([approach[:-1], np.ones(len(approach)-1),
                                             np.zeros(len(approach)-1)]), work])
    if corridor.path_reason(joined[:, :3]) is not None:
        return None
    return _time_and_reference(joined, config.profile), joined


def plan_litter_route(regions, start, allowed_polygon, config):
    """Return executable poses, visited target indices and unresolved indices."""
    assert np.shape(start) == (3,) and all(np.ndim(r) == 2 and np.shape(r)[1] == 2 for r in regions)
    corridor = _Corridor(allowed_polygon, config.profile)
    remaining = list(range(len(regions)))
    order, pieces = [], []
    here = np.asarray(start, dtype=float)
    while remaining:
        candidates = []
        for index in remaining:
            region = regions[index]
            angles = [here[2] % math.pi]
            if len(region) > 1:
                edges = np.roll(region, -1, axis=0) - region
                edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
                angle = math.atan2(edge[1], edge[0]) % math.pi
                if abs(wrap(angle - angles[0])) > 1e-5:
                    angles.append(angle)
            for angle in angles:
                try:
                    result = _candidate(region, here, angle, corridor, config)
                except (ValueError, IndexError):
                    continue
                if result is not None:
                    cost, poses = result
                    candidates.append((cost, index, poses))
        if not candidates:
            break
        _, index, poses = min(candidates, key=lambda c: (c[0], c[1]))
        if pieces:
            pieces[-1] = pieces[-1][:-1]
        pieces.append(poses)
        here = poses[-1, :3]
        order.append(index)
        remaining.remove(index)
    poses = np.vstack(pieces) if pieces else np.empty((0, 5))
    return TaskRoute(poses, np.asarray(order, dtype=int), np.asarray(remaining, dtype=int))
