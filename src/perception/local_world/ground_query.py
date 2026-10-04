"""Indexed, immutable measured support; no extrapolation or ground classification."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import ConvexHull, QhullError


class GroundIndex:
    """XY buckets contain original point indices, including every height layer."""
    def __init__(self, points, cell_size_m=.35):
        self.points = points
        self.cell_size_m = cell_size_m
        if not len(points):
            self.order = np.empty(0, dtype=np.int64)
            self.keys = np.empty((0, 2), dtype=np.int64)
            self.buckets = {}
            return
        cells = np.floor(points[:, :2] / cell_size_m).astype(np.int64)
        self.order = np.lexsort((cells[:, 1], cells[:, 0]))
        ordered = cells[self.order]
        starts = np.r_[0, np.flatnonzero(np.any(np.diff(ordered, axis=0), axis=1)) + 1]
        self.keys = ordered[starts]
        ends = np.r_[starts[1:], len(points)]
        self.buckets = {tuple(key): (int(start), int(end))
                        for key, start, end in zip(self.keys, starts, ends)}

    def nearby(self, xy, radius_m):
        first = np.floor((xy-radius_m) / self.cell_size_m).astype(np.int64)
        last = np.floor((xy+radius_m) / self.cell_size_m).astype(np.int64)
        count = math.prod(int(b)-int(a)+1 for a, b in zip(first, last))
        if count <= len(self.buckets):
            spans = [self.buckets[(x, y)]
                     for x in range(int(first[0]), int(last[0])+1)
                     for y in range(int(first[1]), int(last[1])+1)
                     if (x, y) in self.buckets]
        else:
            keys = self.keys[np.all((self.keys >= first) & (self.keys <= last), axis=1)]
            spans = [self.buckets[tuple(key)] for key in keys]
        if not spans:
            return np.empty(0, dtype=np.int64), 0
        # Restore source order, so the index never changes a measured point set.
        candidates = np.sort(np.concatenate([self.order[start:end] for start, end in spans]))
        selected = np.linalg.norm(self.points[candidates, :2]-xy, axis=1) <= radius_m
        return candidates[selected], len(candidates)


@dataclass(frozen=True)
class GroundFit:
    points: np.ndarray
    anchor: np.ndarray
    equations: np.ndarray | None
    coefficients: np.ndarray | None
    max_residual_m: float | None
    reason: str

    @classmethod
    def build(cls, points):
        anchor = points[:, :2].mean(axis=0)
        if len(points) < 3:
            return cls(points, anchor, None, None, None, 'INSUFFICIENT_RETURNS')
        # A fixed local anchor lets different queries reuse the same fit without
        # losing precision when odom coordinates are far from the origin.
        xy = points[:, :2]-anchor
        try:
            hull = ConvexHull(xy)
        except QhullError:
            return cls(points, anchor, None, None, None, 'DEGENERATE_SUPPORT')
        design = np.column_stack((xy, np.ones(len(points))))
        coefficients = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
        residual = float(np.max(np.abs(design @ coefficients-points[:, 2])))
        reason = 'SUPPORTED' if residual <= .03 and np.linalg.norm(coefficients[:2]) <= .3 else 'ROUGH_OR_SLOPED'
        return cls(points, anchor, hull.equations, coefficients, residual, reason)

    def at(self, query, include_points=False):
        if self.equations is None:
            return {'supported': False, 'reason': self.reason}
        xy = query-self.anchor
        if np.any(self.equations[:, :2] @ xy+self.equations[:, 2] > 1e-8):
            return {'supported': False, 'reason': 'OUTSIDE_SUPPORT'}
        a, b, c = self.coefficients
        result = dict(supported=self.reason == 'SUPPORTED', reason=self.reason,
            z_m=float(c + a*xy[0] + b*xy[1]), plane_normal=[-float(a), -float(b), 1.],
            max_residual_m=self.max_residual_m)
        if include_points:
            result['support_points_odom'] = self.points.tolist()
        return result
