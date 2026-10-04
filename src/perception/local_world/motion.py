"""Sensor-only local motion. Global map corrections never enter this estimator.

The planar speed/gyro prior is a simulation baseline, not a calibrated wheel
odometer. GICP is an optional scan-to-local-map experiment. Its Hessian is
reported as observability evidence, never relabelled as pose covariance.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np


def rigid(value):
    out = np.asarray(value, dtype=float)
    if (out.shape != (4, 4) or not np.isfinite(out).all()
            or not np.allclose(out[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(out[:3, :3].T @ out[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(out[:3, :3]), 1, atol=1e-6)):
        raise ValueError("expected a finite SE(3) transform")
    return out.copy()


def transform(points, target_from_source):
    points = np.asarray(points, dtype=float)
    matrix = rigid(target_from_source)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points).all():
        raise ValueError("expected finite Nx3+ points")
    out = points.copy()
    out[:, :3] = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return out


def planar(x=0., y=0., yaw=0.):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.]])


def twist_step(speed, yaw_rate, duration):
    """Exact constant body-twist integration, including stationary turns."""
    if not np.isfinite([speed, yaw_rate, duration]).all() or duration < 0:
        raise ValueError("invalid motion interval")
    angle = yaw_rate * duration
    if abs(angle) < 1e-8:
        x, y = speed * duration, .5 * speed * duration * angle
    else:
        x, y = speed * math.sin(angle) / yaw_rate, speed * (1 - math.cos(angle)) / yaw_rate
    return planar(x, y, angle)


class TwistSeries:
    """Source-time ZOH integration; refuses gaps and extrapolation past max_age."""
    def __init__(self, records, max_age_s=.25):
        entries = sorted((float(r['stamp_s']), float(r['linear'][0]), float(r['angular'][2]))
                         for r in records if r['kind'] == 'gyro_odometer' and r['frame_id'] == 'base_link')
        self.values = np.asarray(entries, dtype=float).reshape(-1, 3)
        self.max_age_s = max_age_s
        if (not len(self.values) or not np.isfinite(self.values).all()
                or np.any(np.diff(self.values[:, 0]) <= 0) or max_age_s <= 0):
            raise ValueError("twist input must have unique increasing finite source times")

    def relative(self, start, end):
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError("invalid integration interval")
        if end == start:
            return np.eye(4)
        times = self.values[:, 0]
        cursor, out = start, np.eye(4)
        index = int(np.searchsorted(times, cursor + 1e-9, side='right')) - 1
        if index < 0:
            raise ValueError("no twist before interval")
        while cursor < end - 1e-9:
            stop = min(end, times[index + 1] if index + 1 < len(times) else end)
            if stop - times[index] > self.max_age_s + 1e-8:
                raise ValueError("twist source gap: start a new motion epoch")
            out = out @ twist_step(*self.values[index, 1:], stop - cursor)
            cursor, index = stop, index + 1
        return out


@dataclass(frozen=True)
class MotionResult:
    odom_from_base: np.ndarray
    quality: str
    registration_accepted: bool
    inlier_fraction: float | None
    hessian_condition: float | None
    correction_m: float
    covariance_valid: bool = False


class LocalOdometry:
    """Bounded keyframe GICP map with independent speed/gyro initial guess.

    Thresholds are frozen experiment settings. Rejecting registration keeps
    the prior, marks DEGRADED, and does not contaminate the keyframe map.
    """
    def __init__(self, registration='GICP', voxel_m=.25, keyframes=8):
        if registration not in ('GICP', 'ICP', 'PRIOR') or voxel_m <= 0 or keyframes < 1:
            raise ValueError("invalid local odometry configuration")
        self.registration, self.voxel_m = registration, voxel_m
        self.keyframes = deque(maxlen=keyframes)
        self.pose, self.keypose, self.stamp = np.eye(4), np.eye(4), None
        self.target = None

    def update(self, points_base, stamp_s, relative_prior=None):
        points = np.asarray(points_base, dtype=float)
        if (points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points).all()
                or not math.isfinite(stamp_s)):
            raise ValueError("invalid local odometry input")
        points = points[:, :3]
        ranges = np.linalg.norm(points, axis=1)
        points = points[(ranges > 3) & (ranges < 35)]
        if len(points) < 100:
            raise ValueError("insufficient scan support")
        if self.stamp is not None and stamp_s <= self.stamp:
            raise ValueError("odometry source times must increase")
        if self.stamp is not None and relative_prior is None:
            raise ValueError("motion prior required after first scan")
        prior = self.pose @ rigid(relative_prior) if self.stamp is not None else np.eye(4)
        accepted, fraction, condition, correction = False, None, None, 0.
        quality = 'DEGRADED'
        pose = prior
        if self.registration != 'PRIOR':
            import small_gicp
            source, _ = small_gicp.preprocess_points(points, self.voxel_m, num_threads=4)
            if self.target is not None:
                result = small_gicp.align(self.target[0], source, self.target[1], prior,
                    registration_type=self.registration, max_correspondence_distance=.7,
                    num_threads=4, max_iterations=40, translation_epsilon=.0001,
                    rotation_epsilon=.0001)
                candidate = rigid(result.T_target_source)
                fraction = float(result.num_inliers / max(1, len(source.points())))
                eigen = np.linalg.eigvalsh(np.asarray(result.H))
                condition = float(eigen[-1] / max(eigen[0], 1e-12))
                delta = np.linalg.inv(prior) @ candidate
                correction = float(np.linalg.norm(delta[:3, 3]))
                angle = math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
                accepted = bool(result.converged and fraction >= .5 and eigen[0] > 0
                                and condition < 1e8 and correction <= .25 and angle <= math.radians(3))
                if accepted:
                    pose, quality = candidate, 'OK'
            else:
                quality = 'INITIALIZING'
            keydelta = np.linalg.inv(self.keypose) @ pose
            angle = math.acos(float(np.clip((np.trace(keydelta[:3, :3]) - 1) / 2, -1, 1)))
            if self.target is None or (accepted and (np.linalg.norm(keydelta[:3, 3]) > .5 or angle > .08)):
                self.keyframes.append(transform(source.points()[:, :3], pose))
                self.target = small_gicp.preprocess_points(np.concatenate(self.keyframes), self.voxel_m, num_threads=4)
                self.keypose = pose.copy()
        self.pose, self.stamp = pose.copy(), stamp_s
        return MotionResult(pose.copy(), quality, accepted, fraction, condition, correction)


class SweepBuffer:
    """Source scans transformed once into the current LiDAR frame.

    The fifth column is sweep age. There is no per-point deskew without actual
    per-point acquisition times. This buffer never creates free-space evidence.
    """
    def __init__(self, max_sweeps=5, max_age_s=1.):
        if max_sweeps < 1 or max_age_s <= 0:
            raise ValueError("invalid sweep bounds")
        self.scans = deque(maxlen=max_sweeps)
        self.max_age_s, self.identity = max_age_s, None

    def add(self, points_xyzi, stamp_s, odom_from_lidar, epoch, calibration):
        pose = rigid(odom_from_lidar)
        points = np.asarray(points_xyzi, dtype=float)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all() or not math.isfinite(stamp_s):
            raise ValueError("invalid sweep input")
        identity = (epoch, calibration)
        if self.identity != identity:
            self.scans.clear()
            self.identity = identity
        if self.scans and stamp_s <= self.scans[-1][0]:
            raise ValueError("duplicate or reversed sweep timestamp")
        self.scans.append((stamp_s, points.copy(), pose))
        while self.scans and stamp_s - self.scans[0][0] > self.max_age_s:
            self.scans.popleft()
        batches = []
        for stamp, scan, source_pose in reversed(self.scans):
            aligned = scan.copy() if stamp == stamp_s else transform(scan, np.linalg.inv(pose) @ source_pose)
            batches.append(np.column_stack([aligned, np.full(len(scan), stamp_s - stamp)]))
        return np.concatenate(batches), tuple(s[0] for s in self.scans)
