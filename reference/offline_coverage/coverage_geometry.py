"""Native V2 geometry and bounded offline counterphase 4WS model.

Mission coordinates: x forward, y left, yaw CCW, axle midpoint reference.
This is a kinematic research model, not a physics or cleaning-effect model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import yaml
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def transform(pose, point):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * point[0] - s * point[1], y + s * point[0] + c * point[1])


def mission_to_unity(pose, world=(0.0, 0.0, 0.0)):
    """Return Unity x,z,clockwise yaw with an optional Unity world origin."""
    x, z, yaw = -pose[1], pose[0], -pose[2]
    c, s = math.cos(world[2]), math.sin(world[2])
    return world[0] + c*x + s*z, world[1] - s*x + c*z, yaw + world[2]


def unity_to_mission(pose, world=(0.0, 0.0, 0.0)):
    dx, dz = pose[0]-world[0], pose[1]-world[1]
    c, s = math.cos(world[2]), math.sin(world[2])
    return s*dx+c*dz, -(c*dx-s*dz), -(pose[2]-world[2])


@dataclass(frozen=True)
class VehicleGeometry:
    wheelbase: float
    front: float
    rear: float
    left: float
    right: float
    brushes: tuple
    axle_limit: float
    steer_rate: float
    steer_acceleration: float
    steer_tau: float
    source_hashes: dict

    @property
    def curvature_limit(self):
        return 2 * math.tan(self.axle_limit) / self.wheelbase

    @property
    def width(self):
        return self.left-self.right

    @property
    def coverage_width(self):
        lows, highs = [], []
        for b in self.brushes:
            r = b.get("radius", b.get("width", 0)/2)
            lows.append(b["y"]-r)
            highs.append(b["y"]+r)
        return max(highs)-min(lows)

    @property
    def radius(self):
        return math.hypot(max(abs(self.front), abs(self.rear)), max(abs(self.left), abs(self.right)))

    def resolved(self):
        return dict(asdict(self), reference_point="axle_midpoint", units="m, rad, s",
                    coverage_width=self.coverage_width, curvature_limit=self.curvature_limit,
                    evidence="native_simulation_configuration_not_physical_calibration")


def load_geometry(root, spec):
    """Fail closed on changed native geometry; old implementation hashes are history."""
    import json
    root = Path(root)
    paths = {
        "v2_profile": root / "simulation/awsim-autodrrt/config/cleaning-robot-v2-build-profile.json",
        "v2_autoware": root / "simulation/awsim-autodrrt/config/cleaning-robot-v2-autoware.param.yaml",
        "v2_prefab": root / "simulation/awsim-autodrrt/awsim-overlay/Assets/Dg202604/CleaningRobotV2/Prefabs/CleaningRobotEgoVehicle.prefab",
    }
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    for k, sha in hashes.items():
        if sha != spec["source_baseline"][k]["sha256"]:
            raise ValueError(f"native geometry changed: {k}; freeze a new specification")
    profile = json.loads(paths["v2_profile"].read_text(encoding="utf-8"))
    v = yaml.safe_load(paths["v2_autoware"].read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    prefab = paths["v2_prefab"].read_text(encoding="utf-8")
    def field(name):
        values = re.findall(r"^\s+" + re.escape(name) + r": ([0-9.eE+-]+)\s*$", prefab, re.M)
        if len(values) != 1:
            raise ValueError(f"ambiguous prefab field: {name}")
        return float(values[0])
    eq = math.radians(field("_maxSteerTireAngleInput"))
    axle = min(math.atan(math.tan(eq)/2), math.radians(field("_maxAxleSteerTireAngle")))
    brushes = []
    for name in ("mainBrush", "sideBrushLeft", "sideBrushRight"):
        c = profile[name+"Center"]
        b = dict(id=name, x=c["z"], y=-c["x"])
        if name == "mainBrush":
            b.update(shape="rectangle", width=profile["mainBrushFootprintSize"]["x"],
                     length=profile["mainBrushFootprintSize"]["y"])
        else:
            b.update(shape="circle", radius=profile[name+"Radius"])
        brushes.append(b)
    for i, (c, r) in enumerate(zip(profile["innerBrushCenters"], profile["innerBrushRadii"])):
        brushes.append(dict(id=f"innerBrush{i}", x=c["z"], y=-c["x"], shape="circle", radius=r))
    if abs(v["wheel_base"]-profile["wheelBase"]) > 1e-9:
        raise ValueError("wheelbase sources disagree")
    return VehicleGeometry(v["wheel_base"], v["wheel_base"]/2+v["front_overhang"],
        -v["wheel_base"]/2-v["rear_overhang"], v["wheel_tread"]/2+v["left_overhang"],
        -v["wheel_tread"]/2-v["right_overhang"], tuple(brushes), axle,
        math.radians(field("_maxSteerRateDegreesPerSecond")),
        math.radians(field("_maxSteerAccelerationDegreesPerSecondSquared")),
        field("_steerTireAngleTimeConstant"), hashes)


def shape_at(shape, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    if shape["shape"] == "circle":
        # Circumscribed polygon: circle approximation cannot miss a collision.
        geom = Point(shape["x"], shape["y"]).buffer(shape["radius"]/math.cos(math.pi/128), resolution=32)
    else:
        geom = box(shape["x"]-shape["length"]/2, shape["y"]-shape["width"]/2,
                   shape["x"]+shape["length"]/2, shape["y"]+shape["width"]/2)
    return affine_transform(geom, [c, -s, s, c, pose[0], pose[1]])


class GeometryChecker:
    """Continuous-segment conservative bound; independent of coverage raster."""
    def __init__(self, scene, vehicle, margin=0.05):
        self.vehicle = vehicle
        self.margin = margin
        self.allowed = Polygon(scene["allowed"])
        self.obstacles = [Polygon(p) for p in scene.get("obstacles", [])]
        if not self.allowed.is_valid or any(not p.is_valid for p in self.obstacles):
            raise ValueError("invalid collision polygon")
        self.free = prep(self.allowed)
        self.rectangular_allowed = self.allowed.equals(box(*self.allowed.bounds))
        self.allowed_bounds = self.allowed.bounds
        self.body = box(vehicle.rear, vehicle.right, vehicle.front, vehicle.left)
        # Native brushes must all be inside the conservative envelope used below.
        for b in vehicle.brushes:
            if not self.body.covers(shape_at(b, (0, 0, 0))):
                raise ValueError("tool outside collision envelope; update checker explicitly")

    def pose_reason(self, pose, extra=0.0):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        if self.rectangular_allowed and not self.obstacles:
            m = self.margin+extra
            v = self.vehicle
            xmin, ymin, xmax, ymax = self.allowed_bounds
            for x in (v.rear-m, v.front+m):
                for y in (v.right-m, v.left+m):
                    px, py = pose[0]+c*x-s*y, pose[1]+s*x+c*y
                    if px < xmin or px > xmax or py < ymin or py > ymax:
                        return "outside_allowed"
            return None
        body = affine_transform(self.body, [c, -s, s, c, pose[0], pose[1]])
        body = body.buffer(self.margin+extra, resolution=8, join_style=2)
        if not self.free.covers(body):
            return "outside_allowed"
        if any(body.intersects(p) for p in self.obstacles):
            return "collision_obstacle"
        return None

    def segment_reason(self, a, b, step=0.025):
        dyaw = wrap(b[2]-a[2])
        bound = math.dist(a[:2], b[:2])+self.vehicle.radius*abs(dyaw)
        n = max(1, math.ceil(bound/step))
        # Every in-between point is within half a sample's displacement bound.
        for i in range(n+1):
            f = i/n
            p = (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]), a[2]+f*dyaw)
            reason = self.pose_reason(p, bound/(2*n))
            if reason:
                return reason
        return None

    def path_reason(self, poses):
        if not poses:
            return "empty_path"
        reason = self.pose_reason(poses[0])
        if reason:
            return reason
        if self.rectangular_allowed and not self.obstacles and len(poses) > 1:
            # Same subdivisions and inflation as segment_reason, evaluated in
            # arrays. This keeps the collision gate while avoiding Python work
            # for every corner of every dense Fields2Cover candidate.
            points = np.asarray(poses, dtype=float)
            a, b = points[:-1], points[1:]
            dyaw = (b[:, 2]-a[:, 2]+math.pi) % (2*math.pi)-math.pi
            bound = np.linalg.norm(b[:, :2]-a[:, :2], axis=1)+self.vehicle.radius*abs(dyaw)
            n = np.maximum(1, np.ceil(bound/.025).astype(int))
            ids = np.repeat(np.arange(len(a)), n+1)
            starts = np.repeat(np.r_[0, np.cumsum(n+1)[:-1]], n+1)
            f = (np.arange(len(ids))-starts)/n[ids]
            x = a[ids, 0]+f*(b[ids, 0]-a[ids, 0])
            y = a[ids, 1]+f*(b[ids, 1]-a[ids, 1])
            yaw = a[ids, 2]+f*dyaw[ids]
            c, s = np.cos(yaw), np.sin(yaw)
            v = self.vehicle
            inflation = self.margin+bound[ids]/(2*n[ids])
            cx = x+c*(v.front+v.rear)/2-s*(v.left+v.right)/2
            cy = y+s*(v.front+v.rear)/2+c*(v.left+v.right)/2
            ex = abs(c)*((v.front-v.rear)/2+inflation)+abs(s)*((v.left-v.right)/2+inflation)
            ey = abs(s)*((v.front-v.rear)/2+inflation)+abs(c)*((v.left-v.right)/2+inflation)
            xmin, ymin, xmax, ymax = self.allowed_bounds
            bad = np.flatnonzero((cx-ex < xmin) | (cx+ex > xmax) | (cy-ey < ymin) | (cy+ey > ymax))
            return f"outside_allowed:segment={int(ids[bad[0]])}" if len(bad) else None
        for i, (a, b) in enumerate(zip(poses, poses[1:])):
            reason = self.segment_reason(a, b)
            if reason:
                return f"{reason}:segment={i}"
        return None


@dataclass
class KinematicState:
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    axle_angle: float = 0.0
    axle_rate: float = 0.0

    @property
    def pose(self):
        return self.x, self.y, self.yaw

    def step(self, speed_command, curvature_command, dt, vehicle, acceleration):
        if dt <= 0 or acceleration <= 0:
            raise ValueError("positive dt and acceleration required")
        target = np.clip(math.atan(curvature_command*vehicle.wheelbase/2), -vehicle.axle_limit, vehicle.axle_limit)
        error = target-self.axle_angle
        acceleration_step = vehicle.steer_acceleration*dt
        braking_rate = math.sqrt(2*vehicle.steer_acceleration*abs(error)+acceleration_step**2)-acceleration_step
        desired_rate = math.copysign(min(vehicle.steer_rate, braking_rate, abs(error)/max(vehicle.steer_tau, dt)), error)
        self.axle_rate += float(np.clip(desired_rate-self.axle_rate, -vehicle.steer_acceleration*dt, vehicle.steer_acceleration*dt))
        self.axle_angle += self.axle_rate*dt
        # Saturation with rate braking before the hard limit avoids unbounded reset.
        self.axle_angle = float(np.clip(self.axle_angle, -vehicle.axle_limit, vehicle.axle_limit))
        self.speed += float(np.clip(speed_command-self.speed, -acceleration*dt, acceleration*dt))
        omega = self.speed*2*math.tan(self.axle_angle)/vehicle.wheelbase
        mid = self.yaw+omega*dt/2
        self.x += self.speed*math.cos(mid)*dt
        self.y += self.speed*math.sin(mid)*dt
        self.yaw = wrap(self.yaw+omega*dt)


def evaluator_sample(vehicle, pose, t, speed, effective, phase="work", collision_count=0):
    x, z, yaw = mission_to_unity(pose)
    brushes = []
    for b in vehicle.brushes:
        bx, by = transform(pose, (b["x"], b["y"]))
        out = {k: v for k, v in b.items() if k not in ("x", "y")}
        out.update(x=-by, z=bx, yaw=yaw, effective=bool(effective and speed >= 0),
                   rotation_delta_deg=1.0 if effective else 0.0, lift_error_m=0.0 if effective else 0.1)
        brushes.append(out)
    cx, cy = transform(pose, ((vehicle.front+vehicle.rear)/2, (vehicle.left+vehicle.right)/2))
    return dict(t=float(t), x=x, z=z, yaw=yaw, speed=float(speed), phase=phase,
                collision_count=collision_count, brushes=brushes,
                body=[dict(x=-cy, z=cx, yaw=yaw, width=vehicle.width, length=vehicle.front-vehicle.rear)])


def evaluator_task(scene, parameters, resolution=None):
    def convert(poly):
        return [[-p[1], p[0]] for p in poly]
    return dict(schema_version=2, task_id=scene["id"], target=convert(scene["target"]),
        target_holes=[convert(p) for p in scene.get("target_holes", [])],
        allowed=convert(scene["allowed"]), forbidden=[convert(p) for p in scene.get("obstacles", [])],
        temporary_obstacles=[], resolution_m=resolution or parameters["coverage_grid_m"],
        edge_width_m=parameters["edge_band_m"], corner_size_m=parameters["corner_radius_m"],
        revisit_radius_m=parameters["revisit_radius_m"],
        metadata={"collision_source": "offline_conservative_geometry_check",
                  "trace_source": "bounded_counterphase_4ws_kinematic_model",
                  "quality_truth": "none"})
