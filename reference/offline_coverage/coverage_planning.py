"""Fields2Cover v2.0.0 adapter and finite tool-aware candidate search.

All methods share native collision geometry. Upstream turning paths are checked
after conversion; a missing/unsafe connection is never replaced by a chord.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
from shapely.geometry import Polygon

from .coverage_geometry import GeometryChecker, wrap

CONNECTOR_OFFSETS_M = (1.5, -1.5, 3.0, -3.0, 4.5, -4.5, 6.0, -6.0)


def _point(p):
    return (p.getX(), p.getY())


def _line(a, b, step, work=True):
    yaw = math.atan2(b[1]-a[1], b[0]-a[0])
    n = max(1, math.ceil(math.dist(a, b)/step))
    return [[a[0]+(b[0]-a[0])*i/n, a[1]+(b[1]-a[1])*i/n, yaw, 1, work] for i in range(n+1)]


def _f2c_cell(f2c, polygon):
    ring = f2c.LinearRing()
    for x, y in list(polygon.exterior.coords):
        ring.addPoint(f2c.Point(x, y))
    cell = f2c.Cell(ring)
    for hole in polygon.interiors:
        ring = f2c.LinearRing()
        for x, y in hole.coords:
            ring.addPoint(f2c.Point(x, y))
        cell.addRing(ring)
    return cell


def _swaths(f2c, scene, width, angle):
    poly = Polygon(scene["target"], scene.get("target_holes", []))
    generated = f2c.SG_BruteForce().generateSwaths(angle, width, _f2c_cell(f2c, poly))
    ordered = f2c.RP_Boustrophedon().genSortedSwaths(generated)
    return [(_point(ordered.at(i).startPoint()), _point(ordered.at(i).endPoint())) for i in range(ordered.size())]


def _boundary(scene, inset):
    poly = Polygon(scene["target"], scene.get("target_holes", [])).buffer(-inset, join_style=2)
    polygons = [] if poly.is_empty else ([poly] if poly.geom_type == "Polygon" else list(poly.geoms))
    out = []
    for p in polygons:
        vertices = list(p.exterior.coords)
        out.extend((a, b) for a, b in zip(vertices, vertices[1:]) if math.dist(a, b) > .05)
    return out


def _feasible_boundary(scene, inset, checker):
    """Increase a wall-constrained inset without discarding the boundary pass."""
    bound = max(Polygon(scene["target"]).bounds[2]-Polygon(scene["target"]).bounds[0],
                Polygon(scene["target"]).bounds[3]-Polygon(scene["target"]).bounds[1])/2
    while inset < bound:
        lines = _boundary(scene, inset)
        if not lines:
            return None, inset
        if all(checker.segment_reason((*a, math.atan2(b[1]-a[1], b[0]-a[0])),
                                      (*b, math.atan2(b[1]-a[1], b[0]-a[0]))) is None for a, b in lines):
            return lines, inset
        inset += .05
    return None, inset


class F2CAdapter:
    def __init__(self, vehicle, parameters, checker):
        import fields2cover as f2c
        self.f2c, self.vehicle, self.parameters, self.checker = f2c, vehicle, parameters, checker
        self.cache = {}
        self.bounded_cache = {}
        self.planners = {}
        self.curve_cache = {}

    def _direct_connect(self, a, b, radius_factor):
        if math.dist(a[:2], b[:2]) < 1e-7 and abs(wrap(a[2]-b[2])) < 1e-7:
            return [list(a[:3])+[1, False]], None
        key = tuple(round(float(v), 7) for v in (*a[:3], *b[:3], radius_factor))
        if key in self.cache:
            return self.cache[key]
        f2c = self.f2c
        robot = f2c.Robot(self.vehicle.width, self.vehicle.coverage_width)
        robot.setMinTurningRadius(radius_factor/self.vehicle.curvature_limit)
        robot.setMaxDiffCurv(self.parameters["max_diff_curvature_per_m2"])
        robot.setCruiseVel(self.parameters["nominal_speed_mps"])
        reasons = []
        for name in ("PP_DubinsCurvesCC", "PP_ReedsSheppCurvesHC"):
            planner_key = (name, radius_factor)
            if planner_key not in self.planners:
                planner = getattr(f2c, name)()
                planner.setDiscretization(self.parameters["trajectory_step_m"])
                # v2 upstream cache quantizes angles/distances to 1e-3. Keep our
                # exact-pose cache so one turn cannot inherit another endpoint.
                planner.setUsingCache(False)
                self.planners[planner_key] = planner
            c, s = math.cos(a[2]), math.sin(a[2])
            dx, dy = b[0]-a[0], b[1]-a[1]
            local_end = (c*dx+s*dy, -s*dx+c*dy, wrap(b[2]-a[2]))
            curve_key = (name, radius_factor, *(round(v, 9) for v in local_end))
            if curve_key not in self.curve_cache:
                turn = self.planners[planner_key].createTurn(robot, f2c.Point(0., 0.), 0.,
                    f2c.Point(*local_end[:2]), local_end[2])
                local = []
                for i in range(turn.size()):
                    state = turn.getState(i)
                    local.append([state.point.getX(), state.point.getY(), state.angle, int(state.dir), False])
                if turn.size():
                    last = turn.getState(turn.size()-1)
                    end = last.atEnd()
                    if math.dist(local[-1][:2], (end.getX(), end.getY())) > 1e-8:
                        local.append([end.getX(), end.getY(), last.angle, int(last.dir), False])
                self.curve_cache[curve_key] = np.asarray(local)
            local = self.curve_cache[curve_key]
            if not len(local):
                reasons.append(name+":empty")
                continue
            array = local.copy()
            array[:, 0] = a[0]+c*local[:, 0]-s*local[:, 1]
            array[:, 1] = a[1]+s*local[:, 0]+c*local[:, 1]
            array[:, 2] = a[2]+local[:, 2]
            if math.dist(array[-1, :2], b[:2]) > 1e-5 or abs(wrap(array[-1, 2]-b[2])) > 1e-4:
                reasons.append(name+":endpoint_error")
                continue
            if self.checker.rectangular_allowed and not self.checker.obstacles:
                v = self.vehicle
                inscribed = min(v.front, -v.rear, v.left, -v.right)+self.checker.margin
                xmin, ymin, xmax, ymax = self.checker.allowed_bounds
                if (array[:, 0].min() < xmin+inscribed or array[:, 0].max() > xmax-inscribed
                        or array[:, 1].min() < ymin+inscribed or array[:, 1].max() > ymax-inscribed):
                    reasons.append(name+":outside_allowed_inscribed_body_circle")
                    continue
            invalid_direction = np.any((array[:, 3] != -1) & (array[:, 3] != 1))
            distances = np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1)
            angles = abs((np.diff(array[:, 2])+math.pi) % (2*math.pi)-math.pi)
            curvature_failure = np.any((distances > 1e-8) & (angles > distances*self.vehicle.curvature_limit*1.005))
            if invalid_direction or curvature_failure:
                reasons.append(name+":direction_or_curvature_error")
                continue
            reason = self.checker.path_reason(array[:, :3].tolist())
            if reason:
                reasons.append(name+":"+reason)
                continue
            self.cache[key] = (array.tolist(), None)
            return self.cache[key]
        self.cache[key] = (None, ";".join(reasons))
        return self.cache[key]

    def connect(self, a, b, radius_factor):
        key = tuple(round(float(v), 7) for v in (*a[:3], *b[:3], radius_factor))
        if key in self.bounded_cache:
            return self.bounded_cache[key]
        direct, reason = self._direct_connect(a, b, radius_factor)
        if direct is not None:
            return direct, None
        # The shortest free-space curve can leave the allowed region even when
        # a longer connection exists. Search common, bounded heading-aligned
        # intermediate poses; every leg still comes from the same F2C solver.
        # These are development-fixed connector choices, common to A/B/C.
        alternatives = []
        for endpoint in (a, b):
            for distance in CONNECTOR_OFFSETS_M:
                waypoint = (endpoint[0]+distance*math.cos(endpoint[2]),
                            endpoint[1]+distance*math.sin(endpoint[2]), endpoint[2])
                if self.checker.pose_reason(waypoint):
                    continue
                first, _ = self._direct_connect(a, waypoint, radius_factor)
                if first is None:
                    continue
                second, _ = self._direct_connect(waypoint, b, radius_factor)
                if second is not None:
                    combined = first+second[1:]
                    length = sum(math.dist(p[:2], q[:2]) for p, q in zip(combined, combined[1:]))
                    alternatives.append((length, combined))
        if alternatives:
            result = min(alternatives, key=lambda item:item[0])[1], None
        else:
            result = None, reason+";bounded_intermediate_poses_exhausted"
        self.bounded_cache[key] = result
        return result

    def compose(self, scene, lines, factor, extension):
        path, previous = [], scene["start"]
        for line_index, (a, b) in enumerate(lines):
            yaw = math.atan2(b[1]-a[1], b[0]-a[0])
            c, s = math.cos(yaw), math.sin(yaw)
            a = (a[0]-extension*c, a[1]-extension*s)
            b = (b[0]+extension*c, b[1]+extension*s)
            work = _line(a, b, self.parameters["trajectory_step_m"])
            reason = self.checker.segment_reason(work[0][:3], work[-1][:3])
            if reason:
                return None, "work:"+reason
            connection = self.connect(previous, work[0], factor)
            # Identical-pose connection follows the same return contract.
            if isinstance(connection, tuple):
                connection, reason = connection
            else:
                reason = None
            if connection is None:
                return None, f"connection_to_work_{line_index}:"+reason
            path.extend(connection)
            path.extend(work)
            previous = work[-1]
        connection = self.connect(previous, scene["goal"], factor)
        if isinstance(connection, tuple):
            connection, reason = connection
        else:
            reason = None
        if connection is None:
            return None, "return:"+reason
        path.extend(connection)
        # Remove zero-length duplicate poses while retaining work transitions.
        out = [path[0]]
        for p in path[1:]:
            if math.dist(p[:2], out[-1][:2]) < 1e-7 and abs(wrap(p[2]-out[-1][2])) < 1e-7:
                out[-1] = p
            else:
                out.append(p)
        return out, None


def quick_score(path, vehicle, scene, parameters, activation_compensation, equivalent_tool=False, cache=None):
    """Candidate objective only; final result uses the independent SweepBench evaluator.

    Candidate raster uses the common 5cm lattice and samples poses at <=5cm.
    All final paths are re-evaluated at 5cm/2.5cm from actual kinematic traces.
    """
    from matplotlib.path import Path as MplPath
    signature = tuple((p[0], p[1], p[2], p[3], i == 0 or not path[i-1][4]) for i, p in enumerate(path) if p[4])
    points = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    directions = points[:-1, 3]
    breaks = np.r_[0, np.flatnonzero(np.diff(directions))+1, len(directions)]
    estimated = 0.
    for lo, hi in zip(breaks, breaks[1:]):
        length = float(lengths[lo:hi].sum())
        speed = parameters["nominal_speed_mps"] if directions[lo] > 0 else parameters["reverse_speed_mps"]
        acceleration = parameters["nominal_acceleration_mps2"]
        estimated += (length/speed+speed/acceleration if length >= speed*speed/acceleration
                      else 2*math.sqrt(length/acceleration)) + parameters["stable_stop_duration_s"]
    if cache is not None and signature in cache:
        return dict(cache[signature], estimated_time_s=estimated)
    r = parameters["coverage_grid_m"]
    poly = np.asarray(scene["target"])
    xs = np.arange(math.floor(poly[:, 0].min()/r), math.ceil(poly[:, 0].max()/r))*r+r/2
    ys = np.arange(math.floor(poly[:, 1].min()/r), math.ceil(poly[:, 1].max()/r))*r+r/2
    xx, yy = np.meshgrid(xs, ys)
    target = MplPath(poly).contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    for hole in scene.get("target_holes", []):
        target &= ~MplPath(hole).contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    covered = np.zeros_like(target)
    elapsed, work_distance = 0., 0.
    previous = path[0]
    delay_distance = 0 if activation_compensation else parameters["nominal_speed_mps"]*parameters["nominal_activation_delay_s"]
    for index, p in enumerate(path):
        ds = math.dist(p[:2], previous[:2])
        elapsed += ds/(parameters["nominal_speed_mps"] if p[3] > 0 else parameters["reverse_speed_mps"])
        if p[3] != previous[3]:
            elapsed += 2.0
        work_distance = work_distance+ds if p[4] and p[3] > 0 else 0
        previous = p
        if not p[4] or p[3] < 0 or work_distance < delay_distance:
            continue
        reach = vehicle.radius
        ix = np.where(abs(xs-p[0]) <= reach)[0]
        iy = np.where(abs(ys-p[1]) <= reach)[0]
        if not len(ix) or not len(iy):
            continue
        dx, dy = xx[np.ix_(iy, ix)]-p[0], yy[np.ix_(iy, ix)]-p[1]
        c, s = math.cos(p[2]), math.sin(p[2])
        lx, ly = c*dx+s*dy, -s*dx+c*dy
        mask = np.zeros_like(lx, bool)
        if equivalent_tool:
            mask |= (abs(ly) <= vehicle.coverage_width/2) & (abs(lx) <= .05)
        else:
            for b in vehicle.brushes:
                bx, by = lx-b["x"], ly-b["y"]
                if b["shape"] == "circle":
                    mask |= bx*bx+by*by <= b["radius"]**2
                else:
                    mask |= (abs(bx) <= b["length"]/2) & (abs(by) <= b["width"]/2)
        covered[np.ix_(iy, ix)] |= mask
    score = dict(missed_area_m2=float((target & ~covered).sum()*r*r), estimated_time_s=estimated,
                 candidate_coverage_ratio=float((target & covered).sum()/target.sum()))
    if cache is not None:
        cache[signature] = score
    return score


def plan_coverage(scene, method, vehicle, parameters, design):
    started = time.perf_counter()
    checker = GeometryChecker(scene, vehicle, parameters["collision_margin_m"])
    adapter = F2CAdapter(vehicle, parameters, checker)
    for name in ("start", "goal"):
        reason = checker.pose_reason(scene[name])
        if reason:
            return dict(status="infeasible", reason=name+":"+reason, candidates=[], path=[])
    candidates, feasible, score_cache, boundary_cache = [], [], {}, {}
    is_c = method.startswith("C")
    options = list(itertools.product(design["C"]["lateral_offset_deltas_m"], design["C"]["work_segment_extensions_m"],
                  design["C"]["turn_radius_factors"])) if is_c else [(0, 0, 1)]
    # Preserve all declared candidates and deterministic order; no silent early win.
    for angle in design["A"]["swath_angles_rad"]:
        swaths = _swaths(adapter.f2c, scene, vehicle.coverage_width, angle)
        for offset, extension, factor in options:
            lines = list(swaths)
            resolved_inset = None
            if method != "A":
                if offset not in boundary_cache:
                    boundary_cache[offset] = _feasible_boundary(scene, vehicle.coverage_width/2+offset, checker)
                boundary, resolved_inset = boundary_cache[offset]
                if boundary is None:
                    candidates.append(dict(angle=angle, boundary_offset_delta_m=offset, extension_m=extension,
                        radius_factor=factor, feasible=False, reason="boundary_inset_infeasible", resolved_boundary_inset_m=resolved_inset))
                    continue
                lines += boundary
            path, reason = adapter.compose(scene, lines, factor, extension)
            entry = dict(angle=angle, boundary_offset_delta_m=offset, extension_m=extension, radius_factor=factor,
                         feasible=path is not None, reason=reason, resolved_boundary_inset_m=resolved_inset)
            if path is not None:
                score = quick_score(path, vehicle, scene, parameters,
                    activation_compensation=is_c and method != "C_no_activation_timing",
                    equivalent_tool=method == "C_no_tool_geometry", cache=score_cache)
                entry.update(score)
                entry["within_estimated_task_budget"] = score["estimated_time_s"] <= parameters["max_task_time_s"]
                feasible.append((entry, path))
            candidates.append(entry)
    elapsed = time.perf_counter()-started
    if not feasible:
        return dict(status="infeasible", reason="all_candidates_rejected", candidates=candidates, path=[], planning_s=elapsed)
    bounded = [(e, p) for e, p in feasible if e["within_estimated_task_budget"]]
    partial_only = not bounded
    feasible = bounded or feasible
    pareto = [(e, p) for e, p in feasible if not any(
        other["missed_area_m2"] <= e["missed_area_m2"] and other["estimated_time_s"] <= e["estimated_time_s"]
        and (other["missed_area_m2"] < e["missed_area_m2"] or other["estimated_time_s"] < e["estimated_time_s"])
        for other, _ in feasible)]
    # A/B use time; C selects the lowest miss on its frontier, disclosing extra time.
    selected, path = min(pareto if is_c else feasible,
        key=(lambda ep: (ep[0]["missed_area_m2"], ep[0]["estimated_time_s"])) if is_c else (lambda ep: ep[0]["estimated_time_s"]))
    return dict(status="partial_plan" if partial_only else "planned",
                reason="estimated_time_exceeds_task_budget" if partial_only else "candidate_found", path=path, selected=selected,
                candidates=candidates, pareto=[e for e, _ in pareto], planning_s=elapsed,
                path_fields=["x_m", "y_m", "yaw_rad", "direction", "work_segment"],
                upstream="Fields2Cover@3613525c241538fa9fd9df3e1209ae8184627958")
