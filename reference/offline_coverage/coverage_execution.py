"""Feedback tracking for offline coverage experiments; no robot command output."""
from collections import deque
import math

import numpy as np

from .coverage_geometry import GeometryChecker, KinematicState, evaluator_sample, wrap


def execute_path(path, scene, vehicle, parameters, *, compensate_activation=False, extra_steer_delay=0.0):
    if not path:
        raise ValueError("cannot execute an empty plan")
    dt = parameters["kinematic_dt_s"]
    state = KinematicState(*scene["start"])
    checker = GeometryChecker(scene, vehicle, parameters["collision_margin_m"])
    delay_steps = round(extra_steer_delay/dt)
    delay = deque([0.] * delay_steps)
    path = np.asarray(path, dtype=float)
    # Direction belongs to the outgoing state. A cusp always ends in a stop.
    breaks = [0]+[i for i in range(1, len(path)) if path[i, 3] != path[i-1, 3]]+[len(path)-1]
    breaks = sorted(set(breaks))
    trace, states, tool_events = [], [], []
    t, command_since, tool_command = 0., None, False
    collision_count = 0
    max_tracking = 0.
    status = dict(status="running", reason="tracking", arrival_seen=False, start_t=0.)

    def record(effective, phase):
        trace.append(evaluator_sample(vehicle, state.pose, t, state.speed, effective, phase, collision_count))
        states.append(dict(t=t, x=state.x, y=state.y, yaw=state.yaw, speed=state.speed,
                           front_axle_rad=state.axle_angle, rear_axle_rad=-state.axle_angle,
                           front_rate_rad_s=state.axle_rate, tool_command=tool_command,
                           tool_effective=bool(effective and state.speed >= 0)))

    record(False, "executing")
    for lo, hi in zip(breaks, breaks[1:]):
        segment = path[lo:hi+1]
        direction = int(segment[0, 3])
        points = segment[:, :2]
        cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        progress = 0
        while True:
            if t >= parameters["max_task_time_s"]:
                status.update(status="timeout", reason="kinematic_task_time_limit")
                break
            window_end = min(len(segment), progress+60)
            distances = np.linalg.norm(points[progress:window_end]-[state.x, state.y], axis=1)
            nearest = progress+int(np.argmin(distances))
            progress = nearest
            tracking = float(distances.min())
            max_tracking = max(max_tracking, tracking)
            if tracking > 1.0:
                status.update(status="tracking_failed", reason="tracking_error_exceeds_1m_diagnostic")
                break
            remaining = cumulative[-1]-cumulative[progress]
            endpoint_distance = math.dist(state.pose[:2], points[-1])
            if remaining < .06 and endpoint_distance < .06 and abs(state.speed) < .025:
                break
            speed_limit = parameters["nominal_speed_mps"] if direction > 0 else parameters["reverse_speed_mps"]
            speed_command = direction*min(speed_limit, math.sqrt(2*parameters["nominal_acceleration_mps2"]*max(0., remaining-.025)))
            if remaining < .025 and endpoint_distance > .06:
                speed_command = direction*.035
            lookahead = .25 + .5*abs(state.speed)
            target_index = min(len(segment)-1, int(np.searchsorted(cumulative, cumulative[progress]+lookahead)))
            target = segment[target_index]
            dx, dy = target[0]-state.x, target[1]-state.y
            local_y = -math.sin(state.yaw)*dx+math.cos(state.yaw)*dy
            curvature = 2*local_y/max(dx*dx+dy*dy, .01)
            if delay_steps:
                delay.append(curvature)
                curvature = delay.popleft()
            lead = speed_limit*parameters["nominal_activation_delay_s"] if compensate_activation else 0.
            activation_index = min(len(segment)-1, int(np.searchsorted(cumulative, cumulative[progress]+lead)))
            requested = bool(direction > 0 and (segment[progress, 4] or segment[activation_index, 4]))
            if requested != tool_command:
                tool_command = requested
                command_since = t if requested else None
                tool_events.append(dict(t=t, command=requested, source="offline_common_tool_actuator"))
            effective = bool(tool_command and command_since is not None and
                             t-command_since+1e-9 >= parameters["nominal_activation_delay_s"] and direction > 0)
            previous = state.pose
            state.step(speed_command, curvature, dt, vehicle, parameters["nominal_acceleration_mps2"])
            t = round(t+dt, 10)
            reason = checker.segment_reason(previous, state.pose)
            if reason == "collision_obstacle" and collision_count == 0:
                collision_count = 1
            record(effective and reason is None, "executing")
            if reason:
                status.update(status="geometry_failed", reason=reason)
                break
        if status["status"] != "running":
            break
        # Explicit stationary switch; preserve steering dynamics and turn tools off.
        tool_command, command_since = False, None
        tool_events.append(dict(t=t, command=False, source="direction_switch_or_final_stop"))
        for _ in range(math.ceil(parameters["stable_stop_duration_s"]/dt)):
            previous = state.pose
            curvature = 0.
            if delay_steps:
                delay.append(0.)
                curvature = delay.popleft()
            state.step(0., curvature, dt, vehicle, parameters["nominal_acceleration_mps2"])
            t = round(t+dt, 10)
            record(False, "stopping")
            reason = checker.segment_reason(previous, state.pose)
            if reason:
                status.update(status="geometry_failed", reason=reason)
                break
        if status["status"] != "running":
            break
    if status["status"] != "running":
        # Failure cancels tool work and requests a bounded stop. Retain the
        # braking tail instead of ending the trace at a nonzero speed.
        tool_command, command_since = False, None
        status["failure_detected_t"] = t
        tool_events.append(dict(t=t, command=False, source="failure_stop"))
        stop_duration = parameters["stable_stop_duration_s"]+abs(state.speed)/parameters["nominal_acceleration_mps2"]
        for _ in range(math.ceil(stop_duration/dt)):
            curvature = 0.
            if delay_steps:
                delay.append(0.)
                curvature = delay.popleft()
            previous = state.pose
            state.step(0., curvature, dt, vehicle, parameters["nominal_acceleration_mps2"])
            t = round(t+dt, 10)
            reason = checker.segment_reason(previous, state.pose)
            if reason == "collision_obstacle" and collision_count == 0:
                collision_count = 1
            record(False, "failed_stopping")
    position_error = math.dist(state.pose[:2], scene["goal"][:2])
    yaw_error = abs(wrap(state.yaw-scene["goal"][2]))
    arrived = position_error <= parameters["position_arrival_tolerance_m"] and yaw_error <= parameters["yaw_arrival_tolerance_rad"]
    if status["status"] == "running":
        status.update(status="completed" if arrived else "arrival_failed", reason="stable_goal" if arrived else "goal_pose_tolerance", arrival_seen=arrived)
    status.update(end_t=t, position_error_m=position_error, yaw_error_rad=yaw_error,
                  max_tracking_error_m=max_tracking, direction_segments=len(breaks)-1,
                  evidence="offline_kinematic_feedback_tracking")
    return trace, states, tool_events, status
