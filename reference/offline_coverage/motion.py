"""生成 Q-Loop 运动建议，并统一交回 W2 仲裁。"""

from __future__ import annotations

import math
from typing import Iterable

from dg202604.contracts import DecodedRecord
from dg202604.safety import (
    SafetyArbiter,
    SafetyContext,
    SafetyDecision,
    SafetyState,
)


Point = tuple[float, float]


def build_motion_proposal(
    path: Iterable[Point],
    *,
    record_id: str,
    source_run_id: str,
    timestamp_ns: int,
    ttl_ns: int,
    start_yaw_rad: float,
    speed_mps: float,
    lookahead_m: float,
    tool_footprint: Iterable[Point],
) -> DecodedRecord:
    """根据路径前两点生成一个 `D-MOTION-PROP` 运动建议。"""

    points = tuple(path)
    if len(points) < 2:
        raise ValueError("motion proposal requires two points")
    if ttl_ns <= 0 or lookahead_m <= 0 or not math.isfinite(speed_mps):
        raise ValueError("positive ttl/lookahead and finite speed required")
    if not math.isfinite(start_yaw_rad) or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("finite path and yaw required")
    start = points[0]
    target = points[1]
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    distance = math.hypot(dx, dy)
    if speed_mps == 0 and distance > 1e-9:
        raise ValueError("stop proposal must repeat the current position")
    local_y = -math.sin(start_yaw_rad) * dx + math.cos(start_yaw_rad) * dy
    # 使用纯追踪几何求曲率；完整控制器动态不属于 W6。
    curvature = 2.0 * local_y / max(distance, lookahead_m) ** 2

    # 业务边界：W6 只能创建建议，不能直接创建可执行 D-MOTION。
    return DecodedRecord(
        contract_id="D-MOTION-PROP",
        schema_version="1.0",
        record_id=record_id,
        producer="C-PLAN-LOCAL",
        frame="odom",
        timestamp_ns=timestamp_ns,
        time_domain="system_monotonic",
        source_run_id=source_run_id,
        evidence_status="offline_measured",
        quality_status="untrusted_until_arbitrated",
        payload={
            "trajectory": [
                {
                    "x_m": start[0],
                    "y_m": start[1],
                    "yaw_rad": start_yaw_rad,
                    "time_from_start_ms": 0.0,
                },
                {
                    "x_m": target[0],
                    "y_m": target[1],
                    "yaw_rad": start_yaw_rad + curvature * distance,
                    "time_from_start_ms": distance / abs(speed_mps) * 1000.0 if speed_mps else ttl_ns/1e6,
                },
            ],
            "velocity_mps": speed_mps,
            "tool_footprint": [list(point) for point in tool_footprint],
            "costs": {"curvature_abs": abs(curvature)},
            "valid_until_ns": timestamp_ns + ttl_ns,
        },
    )


def arbitrate_motion_proposal(
    proposal: DecodedRecord,
    *,
    arbiter: SafetyArbiter,
    current_state: SafetyState,
    context: SafetyContext,
) -> SafetyDecision:
    """把 W6 建议交给 W2；只有 W2 可以产生可执行运动命令。"""

    return arbiter.arbitrate(
        proposal,
        current_state=current_state,
        context=context,
    )


def build_segment_motion_proposal(
    segment: dict,
    *,
    current_position: Point,
    speed_mps: float,
    **proposal_fields,
) -> DecodedRecord:
    """Refresh one forward/reverse/stop segment using the caller's current clock.

    Input must already be transformed to odom. Mission-map coordinates and an
    entire mixed-direction route cannot be mislabeled as an executable command.
    The returned record still requires W2 arbitration.
    """
    if segment.get("frame") != "odom":
        raise ValueError("segment must be explicitly transformed to odom")
    direction = segment.get("direction")
    if direction not in ("forward", "reverse", "stop"):
        raise ValueError("segment direction must be forward, reverse, or stop")
    if direction == "stop":
        target, velocity = current_position, 0.
    else:
        points = tuple(segment.get("points", ()))
        if not points or not math.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError("moving segment requires points and a positive speed magnitude")
        target = points[-1]
        for point in points:
            if math.dist(current_position, point) >= proposal_fields["lookahead_m"]:
                target = point
                break
        velocity = speed_mps if direction == "forward" else -speed_mps
    return build_motion_proposal([current_position, target], speed_mps=velocity, **proposal_fields)
