"""Business commands for the existing, bounded forward-road controller.

This module never publishes actuator commands. The road loop uses ``hold`` as
one more stop condition; receipts are completed by advancing vehicle feedback.
Brush commands are logical only: no brush actuator or feedback exists, so the
state machine confirms them itself and marks ``feedbackAvailable=False``.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time


PROTOCOL = 1
ACTIONS = ('dispatch', 'pause', 'resume', 'cancel', 'stop', 'tool_on', 'tool_off')
TERMINAL = ('succeeded', 'rejected', 'failed', 'timed_out')
TASK_ENDED = ('completed', 'canceled', 'failed')
_SCRIPT_PATH = Path(__file__).resolve()
# Local checkout: E:/Horizon/simulation/carla-autoware/scripts. The remote
# runtime mounts the same directory at /integration/scripts, so its project
# root is supplied by the container's /repo mount instead of a fourth parent.
ROOT = (_SCRIPT_PATH.parents[3] if len(_SCRIPT_PATH.parents) >= 4 else Path('/repo'))
COVERAGE_GRID_M = .05  # workbench/tasks/20260908-CARLA主线迁移/实验规格-v1.json common_parameters.coverage_grid_m

# 三轮五刷逻辑足迹。车辆系：x 前、y 左、后轴地面参考点，单位 m。
# 这不是实车接触标定（config/tricycle-navigation.json: brush_contact_valid=false），
# 仅供覆盖账本与覆盖规划使用；数值来源：
# - 外刷中心与半径：simulation/carla-autoware/tests/test_tricycle_work_geometry.py 的名义盘刷
#   nominal_axis_intersection_with_z0_m=[1.322, ±.824]、nominal_radius_m=.25。
# - 内刷中心：assets/tricycle-20260911/evidence/runtime/result.json brush_ray_checks 射线原点
#   (0.84, ±0.6)（CARLA 车辆系 y 向右，此处翻转为 y 向左）；半径无三轮车来源，
#   借用 simulation/awsim-autodrrt/config/cleaning-robot-v2-build-profile.json innerBrushRadii=0.145 作示例。
# - 主滚刷：位置与尺寸无来源；示例取 x=0.45、宽 0.72（geometry.json 网格名 roller_hanger_0.72）、长 0.10。
TRICYCLE_BRUSHES = (
    dict(id='left-outer', shape='circle', x=1.322, y=.824, radius=.25),
    dict(id='right-outer', shape='circle', x=1.322, y=-.824, radius=.25),
    dict(id='left-inner', shape='circle', x=.84, y=.6, radius=.145),
    dict(id='right-inner', shape='circle', x=.84, y=-.6, radius=.145),
    dict(id='roller', shape='rectangle', x=.45, y=0., length=.1, width=.72),
)
BRUSH_SOURCE = 'console_session.TRICYCLE_BRUSHES (unverified example footprints)'


def brush_forward_reach(brushes=TRICYCLE_BRUSHES):
    """Farthest brush footprint point ahead of the rear-axle reference, in metres.

    Route poses are rear-axle poses, so a work segment that starts at pose k is
    already under the brushes this far before the axle reaches k; the driver
    switches the logical brush on that far ahead of a work segment.
    """
    reach = 0.
    for b in brushes:
        if b['shape'] == 'circle':
            reach = max(reach, b['x']+b['radius'])
        elif b['shape'] == 'rectangle':
            reach = max(reach, b['x']+b['length']/2)
        else:
            raise ValueError('unsupported brush shape: '+str(b['shape']))
    return reach


def _qloop_importable():
    if str(ROOT/'dg202604/src') not in sys.path:
        sys.path.append(str(ROOT/'dg202604/src'))


def tricycle_vehicle():
    """Tricycle body/steering from the navigation asset plus the logical brushes above."""
    from dataclasses import replace
    _qloop_importable()
    from tricycle_navigation import coverage_vehicle, load_profile
    return replace(coverage_vehicle(load_profile()), brushes=TRICYCLE_BRUSHES)


class CoverageLedger:
    """Brush footprint × pose stream on the common 5 cm lattice.

    Bookkeeping is ``dg202604.qloop.cleaning.coverage``; footprints come from
    ``coverage_geometry.shape_at``. The result is a pure function of the pose
    and brush-state sequence, so replaying one trajectory yields equal numbers.
    """

    def __init__(self, polygon, brushes=TRICYCLE_BRUSHES, grid=COVERAGE_GRID_M, targets=(),
                 cleaning_policy='full_coverage'):
        _qloop_importable()
        import numpy as np
        import shapely
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from dg202604.qloop.coverage_geometry import shape_at
        self.np, self.shapely, self.unary_union, self.shape_at = np, shapely, unary_union, shape_at
        self.grid, self.brushes = float(grid), tuple(brushes)
        target = Polygon(polygon)
        if not target.is_valid or target.is_empty or target.area <= 0:
            raise ValueError('作业区多边形无效')
        self.target_cells = frozenset(self._cells(target))
        self.litter_targets = deepcopy(list(targets))
        self.cleaning_policy = cleaning_policy
        self.reach = max(abs(v) for b in brushes for v in shape_at(b, (0., 0., 0.)).bounds)
        self.swept = []
        self.previous = None
        self.samples = 0
        self.on_distance = 0.

    def _cells(self, geometry):
        g, np = self.grid, self.np
        xmin, ymin, xmax, ymax = geometry.bounds
        ix = np.arange(math.floor(xmin/g), math.ceil(xmax/g)+1)
        iy = np.arange(math.floor(ymin/g), math.ceil(ymax/g)+1)
        ii, jj = np.meshgrid(ix, iy)
        inside = self.shapely.contains_xy(geometry, (ii.ravel()+.5)*g, (jj.ravel()+.5)*g)
        return list(zip(ii.ravel()[inside].tolist(), jj.ravel()[inside].tolist()))

    def _footprint(self, pose):
        return self._cells(self.unary_union([self.shape_at(b, pose, conservative=False) for b in self.brushes]))

    def integrate(self, pose, brush_on):
        pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        if not all(math.isfinite(v) for v in pose):
            raise ValueError('non-finite pose')
        self.samples += 1
        if self.previous is not None and self.previous[1]:
            # The brush state at the previous sample governs the interval up to this one.
            prev = self.previous[0]
            distance = math.dist(prev[:2], pose[:2])
            dyaw = (pose[2]-prev[2]+math.pi) % (2*math.pi)-math.pi
            self.on_distance += distance
            n = max(1, math.ceil(max(distance, abs(dyaw)*self.reach)/(self.grid/2)))
            for k in range(1, n+1):
                f = k/n
                self.swept.extend(self._footprint((prev[0]+f*(pose[0]-prev[0]), prev[1]+f*(pose[1]-prev[1]), prev[2]+f*dyaw)))
        elif brush_on:
            self.swept.extend(self._footprint(pose))
        self.previous = (pose, bool(brush_on))

    def report(self, *, final):
        from dg202604.qloop.cleaning import coverage
        g = self.grid
        result = coverage(self.target_cells, self.swept, [(0, 0)])
        area = len(self.target_cells)*g*g
        covered = len(result.covered_cells)*g*g
        report = dict(final=bool(final), gridM=g, areaM2=round(area, 4), coveredM2=round(covered, 4),
            coverageRatio=round(covered/area, 6) if area else None,
            overlapVisits=result.overlap_visits, brushOnDistanceM=round(self.on_distance, 3),
            poseSamples=self.samples, frameId='map', feedbackAvailable=False,
            footprintVerified=False, footprintSource=BRUSH_SOURCE, status='measured', reason=None)
        residual = set(result.missed_cells) if self.cleaning_policy == 'full_coverage' else set()
        if self.litter_targets:
            from shapely.geometry import Polygon
            visited = set(self.swept)
            rows = []
            for item in self.litter_targets:
                if item['kind'] == 'point':
                    cells = {(math.floor(item['point'][0]/g), math.floor(item['point'][1]/g))}
                else:
                    cells = set(self._cells(Polygon(item['polygon'])))
                if self.cleaning_policy == 'on_demand':
                    residual.update(cells-visited if item['visible'] else cells)
                ratio = len(cells & visited)/len(cells) if cells and self.samples and item['visible'] else None
                state = ('unobserved' if ratio is None else 'covered' if ratio == 1. else
                         'partial' if ratio > 0. else 'missed')
                rows.append(dict(id=item['id'], state=state, coverageRatio=ratio))
            report['targetCoverage'] = rows
        report.update(residualCount=len(residual),
            residualCells=[[round((i+.5)*g, 3), round((j+.5)*g, 3)] for i, j in sorted(residual)])
        if not self.samples:
            report.update(status='unknown', coverageRatio=None, coveredM2=None,
                          reason='驾驶端未提供位姿流，覆盖率未知')
        return report


class ConsoleSession:
    def __init__(self, session_id, map_manifest, *, wall_clock=time.time,
                 monotonic=time.monotonic, feedback_timeout_s=30.):
        self.session_id = session_id
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        if not isinstance(feedback_timeout_s, (int, float)) or not math.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0:
            raise ValueError('feedback_timeout_s must be finite and positive')
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.map_manifest = map_manifest
        self.map = None
        self.task = None
        self.route = None
        self.ledger = None
        self.receipts = {}
        self.requests = {}
        self.pending = None
        self.last_heartbeat = None
        self.last_feedback = None
        self.feedback_stamp = None
        self.stopped_since = None
        self.pending_since = None
        self.reason = '等待工作台连接'
        self.ready = False
        self.sequence = 0
        self.speed_limit = .6
        self.last_hold_request = -math.inf
        self.brush = dict(state='off', commandedAt=None, confirmedAt=None, feedbackAvailable=False, reason='初始状态')
        # Area tasks follow the route's work segments automatically; a manual
        # tool_off holds the brush off until the next tool_on or dispatch.
        self.brush_manual_off = False
        # Optional persistent work memory (研究/地图变化证据与版本维护.md §3, §4.3):
        # the loop sets ``work_memory_path``; None keeps the session file-free.
        self.work_memory_path = None
        self.blocked_stops = []
        self.work_memory_status = None

    def configure_route(self, origin, length, lanelet):
        if self.map is not None:
            return
        x, y, yaw = map(float, origin)
        end = dict(x=x+length*math.cos(yaw), y=y+length*math.sin(yaw))
        route = dict(origin=[x, y, yaw], length=float(length), lanelet=str(lanelet),
                     manifest=self.map_manifest)
        digest = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()[:16]
        self.map = dict(id='town01-'+digest, version=1, name='Town01 行驶验证段',
            siteId='town01', frameId='map', units='m', review='reviewed', release='published',
            createdAt=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.wall_clock())),
            source='ros', manifest=self.map_manifest,
            features=[dict(id='forward-route', name=f'前进直线 {length:g} m', kind='route',
                           points=[dict(x=x, y=y), end], visible=True)],
            route=route)

    def heartbeat(self, message):
        if (message.get('protocol') == PROTOCOL
                and message.get('sessionId') == self.session_id
                and isinstance(message.get('sentAt'), (int, float))
                and -.5 <= self.wall_clock()-message['sentAt'] <= 3.):
            self.last_heartbeat = self.monotonic()

    @property
    def connected(self):
        return self.last_heartbeat is not None and self.monotonic()-self.last_heartbeat <= 3.

    @property
    def hold(self):
        return (not self.connected or self.task is None
                or self.task['state'] not in ('queued', 'running'))

    @property
    def route_length(self):
        return self.route['length'] if self.route else self.map['route']['length']

    def transition(self, command_id, status, reason, source_time):
        receipt = self.receipts[command_id]
        if receipt['status'] == status:
            return
        receipt.update(status=status, reason=reason)
        receipt['timeline'].append(dict(state=status, at=self.wall_clock(),
                                        sourceTime=source_time, description=reason))

    def _set_brush(self, state, source_time, reason):
        self.brush = dict(state=state, commandedAt=source_time, confirmedAt=source_time,
                          feedbackAvailable=False, reason=reason)

    def auto_brush(self, demand, source_time):
        """Follow the coverage route's work segments; returns the applied logical state.

        Only a running/queued ``clean_area`` task is driven automatically, and a
        manual ``tool_off`` overrides the demand until ``tool_on`` or a new dispatch.
        """
        if (self.task is None or self.task.get('type') != 'clean_area' or self.route is None
                or self.task['state'] not in ('queued', 'running') or self.brush_manual_off):
            return self.brush['state']
        wanted = 'on' if demand else 'off'
        if self.brush['state'] != wanted:
            self._set_brush(wanted, source_time, '作业段自动开启' if demand else '离开作业段自动关闭')
        return wanted

    BLOCKED_REASONS = ('geometry_obstacle', 'object_obstacle', 'predicted_path_conflict')

    def _note_blocked(self, reason, pose, source_time):
        """Remember where an area task was held by an obstacle; one entry per 0.5 m per reason."""
        if (reason not in self.BLOCKED_REASONS or pose is None or self.task is None
                or self.task.get('type') != 'clean_area' or self.task['state'] not in ('queued', 'running')):
            return
        x, y = float(pose[0]), float(pose[1])
        for entry in self.blocked_stops:
            if entry['reason'] == reason and math.dist((x, y), (entry['x'], entry['y'])) < .5:
                entry['last_s'] = source_time
                return
        self.blocked_stops.append(dict(reason=reason, x=x, y=y, first_s=source_time, last_s=source_time))

    def _finish_report(self):
        if self.ledger is not None and self.task is not None:
            self.task['coverageReport'] = self.ledger.report(final=self.task['state'] in TASK_ENDED)
            if self.task['state'] in TASK_ENDED:
                self._persist_work_memory()

    def _persist_work_memory(self):
        """Append residual cells and blocked stops of the ended area task to the SQLite work memory.

        Never raises into the driving loop: failures are reported in ``work_memory_status``
        and on the task so the operator sees that nothing was persisted.
        """
        if self.work_memory_path is None or self.task is None:
            return
        report = self.task.get('coverageReport') or {}
        task_id, version = self.task['id'], self.map['id']
        try:
            _qloop_importable()
            root = str(ROOT/'dg202604')
            if root not in sys.path:
                sys.path.append(root)
            from perception.local_world.memory import SubmapMemory
            memory = SubmapMemory(self.work_memory_path)
            try:
                written = 0
                cells = report.get('residualCells') or []
                if report.get('status') == 'measured' and cells:
                    written += memory.add_work_memory(f'{task_id}:residual', task=task_id, map_version=version,
                        kind='RESIDUAL', geometry=cells, capture_s=self.feedback_stamp or 0.)
                for index, stop in enumerate(self.blocked_stops):
                    written += memory.add_work_memory(f'{task_id}:blocked:{index}', task=task_id, map_version=version,
                        kind='BLOCKED', geometry=[[stop['x'], stop['y']]], capture_s=stop['first_s'], reason=stop['reason'])
                self.work_memory_status = dict(status='written', path=str(self.work_memory_path), rows=written,
                    residualCells=len(cells) if report.get('status') == 'measured' else 0,
                    blockedStops=len(self.blocked_stops))
            finally:
                memory.close()
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            self.work_memory_status = dict(status='failed', path=str(self.work_memory_path), reason=str(exc))
        self.task['workMemory'] = deepcopy(self.work_memory_status)

    def _dispatch_error(self, task, speed):
        """Return (error, route, ledger); never substitutes a route or a ledger."""
        if not isinstance(task, dict):
            return '缺少有效任务', None, None
        if self.task and self.task['state'] not in TASK_ENDED:
            return '车辆已有未结束任务', None, None
        if not self.map or task.get('mapVersionId') != self.map['id']:
            return '任务必须绑定当前已加载地图', None, None
        if not isinstance(task.get('speed'), (float, int)) or not math.isfinite(task['speed']) or not .2 <= task['speed'] <= .6:
            return '行驶速度须在 0.2–0.6 m/s', None, None
        if abs(speed) >= .05:
            return '车辆尚未停稳', None, None
        if task.get('type', 'forward-route') != 'clean_area':
            if task.get('featureId') != 'forward-route':
                return '任务必须绑定当前已加载地图与验证路线', None, None
            if task.get('mode') != 'off':
                return '刷具尚未接入；当前只支持行驶验证', None, None
            if getattr(self, 'progress', 0.) >= self.map['route']['length']-.3:
                return '本段已到终点，请启动新的行驶验证会话', None, None
            return None, None, None
        route = task.get('route')
        poses = route.get('poses') if isinstance(route, dict) else None
        if (not isinstance(poses, list) or len(poses) < 2 or not all(
                isinstance(p, list) and len(p) == 5 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p[:3])
                and p[3] in (-1, 1) and isinstance(p[4], bool) for p in poses)):
            return '覆盖路径无效：缺少 [x, y, yaw, direction, work_segment] 位姿序列', None, None
        if not isinstance(route.get('length'), (int, float)) or not math.isfinite(route['length']) or route['length'] <= 0:
            return '覆盖路径长度无效', None, None
        if task.get('mode') not in ('off', 'standard', 'edge'):
            return '清扫模式无效', None, None
        polygon = task.get('polygon')
        if not isinstance(polygon, list) or len(polygon) < 3:
            return '作业区多边形无效', None, None
        try:
            ledger = CoverageLedger(polygon, targets=task.get('litterTargets', ()),
                                    cleaning_policy=task.get('cleaningPolicy', 'full_coverage'))
        except ImportError as exc:
            return '覆盖账本不可用：'+str(exc), None, None
        except ValueError as exc:
            return '覆盖账本拒绝作业区：'+str(exc), None, None
        return None, dict(kind='coverage', poses=deepcopy(poses), length=float(route['length'])), ledger

    def receive(self, request, source_time, speed):
        """Validate before releasing the existing planner's stop gate."""
        command_id = request.get('commandId')
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
            return
        if command_id in self.requests:
            # A conflicting ID must not alter the original execution or receipt.
            return
        self.requests[command_id] = deepcopy(request)
        action = request.get('action')
        self.receipts[command_id] = dict(commandId=command_id, action=action,
            taskId=request.get('taskId'), status='requested', timeline=[])
        error, route, ledger = None, None, None
        issued = request.get('issuedAt')
        if request.get('protocol') != PROTOCOL or request.get('sessionId') != self.session_id:
            error = '协议或驾驶会话不匹配'
        elif not isinstance(issued, (float, int)) or not math.isfinite(issued) or not -.5 <= self.wall_clock()-issued <= 5.:
            error = '命令已过期或时间无效'
        elif action not in ACTIONS:
            error = '当前驾驶器未接入此操作'
        elif action in ('dispatch', 'resume') and issued <= self.last_hold_request:
            error = '该运动请求早于最近的停车或暂停，已失效'
        elif not self.connected and action != 'stop':
            error = '工作台心跳已中断'
        elif self.pending is not None and action not in ('stop', 'tool_off'):
            error = '上一条命令仍在执行'
        elif action in ('dispatch', 'resume') and not self.ready:
            error = '定位、感知或控制输入尚未就绪'
        elif action in ('pause', 'resume', 'cancel') and (
                self.task is None or request.get('taskId') != self.task['id']):
            error = '任务与当前驾驶任务不匹配'
        elif action == 'dispatch':
            task = request.get('task')
            if not isinstance(task, dict) or task.get('id') != request.get('taskId'):
                error = '缺少有效任务'
            else:
                error, route, ledger = self._dispatch_error(task, speed)
        elif action == 'pause' and self.task['state'] != 'running':
            error = '只有运行中的任务可以暂停'
        elif action == 'resume' and self.task['state'] != 'paused':
            error = '只有已停稳的暂停任务可以恢复'
        elif action == 'cancel' and self.task['state'] not in ('queued', 'running', 'pausing', 'paused'):
            error = '当前任务不可取消'
        elif action == 'tool_on' and (self.task is None or self.task['state'] not in ('queued', 'running')):
            error = '只有排队或运行中的任务可以开启刷具'
        elif action in ('tool_on', 'tool_off') and request.get('taskId') not in (None, self.task['id'] if self.task else None):
            error = '任务与当前驾驶任务不匹配'
        if error:
            self.transition(command_id, 'rejected', error, source_time)
            return
        if action in ('tool_on', 'tool_off'):
            # No brush actuator or feedback exists; the logical state is the confirmation.
            self.transition(command_id, 'accepted', '驾驶器已接受刷具逻辑命令', source_time)
            self._set_brush('on' if action == 'tool_on' else 'off', source_time, '工作台命令 '+action)
            self.brush_manual_off = action == 'tool_off' and self.task is not None and self.task['state'] not in TASK_ENDED
            self.transition(command_id, 'succeeded', '刷具逻辑状态已切换；无实车刷具反馈（feedbackAvailable=false）', source_time)
            return
        if self.pending is not None:
            self.transition(self.pending, 'failed', '已被停车请求替代', source_time)
        self.pending = command_id
        self.pending_since = self.monotonic()
        self.stopped_since = None
        self.transition(command_id, 'accepted', '驾驶器已接受，等待执行反馈', source_time)
        if action == 'dispatch':
            self.task = dict(id=request['taskId'], state='queued', progress=0., distance=0.,
                             mapVersionId=self.map['id'], reason='等待车辆开始运动',
                             type=request['task'].get('type', 'forward-route'))
            self.route, self.ledger = route, ledger
            self.blocked_stops, self.work_memory_status = [], None
            self.speed_limit = float(request['task']['speed'])
            self._set_brush('off', source_time, '任务下发时刷具关闭')
            self.brush_manual_off = False
        elif action == 'resume':
            self.task.update(state='queued', reason='等待恢复运动反馈')
        elif self.task and self.task['state'] not in TASK_ENDED:
            self.task.update(state='canceling' if action == 'cancel' else 'pausing',
                             reason='等待制动与停稳反馈')
        if action in ('pause', 'cancel', 'stop'):
            self.last_hold_request = self.wall_clock()
            if self.brush['state'] == 'on':
                self._set_brush('off', source_time, '停车、暂停或取消请求时关闭刷具')
        self.reason = '等待车辆执行反馈'
        self.transition(command_id, 'executing', '已交给现有规划与控制链', source_time)

    def watchdog(self, source_time):
        stale_feedback = self.last_feedback is None or self.monotonic()-self.last_feedback > 5.
        if (not self.connected or stale_feedback) and self.task and self.task['state'] in ('queued', 'running'):
            reason = '工作台心跳中断，停车后需手动恢复' if not self.connected else '车辆反馈停止推进，停车后需手动恢复'
            self.task.update(state='pausing', reason=reason)
            self.reason = reason
            self.last_hold_request = self.wall_clock()
        if self.pending is not None and self.monotonic()-self.pending_since > self.feedback_timeout_s:
            self.transition(self.pending, 'timed_out', '未在期限内取得执行反馈；已保持停车', source_time)
            self.pending = None
            if self.task and self.task['state'] in ('queued', 'running'):
                self.task.update(state='pausing', reason='命令执行反馈超时')
                self.last_hold_request = self.wall_clock()
        return self.hold or stale_feedback

    def observe(self, *, source_time, speed, brake, progress, ready, reason, goal, pose=None):
        """``pose`` is the EKF map-frame (x, y, yaw); without it the ledger reports unknown."""
        self.ready = ready
        self.progress = max(0., float(progress))
        self.reason = reason
        if self.feedback_stamp is not None and source_time <= self.feedback_stamp:
            return
        self.feedback_stamp = source_time
        self.last_feedback = self.monotonic()
        if abs(speed) < .05 and brake >= .5:
            if self.stopped_since is None:
                self.stopped_since = source_time
        else:
            self.stopped_since = None
        stopped = self.stopped_since is not None and source_time-self.stopped_since >= .5-1e-6
        if self.task and self.task['state'] not in TASK_ENDED:
            if self.ledger is not None and pose is not None:
                self.ledger.integrate(pose, self.brush['state'] == 'on')
            self._note_blocked(reason, pose, source_time)
            length = self.route_length
            self.task.update(distance=min(length, self.progress),
                progress=min(100., self.progress/length*100))
            if self.task['state'] in ('queued', 'running'):
                self.task['reason'] = reason
            if self.task['state'] == 'queued' and speed > .05 and ready and not self.hold:
                self.task['state'] = 'running'
            if self.task['state'] == 'pausing' and stopped:
                self.task['state'] = 'paused'
                if self.task['reason'] == '等待制动与停稳反馈':
                    self.task['reason'] = '车辆已停稳，需手动恢复任务'
                self._finish_report()
            if self.task['state'] == 'canceling' and stopped:
                self.task.update(state='canceled', reason='任务已取消并停稳')
                self._finish_report()
            if goal and ready and stopped and self.task['state'] == 'running':
                self.task.update(state='completed', progress=100., reason='到达终点并已停稳')
                if self.brush['state'] == 'on':
                    self._set_brush('off', source_time, '任务完成，刷具关闭')
                self._finish_report()
        if self.pending:
            action = self.receipts[self.pending]['action']
            if action in ('dispatch', 'resume'):
                confirmed = self.task is not None and self.task['state'] == 'running' and speed > .05 and ready
                description = '车辆反馈已开始运动' if action == 'dispatch' else '车辆反馈已恢复运动'
            else:
                confirmed = stopped
                description = '车辆已反馈制动，连续停稳 0.5 仿真秒'
            if confirmed:
                self.transition(self.pending, 'succeeded', description, source_time)
                self.pending = None

    def snapshot(self, source_time):
        self.sequence += 1
        capabilities = list(ACTIONS)
        if self.map and not self.route and getattr(self, 'progress', 0.) >= self.map['route']['length']-.3:
            capabilities.remove('dispatch')
        return dict(protocol=PROTOCOL, sessionId=self.session_id, sequence=self.sequence,
            sourceTime=source_time, frameId='map', map=self.map, task=deepcopy(self.task),
            receipts=deepcopy(list(self.receipts.values())), ready=self.ready,
            connected=self.connected, hold=self.hold, reason=self.reason,
            capabilities=capabilities, speedLimit=self.speed_limit,
            toolsAvailable=False, brushState=deepcopy(self.brush), feedbackTime=self.feedback_stamp)
