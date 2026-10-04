"""Boustrophedon (往复扫掠) coverage path generator; no Fields2Cover dependency.

Pure geometry over a map-frame polygon. Nothing here reads CARLA ground truth
and nothing knows about obstacles: the ``allowed`` region is only the work
polygon buffered outward by ``turn_radius + ALLOWED_MARGIN_M``; it is a planning
corridor for the turns, not free-space knowledge.

Row (swath) lines are placed across the polygon perpendicular to the sweep
direction (default: the polygon's longest edge). Each row line is clipped by the
polygon into one or more work segments; consecutive rows whose segments overlap
form a cell that is swept back and forth. Rows are joined by:

- semicircle of radius ``gap/2`` when ``gap >= 2*turn_radius`` (else a flat
  arc-straight-arc turn of radius ``turn_radius`` if the semicircle leaves
  ``allowed``);
- otherwise an omega (Dubins RLR, three arcs of ``turn_radius``) bulb beyond the
  row end when it fits inside ``allowed``;
- otherwise a fishtail (forward arc, reverse arc, forward straight) whose reverse
  arc carries ``direction=-1``; the driver separates direction switches by a stop.

Rows are exactly ``row_spacing`` apart, centred on the polygon extent (see
``_rows``). Within a cell the rows are visited in ``skip-row`` order first (every
other row, then the rest) whenever that avoids a reverse segment; otherwise
sequentially. Transits between cells are shortest paths through a visibility graph
over the ``allowed`` polygon vertices.

Every joining segment is verified against ``allowed`` and the planner raises
``ValueError`` instead of emitting a path that leaves it. The approach from the
start pose to the first row is a straight non-work segment and is *not* checked
(the vehicle may be far from the area).

Output poses follow the ``console_session`` route contract
``[x, y, yaw, direction, work_segment]`` sampled every ``STEP_M``.
"""
import math

try:
    from shapely.geometry import LineString, MultiLineString, Point, Polygon
except ImportError:  # The bridge image does not carry the optional geometry stack.
    LineString = MultiLineString = Point = Polygon = None

STEP_M = .2
ALLOWED_MARGIN_M = .5           # turn corridor beyond the polygon, added to turn_radius
OVERLAP_FRACTION = .1           # swath = five-brush lateral width minus 10 % overlap
PLANNER = 'boustrophedon'
_EPS = 1e-6


def _require_shapely():
    if Polygon is None:
        raise RuntimeError(
            '往复区域规划需要 Shapely；当前桥接运行时未安装可选几何依赖')


def tricycle_tool_geometry():
    """Swath width and minimum turn radius of the tricycle, with their sources."""
    from console_session import BRUSH_SOURCE, tricycle_vehicle
    vehicle = tricycle_vehicle()
    width = vehicle.coverage_width
    if not width:
        raise ValueError('车辆刷具足迹缺失，无法推导扫掠宽度；请显式传入 swath_width')
    # front_steered_tricycle: curvature = tan(axle_limit)/wheelbase (coverage_geometry.VehicleGeometry)
    radius = 1./vehicle.curvature_limit
    return dict(
        swath_width=width*(1.-OVERLAP_FRACTION), turn_radius=radius,
        swath_source='%s lateral extent %.3f m minus %d %% overlap' % (BRUSH_SOURCE, width, OVERLAP_FRACTION*100),
        turn_radius_source='tricycle_navigation.coverage_vehicle(load_profile()): wheelbase %.3f m / tan(max_steer %.3f rad) '
                           'via dg202604 coverage_geometry.VehicleGeometry.curvature_limit' % (vehicle.wheelbase, vehicle.axle_limit),
        vehicle_source=dict(vehicle.source_hashes), brushes=list(vehicle.brushes))


def _wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def _polygon(polygon):
    _require_shapely()
    if not isinstance(polygon, (list, tuple)) or not 3 <= len(polygon) <= 64:
        raise ValueError('作业区多边形须为 3–64 个顶点')
    shape = Polygon([(float(x), float(y)) for x, y in polygon])
    if not shape.is_valid or shape.area <= 0:
        raise ValueError('作业区多边形自相交或无效')
    return shape


def _longest_edge_angle(shape):
    coords = list(shape.exterior.coords)
    a, b = max(zip(coords, coords[1:]), key=lambda e: math.dist(*e))
    return math.atan2(b[1]-a[1], b[0]-a[0]) % math.pi


def _rows(shape, swath, spacing):
    """Row v-coordinates, exactly ``spacing`` apart and centred on the polygon extent.

    n = ceil((extent-swath)/spacing)+1 rows; the set is shifted so that the first
    and last rows overhang the extent equally (never more than swath/2).
    """
    _, vmin, _, vmax = shape.bounds
    extent = vmax-vmin
    if extent <= swath+_EPS:
        return [(vmin+vmax)/2]
    n = math.ceil((extent-swath)/spacing-_EPS)+1
    first = (vmin+vmax)/2-(n-1)*spacing/2
    return [first+k*spacing for k in range(n)]


def _segments(shape, v):
    umin, _, umax, _ = shape.bounds
    cut = LineString([(umin-1., v), (umax+1., v)]).intersection(shape)
    parts = list(cut.geoms) if isinstance(cut, MultiLineString) else [cut]
    out = []
    for part in parts:
        if isinstance(part, LineString) and part.length > STEP_M/4:
            us = [c[0] for c in part.coords]
            out.append((min(us), max(us)))
    return sorted(out)


def _cells(rows):
    """Group row segments into boustrophedon cells (consecutive rows, overlapping u-range)."""
    done, open_ = [], []
    for k, (v, segs) in enumerate(rows):
        matches = [[c for c in open_ if c[-1][0] == k-1 and c[-1][2] > s[0] and s[1] > c[-1][1]] for s in segs]
        kept = []
        for s, cs in zip(segs, matches):
            if len(cs) == 1 and sum(cs[0] in other for other in matches) == 1:
                cs[0].append((k, s[0], s[1], v))
                kept.append(cs[0])
            else:
                kept.append([(k, s[0], s[1], v)])
        done.extend(c for c in open_ if c not in kept)
        open_ = kept
    return done+open_


def _arc(centre, radius, a0, a1, yaw_offset, direction, work=False):
    n = max(1, math.ceil(radius*abs(a1-a0)/STEP_M))
    return [[centre[0]+radius*math.cos(a0+(a1-a0)*i/n), centre[1]+radius*math.sin(a0+(a1-a0)*i/n),
             a0+(a1-a0)*i/n+yaw_offset, direction, work] for i in range(n+1)]


def _straight(a, b, yaw=None, direction=1, work=False):
    n = max(1, math.ceil(math.dist(a, b)/STEP_M))
    yaw = math.atan2(b[1]-a[1], b[0]-a[0]) if yaw is None else yaw
    return [[a[0]+(b[0]-a[0])*i/n, a[1]+(b[1]-a[1])*i/n, yaw, direction, work] for i in range(n+1)]


def _turn_candidates(d, radius):
    """Canonical turns from (0,0,yaw 0) to (0,d,yaw pi), all radii >= ``radius``."""
    if d >= 2*radius-_EPS:
        r = d/2
        yield 'semicircle', _arc((0., r), r, -math.pi/2, math.pi/2, math.pi/2, 1)
        first = _arc((0., radius), radius, -math.pi/2, 0., math.pi/2, 1)          # flat turn: arc, straight, arc
        yield 'flat', first+_straight(first[-1][:2], (radius, d-radius), math.pi/2)[1:] \
            + _arc((0., d-radius), radius, 0., math.pi/2, math.pi/2, 1)[1:]
        return
    c1, c3 = (0., -radius), (0., d+radius)
    c2 = (math.sqrt(max(0., 4*radius**2-(radius+d/2)**2)), d/2)
    a12, a23 = math.atan2(c2[1]-c1[1], c2[0]-c1[0]), math.atan2(c3[1]-c2[1], c3[0]-c2[0])
    yield 'omega', (_arc(c1, radius, math.pi/2, a12, -math.pi/2, 1)
                    + _arc(c2, radius, a12+math.pi, a23+2*math.pi, math.pi/2, 1)[1:]
                    + _arc(c3, radius, a23+math.pi, -math.pi/2, -math.pi/2, 1)[1:])
    alpha = math.acos(-d/(2*radius))
    c1 = (0., radius)
    forward = _arc(c1, radius, -math.pi/2, -math.pi/2+alpha, math.pi/2, 1)
    p1 = forward[-1]
    c2 = (p1[0]+radius*math.sin(alpha), p1[1]-radius*math.cos(alpha))
    n = max(1, math.ceil(radius*(math.pi-alpha)/STEP_M))
    reverse = [[c2[0]-radius*math.sin(t), c2[1]+radius*math.cos(t), t, -1, False]
               for t in (alpha+(math.pi-alpha)*i/n for i in range(n+1))]
    yield 'fishtail', forward+reverse[1:]+_straight(reverse[-1][:2], (0., d), math.pi)[1:]


def _place(canonical, origin, su, sv):
    return [[origin[0]+su*a, origin[1]+sv*b, math.atan2(sv*math.sin(yaw), su*math.cos(yaw)), direction, work]
            for a, b, yaw, direction, work in canonical]


def _inside(allowed, poses):
    return all(allowed.covers(_point(p)) for p in poses)


def _transit(allowed, a, b):
    """Shortest polyline from a to b inside ``allowed`` (visibility graph, Dijkstra)."""
    if allowed.covers(LineString([a, b])):
        return [a, b]
    nodes = [tuple(a), tuple(b)]+[tuple(v) for v in allowed.exterior.coords[:-1]]
    visible = lambda p, q: allowed.covers(LineString([p, q]))
    dist, prev, todo = {0: 0.}, {}, {0}
    done = set()
    while todo:
        i = min(todo, key=dist.get)
        todo.remove(i)
        done.add(i)
        if i == 1:
            break
        for j, q in enumerate(nodes):
            if j in done or not visible(nodes[i], q):
                continue
            alt = dist[i]+math.dist(nodes[i], q)
            if alt < dist.get(j, math.inf):
                dist[j], prev[j] = alt, i
                todo.add(j)
    if 1 not in dist:
        raise ValueError('单元间过渡段无法在 allowed 区内连通')
    path, i = [], 1
    while True:
        path.append(nodes[i])
        if i == 0:
            return path[::-1]
        i = prev[i]


def _point(p):
    return Point(p[0], p[1])


def _forward_order(n, min_gap):
    """Visiting order of rows 0..n-1 whose consecutive index gaps are all >= ``min_gap``.

    Starting rows are tried nearest-first (row 0 is nearest the vehicle); within a start,
    depth-first with the smallest admissible gap first. None when no order exists.
    """
    for first in range(n):
        order, used = [first], {first}

        def extend():
            if len(order) == n:
                return True
            last = order[-1]
            for k in sorted(range(n), key=lambda k: (abs(k-last), k)):
                if k in used or abs(k-last) < min_gap:
                    continue
                order.append(k); used.add(k)
                if extend():
                    return True
                order.pop(); used.remove(k)
            return False
        if extend():
            return order
    return None


def _orderings(rows, spacing, radius, requested=None):
    """Candidate visiting orders for one cell's rows (rows are consecutive; row 0 nearest).

    ``skip-row``: an order whose consecutive rows are far enough apart (>= 2*radius) for a
    forward semicircle/flat turn, so no reverse segment is needed; ``sequential``: 0,1,2,...
    """
    n = len(rows)
    orders = [('sequential', list(rows))]
    min_gap = math.ceil((2*radius-_EPS)/spacing)
    if min_gap > 1:
        found = _forward_order(n, min_gap)
        if found is not None:
            orders.insert(0, ('skip-row', [rows[i] for i in found]))
    if requested is not None:
        orders = [o for o in orders if o[0] == requested]
        if not orders:
            raise ValueError('无法对该单元使用行序 %s' % requested)
    return orders


def _sweep_cell(order, here, su, allowed, radius):
    """Sweep ``order`` (list of (k, u0, u1, v)) alternating direction from ``here``.

    Returns (poses, turn_types, row_gaps, has_reverse); raises ValueError when a join
    cannot be placed inside ``allowed``.
    """
    poses, types, gaps = [], [], []
    for i, (_, u0, u1, v) in enumerate(order):
        a, b = ((u0, v), (u1, v)) if su > 0 else ((u1, v), (u0, v))
        row = _straight(a, b, 0. if su > 0 else math.pi, 1, True)
        if i == 0 and math.dist(here, a) > _EPS:       # skip-row may start at a row other than the nearest
            corners = _transit(allowed, here, a)
            for p, q in zip(corners, corners[1:]):
                poses.extend(_straight(p, q)[1:] if poses else _straight(p, q))
        if poses and math.dist(poses[-1][:2], a) <= _EPS:
            poses.pop()                                # the turn end pose becomes the row's first work pose
        poses.extend(row)
        if i+1 == len(order):
            break
        _, n0, n1, nv = order[i+1]
        next_start = n1 if su > 0 else n0              # next row is swept the opposite way
        overshoot = su*(next_start-b[0])
        end = b
        if overshoot > _EPS:                           # extend along this row before turning
            end = (next_start, v)
            ext = _straight(b, end, 0. if su > 0 else math.pi)
            if not _inside(allowed, ext):
                raise ValueError('行末延伸段离开 allowed 区')
            poses.extend(ext[1:])
        d = abs(nv-v)
        sv = 1 if nv > v else -1
        for name, canonical in _turn_candidates(d, radius):
            turn = _place(canonical, end, su, sv)
            if _inside(allowed, turn):
                break
        else:
            raise ValueError('行距 %.3f m、转弯半径 %.3f m 的换行段无法落在 allowed 区内' % (d, radius))
        types.append(name)
        gaps.append(d)
        poses.extend(turn[1:])
        if overshoot < -_EPS:                          # back along the next row to its start
            poses.extend(_straight(turn[-1][:2], (next_start, nv), math.pi if su > 0 else 0.)[1:])
        su = -su
    return poses, types, gaps, any(p[3] == -1 for p in poses)


def plan_boustrophedon(polygon, start, swath_width, row_spacing=None, turn_radius=None, sweep_angle=None, ordering=None):
    """Return dict(poses, pathLength, rows, swathWidth, rowSpacing, turnRadius, allowedPolygon, planner).

    ``polygon``: map-frame [[x, y], ...]; ``start``: (x, y, yaw); widths/radii in metres.
    ``ordering``: None (auto: prefer a forward-only order), 'skip-row' or 'sequential'.
    """
    if turn_radius is None or not turn_radius > 0:
        raise ValueError('须显式给出正的最小转弯半径 turn_radius')
    if not swath_width > 0:
        raise ValueError('须显式给出正的扫掠宽度 swath_width')
    spacing = swath_width if row_spacing is None else float(row_spacing)
    if not 0 < spacing <= swath_width+_EPS:
        raise ValueError('行距须为正且不大于扫掠宽度')
    shape = _polygon(polygon)
    theta = _longest_edge_angle(shape) if sweep_angle is None else float(sweep_angle) % math.pi
    c, s = math.cos(theta), math.sin(theta)
    to_local = lambda x, y: (c*x+s*y, -s*x+c*y)
    local = Polygon([to_local(x, y) for x, y in shape.exterior.coords])
    margin = turn_radius+ALLOWED_MARGIN_M
    allowed = local.buffer(margin, join_style=2)
    rows = [(v, _segments(local, v)) for v in _rows(local, swath_width, spacing)]
    cells = _cells(rows)
    if not cells:
        raise ValueError('多边形内没有可扫掠的行')

    start_local = (*to_local(start[0], start[1]), start[2]-theta)
    poses = [[start_local[0], start_local[1], start_local[2], 1, False]]
    turn_types, row_gaps, orderings, work_rows = [], [], [], 0
    remaining = list(cells)

    def entry_options(cell):
        first, last = cell[0], cell[-1]
        return [((first[1], first[3]), 1, 1), ((first[2], first[3]), -1, 1),
                ((last[1], last[3]), 1, -1), ((last[2], last[3]), -1, -1)]

    while remaining:
        here = poses[-1][:2]
        cell, (entry, su, sv) = min(((cell, opt) for cell in remaining for opt in entry_options(cell)),
                                     key=lambda co: (math.dist(here, co[1][0]), remaining.index(co[0]), co[1][1], co[1][2]))
        remaining.remove(cell)
        base = cell if sv > 0 else cell[::-1]
        if math.dist(here, entry) > _EPS:
            corners = [here, entry] if len(poses) == 1 else _transit(allowed, here, entry)
            for p, q in zip(corners, corners[1:]):
                poses.extend(_straight(p, q)[1:])
        attempts = []
        for name, order in _orderings(base, spacing, turn_radius, ordering):
            try:
                attempts.append((name, _sweep_cell(order, poses[-1][:2], su, allowed, turn_radius)))
            except ValueError as error:
                attempts.append((name, error))
        chosen = next((a for a in attempts if not isinstance(a[1], Exception) and not a[1][3]), None) \
            or next((a for a in attempts if not isinstance(a[1], Exception)), None)
        if chosen is None:
            raise attempts[0][1]
        name, (cell_poses, types, gaps, _) = chosen
        if math.dist(poses[-1][:2], cell_poses[0][:2]) <= _EPS:
            poses.pop()                                # the transit end pose becomes the first work pose
        poses.extend(cell_poses)
        work_rows += len(base)
        turn_types.extend(types)
        row_gaps.extend(gaps)
        orderings.append(name)

    def to_map(p):
        return [c*p[0]-s*p[1], s*p[0]+c*p[1], _wrap(p[2]+theta), p[3], p[4]]
    out = [to_map(p) for p in poses]
    allowed_map = [list(to_map((x, y, 0., 1, False))[:2]) for x, y in allowed.exterior.coords[:-1]]
    length = sum(math.dist(a[:2], b[:2]) for a, b in zip(out, out[1:]))
    return dict(poses=out, pathLength=length, rows=work_rows, swathWidth=float(swath_width), rowSpacing=spacing,
                turnRadius=float(turn_radius), allowedPolygon=allowed_map, planner=PLANNER,
                sweepAngle=theta, cells=len(cells), turnTypes=turn_types, rowGaps=row_gaps, orderings=orderings)


def plan_area(polygon, start):
    """Gateway adapter: tricycle geometry + ``plan_boustrophedon`` -> console route dict."""
    tool = tricycle_tool_geometry()
    result = plan_boustrophedon(polygon, start, tool['swath_width'], turn_radius=tool['turn_radius'])
    return dict(poses=result['poses'], length=result['pathLength'], allowed=result['allowedPolygon'],
                allowedSource='polygon buffered by turn radius + %g m; no obstacle map' % ALLOWED_MARGIN_M,
                status='planned', reason='rows=%d cells=%d turns=%s ordering=%s' % (result['rows'], result['cells'],
                    ','.join(sorted(set(result['turnTypes']))) or 'none', ','.join(sorted(set(result['orderings'])))),
                planner=PLANNER, swathWidth=result['swathWidth'], rowSpacing=result['rowSpacing'],
                turnRadius=result['turnRadius'], swathSource=tool['swath_source'],
                turnRadiusSource=tool['turn_radius_source'],
                vehicleSource=tool['vehicle_source'], brushes=tool['brushes'])
