"""Typed curb chains from supported height runs in one filtered scan band."""
from dataclasses import dataclass
import math

import numpy as np

BOUNDARY_TYPES = ('RAISED', 'FLUSH', 'RAMP')
RAISED, FLUSH, RAMP = range(3)
SCORE_TERMS = ('height', 'support', 'persistence', 'history', 'semantic')


@dataclass(frozen=True)
class CurbConfig:
    cell_m: float = .05
    ground_window_m: float = .30
    ground_slope_max: float = .30
    ground_residual_max_m: float = .025
    step_min_m: float = .06
    step_max_m: float = .25
    step_nominal_m: float = .12
    step_run_max_m: float = .12
    flush_max_m: float = .02
    ramp_run_min_m: float = .15
    ramp_run_max_m: float = .60
    ramp_slope_min: float = .10
    ramp_slope_max: float = .80
    noise_multiplier: float = 3.
    range_increment: float = .001
    density_increment_m: float = .01
    min_ground_points: int = 6
    min_raised_points: int = 3
    min_cell_points: int = 2
    sustain_cells: int = 2
    min_chain_points: int = 3
    max_points: int = 60000
    max_cells: int = 4096
    max_candidates_per_column: int = 8
    max_lateral_step_m: float = .15
    continuity_weight: float = .4
    history_scale_m: float = .20
    semantic_min_score: float = .5
    score_weights: tuple = (.20, .20, .15, .25, .20)
    simplify_m: float = .003
    vertex_spacing_m: float = .10
    max_turn_deg: float = 30.


@dataclass
class CurbSegment:
    boundary_type: str
    xyz: np.ndarray
    sigma_m: np.ndarray
    score: np.ndarray


@dataclass
class CurbResult:
    segments: list
    reason: str
    counts: dict
    diagnostics: dict | None = None


def _hull(points):
    ordered = sorted(set(map(tuple, points)))
    if len(ordered) < 3:
        return np.empty((0, 2))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    result = np.asarray(lower[:-1]+upper[:-1])
    return result if len(result) >= 3 else np.empty((0, 2))


def _inside(point, hull):
    if len(hull) < 3:
        return False
    edges = np.roll(hull, -1, axis=0)-hull
    delta = point-hull
    return bool(np.all(edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0] >= -1e-10))


class _Grid:
    def __init__(self, points, ground, keep, band, side, config):
        self.points, self.ground, self.keep = points[keep], ground[keep], keep
        self.side, self.config, self.cells, self.fits = side, config, {}, {}
        xy = np.column_stack((self.points[:, 0]-band[0], side*self.points[:, 1]-band[2]))
        keys = np.floor(xy/config.cell_m+1e-9).astype(np.int64)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys[order], axis=0), axis=1))+1]
        for ids in np.split(order, starts[1:]):
            if len(ids):
                self.cells[tuple(keys[ids[0]])] = ids

    def plane(self, column, lateral, query):
        config = self.config
        width = math.ceil(config.ground_window_m/config.cell_m)
        groups = [self.cells[(x, y)] for x in range(column-1, column+2)
                  for y in range(lateral-width+1, lateral+1) if (x, y) in self.cells]
        ids = np.concatenate(groups) if groups else np.empty(0, dtype=int)
        ids = np.sort(ids[self.ground[ids]])
        key = ids.tobytes()
        if key not in self.fits:
            self.fits[key] = None
            if len(ids) >= config.min_ground_points:
                points = self.points[ids]
                anchor = points.mean(axis=0)
                relative = points-anchor
                design = np.column_stack((relative[:, :2], np.ones(len(points))))
                # Low face returns can pass the coarse ground mask. Seed on the
                # lower half, then fit only the supported plane's residual inliers.
                seed = np.argsort(points[:, 2])[:max(config.min_ground_points, len(points)//2)]
                coefficient, _, rank, _ = np.linalg.lstsq(design[seed], relative[seed, 2], rcond=None)
                residuals = relative[:, 2]-design @ coefficient
                centre = np.median(residuals)
                scale = 1.4826*np.median(abs(residuals-centre))
                inliers = abs(residuals-centre) <= max(3*scale, 1e-9)
                if rank == 3 and inliers.sum() >= config.min_ground_points:
                    coefficient, _, rank, _ = np.linalg.lstsq(design[inliers], relative[inliers, 2], rcond=None)
                    residual = float(np.max(abs(design[inliers] @ coefficient-relative[inliers, 2])))
                    hull = _hull(relative[inliers, :2])
                    if (rank == 3 and len(hull) and residual <= config.ground_residual_max_m
                            and np.linalg.norm(coefficient[:2]) <= config.ground_slope_max):
                        self.fits[key] = dict(anchor=anchor, coefficient=coefficient, hull=hull,
                                              residual=residual, ids=ids[inliers])
        fit = self.fits[key]
        return fit if fit is not None and _inside(query[:2]-fit['anchor'][:2], fit['hull']) else None

    def records(self):
        return [dict(cell=(int(column), int(lateral)), point_count=len(ids),
                     z_range_m=(float(self.points[ids, 2].min()), float(self.points[ids, 2].max())),
                     point_ids=self.keep[ids]) for (column, lateral), ids in self.cells.items()]


def _height(points, fit):
    relative = points-fit['anchor']
    return relative[:, 2]-relative[:, :2] @ fit['coefficient'][:2]-fit['coefficient'][2]


def _prior_score(point, priors, config):
    best = math.inf
    for line in priors:
        a, b = line[:-1], line[1:]
        dx = b[:, 0]-a[:, 0]
        hit = ((np.minimum(a[:, 0], b[:, 0]) <= point[0]) & (point[0] <= np.maximum(a[:, 0], b[:, 0]))
               & (np.abs(dx) > 1e-9))
        if hit.any():
            y = a[hit, 1]+(point[0]-a[hit, 0])/dx[hit]*(b[hit, 1]-a[hit, 1])
            best = min(best, float(np.min(np.abs(point[1]-y))))
    return math.exp(-.5*(best/config.history_scale_m)**2) if best < math.inf else 0.


def _band_score(point, bands, side):
    lateral, scores = side*point[1], []
    for xmin, xmax, lo, hi, score in bands:
        if xmin <= point[0] <= xmax and lo <= lateral <= hi:
            scores.append(score*(1.-.5*abs(lateral-(lo+hi)/2)/max((hi-lo)/2, 1e-9)))
    return max(scores, default=0.)


def _cell_level(heights, threshold):
    """Use the observed top of a mixed vertical face, otherwise the median."""
    top = float(np.percentile(heights, 90))
    if heights.max()-heights.min() >= threshold and top >= threshold:
        return top
    return float(np.median(heights))


def _profile(grid, column, lateral, anchor, fit, threshold, bands):
    config, side, points, ground = grid.config, grid.side, grid.points, grid.ground
    outward = []
    for y in range(lateral+1, lateral+math.ceil(config.ramp_run_max_m/config.cell_m)+2):
        current = grid.cells.get((column, y))
        if current is None or len(current) < config.min_cell_points:
            break
        outward.append((y, current, _cell_level(_height(points[current], fit), threshold)))
    if not outward:
        return None
    first_y, first_ids, first_height = outward[0]
    first_anchor = points[first_ids].mean(axis=0)
    run = float(side*(first_anchor[1]-anchor[1]))
    if threshold <= first_height <= config.step_max_m and 0 < run <= config.step_run_max_m:
        supporting = [row for row in outward[:config.sustain_cells] if threshold <= row[2] <= config.step_max_m]
        if len(supporting) == config.sustain_cells:
            return RAISED, supporting, first_ids
    if 0 < abs(first_height) < threshold:
        previous, previous_distance, direction = 0., 0., 1. if first_height > 0 else -1.
        for position, (_, current, height) in enumerate(outward):
            distance = float(side*(points[current, 1].mean()-anchor[1]))
            increment = height-previous
            slope = direction*increment/max(distance-previous_distance, 1e-9)
            if (slope > config.ramp_slope_max or (position > 0 and slope < config.ramp_slope_min)
                    or abs(increment) >= threshold or abs(height) > config.step_max_m):
                break
            previous, previous_distance = height, distance
            if (position >= 2 and direction*height >= threshold
                    and config.ramp_run_min_m <= distance <= config.ramp_run_max_m
                    and config.ramp_slope_min <= abs(height)/distance <= config.ramp_slope_max):
                return RAMP, outward[:position+1], first_ids
    if abs(first_height) <= config.flush_max_m:
        # A small first increment may begin a ramp: several observed cells must
        # stay level before a road-band seam counts as a flush boundary.
        upper_ground = first_ids[ground[first_ids]]
        sustained = outward[:config.sustain_cells]
        if (len(upper_ground) >= config.min_cell_points and len(sustained) == config.sustain_cells
                and all(abs(row[2]) <= config.flush_max_m for row in sustained)
                and _band_score((anchor+first_anchor)/2, bands, side) > 0
                and grid.plane(column, first_y, points[upper_ground].mean(axis=0)) is not None):
            return FLUSH, sustained, first_ids
    return None


def _candidate(grid, column, lateral, ids, origin, priors, bands, debug):
    config, side, points = grid.config, grid.side, grid.points
    lower = ids[grid.ground[ids]]
    if len(lower) < config.min_cell_points:
        return None
    anchor = points[lower].mean(axis=0)
    fit = grid.plane(column, lateral, anchor)
    if fit is None:
        return None
    lower = lower[np.isin(lower, fit['ids'])]
    if len(lower) < config.min_cell_points:
        return None
    anchor = points[lower].mean(axis=0)
    noise = fit['residual']
    density = min(1., len(lower)/(2.*config.min_cell_points))
    sensor_range = float(np.linalg.norm(anchor-origin))
    threshold = (config.step_min_m+config.noise_multiplier*noise
                 +config.range_increment*sensor_range+config.density_increment_m*(1.-density))
    profile = _profile(grid, column, lateral, anchor, fit, threshold, bands)
    if profile is None:
        return None
    kind, supporting, first_ids = profile
    support_ids = np.concatenate([row[1] for row in supporting])
    raised_ids, first_support = np.empty(0, dtype=int), first_ids
    if kind == RAISED:
        heights = _height(points[support_ids], fit)
        raised_ids = support_ids[(heights >= threshold) & (heights <= config.step_max_m)]
        bracket_ids = np.r_[ids, first_ids]
        heights = _height(points[bracket_ids], fit)
        lower = bracket_ids[grid.ground[bracket_ids] & (abs(heights) <= config.noise_multiplier*noise+1e-9)]
        first_support = bracket_ids[(heights >= threshold) & (heights <= config.step_max_m)]
        if len(raised_ids) < config.min_raised_points or not len(first_support):
            return None
        anchor = points[lower].mean(axis=0)
    # The root lies in the observed lower/upper bracket; its Z is the measured lower ground.
    near = float(np.max(side*points[lower, 1]))
    far = float(np.min(side*points[first_support, 1]))
    if far < near or far-near > config.step_run_max_m:
        return None
    point = np.array((anchor[0], side*(near+far)/2, anchor[2]))
    height = max((row[2] for row in supporting), key=abs)
    terms = (max(0., 1.-abs(height-config.step_nominal_m)/config.step_max_m) if kind == RAISED else 1.,
             min(1., len(fit['ids'])/(2.*config.min_ground_points))*(1.-noise/config.ground_residual_max_m),
             min(1., len(supporting)/config.sustain_cells),
             _prior_score(point, priors, config), _band_score(point, bands, side))
    sigma, score = float(np.hypot((far-near)/2, noise)), float(np.dot(config.score_weights, terms))
    detail = None if not debug else dict(column=int(column), point=point, boundary_type=BOUNDARY_TYPES[kind],
        sigma_m=sigma, step_height_m=float(height), score=score, score_terms=dict(zip(SCORE_TERMS, terms)),
        thresholds=dict(step_min_m=float(threshold), step_max_m=config.step_max_m, flush_max_m=config.flush_max_m,
                        ground_noise_m=noise, ground_density=density, range_m=sensor_range,
                        step_run_max_m=config.step_run_max_m),
        ground_ids=grid.keep[fit['ids']], raised_ids=grid.keep[raised_ids], support_ids=grid.keep[support_ids])
    return (column, point[0], point[1], point[2], kind, sigma, score), detail


def _select_chains(groups, y, score, config):
    """First-order dynamic programming over columns; a missing column terminates support."""
    chains, layers, previous = [], [], None
    def finish():
        if not layers:
            return
        index = int(np.argmax(layers[-1][1]))
        chain = []
        for ids, _, parents in reversed(layers):
            chain.append(ids[index])
            index = parents[index]
        chains.append(np.array(chain[::-1]))
        layers.clear()
    for column, ids in groups:
        if previous is not None and column != previous+1:
            finish()
        scores = score[ids].copy()
        parents = np.full(len(ids), -1, dtype=int)
        if layers:
            old, old_scores, _ = layers[-1]
            distance = np.abs(y[ids][:, None]-y[old][None])
            transitions = old_scores[None]-config.continuity_weight*distance/config.max_lateral_step_m
            transitions[distance > config.max_lateral_step_m] = -np.inf
            parents = np.argmax(transitions, axis=1)
            best = transitions[np.arange(len(ids)), parents]
            if np.isfinite(best).any():
                scores += best
            else:
                finish()
                parents[:] = -1
        layers.append((ids, scores, parents))
        previous = column
    finish()
    return chains


def _simplify(xy, config):
    """Indices of the vertices kept between measured end points; detail survives at curves."""
    if len(xy) <= 3:
        return np.arange(len(xy))
    selected, pending = {0, len(xy)-1}, [(0, len(xy)-1)]
    while pending:
        start, end = pending.pop()
        if end-start <= 1:
            continue
        delta = xy[end]-xy[start]
        distance = np.linalg.norm(delta)
        errors = np.linalg.norm(xy[start+1:end]-xy[start], axis=1) if distance <= 1e-12 else np.abs(
            delta[0]*(xy[start+1:end, 1]-xy[start, 1])-delta[1]*(xy[start+1:end, 0]-xy[start, 0]))/distance
        if errors.max() > config.simplify_m:
            index = start+1+int(np.argmax(errors))
        elif distance > config.vertex_spacing_m:
            index = (start+end)//2
        else:
            continue
        selected.add(index)
        pending.extend(((start, index), (index, end)))
    if len(selected) < 3:
        selected.add(len(xy)//2)
    return np.array(sorted(selected))


def _segments(table, chain, config):
    segments = []
    for piece in np.split(chain, np.flatnonzero(np.diff(table[chain, 4]))+1):
        # Heading uses several measured columns so that one lateral cell step
        # of quantisation does not cut a gentle curve into pieces.
        xy, span, cuts = table[piece, 1:3], config.min_chain_points-1, [0]
        for pivot in range(span, len(piece)-span):
            a, b = xy[pivot]-xy[pivot-span], xy[pivot+span]-xy[pivot]
            cosine = np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12)
            if pivot-cuts[-1] >= span and np.degrees(np.arccos(np.clip(cosine, -1, 1))) > config.max_turn_deg:
                cuts.append(pivot)
        cuts.append(len(piece)-1)
        for start, end in zip(cuts, cuts[1:]):
            part = piece[start:end+1]
            if len(part) >= config.min_chain_points:
                part = part[_simplify(table[part, 1:3], config)]
                segments.append(CurbSegment(BOUNDARY_TYPES[int(table[part[0], 4])], table[part, 1:4],
                                            table[part, 5], table[part, 6]))
    return segments


def detect_curb(points_xyz, ground_mask, band, side, *, sensor_origin=(0., 0., 0.), prior_xy=(),
                road_bands=(), config=None, debug=False):
    """Typed root chains inside ``band`` = (xmin, xmax, outward_min, outward_max) on ``side`` +1 left / -1 right.

    ``prior_xy`` holds (m, 2) base_link polylines and ``road_bands`` holds
    (xmin, xmax, outward_lo, outward_hi, score) tuples; both only re-rank candidates.
    """
    config = config or CurbConfig()
    points, ground = np.asarray(points_xyz, dtype=float), np.asarray(ground_mask, dtype=bool)
    band, origin = np.asarray(band, dtype=float), np.asarray(sensor_origin, dtype=float)
    assert points.shape[1:] == (3,) and ground.shape == points.shape[:1] and band.shape == (4,)
    counts = dict(input_points=len(points), band_points=0, cells=0, plane_fits=0, candidates=0)
    shape = np.ceil((band[[1, 3]]-band[[0, 2]])/config.cell_m).astype(int)
    if int(shape[0])*int(shape[1]) > config.max_cells:
        return CurbResult([], 'CELL_BUDGET_EXCEEDED', counts)
    keep = np.flatnonzero((points[:, 0] >= band[0]) & (points[:, 0] < band[1])
                          & (side*points[:, 1] >= band[2]) & (side*points[:, 1] < band[3]))
    counts['band_points'] = len(keep)
    if len(keep) > config.max_points:
        return CurbResult([], 'POINT_BUDGET_EXCEEDED', counts)
    priors = [np.asarray(line, dtype=float) for line in prior_xy]
    bands = [tuple(map(float, row)) for row in road_bands if row[4] >= config.semantic_min_score]
    grid = _Grid(points, ground, keep, band, side, config)
    rows, details = [], []
    for (column, lateral), ids in grid.cells.items():
        found = _candidate(grid, column, lateral, ids, origin, priors, bands, debug)
        if found is not None:
            rows.append(found[0])
            details.append(found[1])
    table = np.array(rows, dtype=float).reshape(-1, 7)
    order = np.lexsort((side*table[:, 2], -table[:, 6], table[:, 0]))
    table = table[order]
    columns, starts = np.unique(table[:, 0], return_index=True)
    groups = [(int(column), np.arange(start, min(end, start+config.max_candidates_per_column)))
              for column, start, end in zip(columns, starts, np.r_[starts[1:], len(table)])]
    selected = np.concatenate([ids for _, ids in groups]) if groups else np.empty(0, dtype=int)
    segments = [segment for chain in _select_chains(groups, table[:, 2], table[:, 6], config)
                for segment in _segments(table, chain, config)]
    counts.update(cells=len(grid.cells), candidates=len(selected),
                  plane_fits=sum(fit is not None for fit in grid.fits.values()))
    diagnostics = dict(candidates=[details[order[i]] for i in selected], cells=grid.records()) if debug else None
    return CurbResult(segments, 'MEASURED_CHAINS' if segments else 'NO_SUPPORTED_BOUNDARY', counts, diagnostics)
