"""Six typed evidence layers over overlapping rolling 3-D workspaces.

Measurements alone may clear traversed volume. Semantic predictions, absence
of detections and expired evidence never clear space. Coarse observations are
not expanded into fine free cells. All coordinates are in one odom epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import copy
import math
import time
from threading import Lock, RLock
import numpy as np

from .motion import rigid, transform
from .raycast import traverse
from .ground_query import GroundFit, GroundIndex
from ..heal.semantics import CLASS_NAMES, GRID_ORIGIN_M, GRID_RESOLUTION_M, SCORE_SCALE, validate_grid


@dataclass(frozen=True)
class Evidence:
    observation_id: str
    correlation_group: str
    capture_s: float
    state_s: float
    last_direct_s: float
    valid_until_s: float
    motion_epoch: int
    calibration_version: str
    frame_id: str = 'odom'
    map_version: str = 'unversioned'
    origin: str = 'MEASUREMENT'
    support: str = 'DIRECT'
    metric_valid: bool = True
    covariance_valid: bool = False
    parent_ids: tuple[str, ...] = ()
    available_sources: tuple[str, ...] = ()
    sensor_id: str = ''

    def __post_init__(self):
        if (not self.observation_id or not self.correlation_group or not self.calibration_version
                or self.motion_epoch < 0 or self.frame_id != 'odom'
                or not np.isfinite([self.capture_s, self.state_s, self.last_direct_s, self.valid_until_s]).all()
                or not self.last_direct_s <= self.capture_s <= self.state_s <= self.valid_until_s
                or self.origin not in ('MEASUREMENT', 'MODEL', 'PROPAGATED', 'MAP_PRIOR')):
            raise ValueError('invalid evidence metadata')


@dataclass(frozen=True)
class Scale:
    name: str
    resolution_m: float
    radius_m: float

    def __post_init__(self):
        if not self.name or not np.isfinite([self.resolution_m, self.radius_m]).all() or not 0 < self.resolution_m < self.radius_m:
            raise ValueError('invalid rolling scale')


@dataclass(frozen=True)
class WorldConfig:
    scales: tuple[Scale, ...] = (Scale('near', .15, 8.), Scale('far', .5, 25.))
    occupancy_ttl_s: float = 1.2
    free_ttl_s: float = .6
    ground_ttl_s: float = 1.2
    z_min_m: float = -2.
    z_max_m: float = 4.
    ground_cell_size_m: float = .35
    ground_cache_entries: int = 2048

    def __post_init__(self):
        if (not self.scales or len({s.name for s in self.scales}) != len(self.scales)
                or any(not math.isfinite(v) or v <= 0 for v in (self.occupancy_ttl_s, self.free_ttl_s, self.ground_ttl_s))
                or not np.isfinite([self.z_min_m, self.z_max_m]).all() or self.z_min_m >= self.z_max_m
                or not math.isfinite(self.ground_cell_size_m) or self.ground_cell_size_m <= 0
                or isinstance(self.ground_cache_entries, bool) or not isinstance(self.ground_cache_entries, int)
                or self.ground_cache_entries < 0):
            raise ValueError('invalid world configuration')


class VoxelTable:
    """Packed numeric batch joins avoid a Python object per observed voxel."""
    def __init__(self):
        self.keys = np.empty((0, 3), dtype=np.int64)
        self.scores = np.empty(0, dtype=np.int8)
        self.stamps = np.empty(0)
        self.expires = np.empty(0)
        self.sources = np.empty(0, dtype=np.int64)
        self.hazard_expires = np.empty(0)
        self.hazard_sources = np.empty(0, dtype=np.int64)
        # Optional odom AABBs of complete same-observation finer hit voxels.
        # Only coarse tables allocate this (48 bytes per retained voxel).
        # NaN is an incomplete certificate, never empty space. The box is
        # accumulated for the entire continuous lifetime of a coarse hazard.
        self.hazard_bounds = None
        self.evidence = {}
        self.serial = 0

    def __len__(self):
        return len(self.keys)

    def trim(self, now, center, resolution, radius):
        if not len(self):
            return
        keep = ((self.expires >= now) & (np.max(np.abs(
            (self.keys[:, :2] + .5) * resolution - center[:2]), axis=1) <= radius + resolution))
        for name in ('keys', 'scores', 'stamps', 'expires', 'sources', 'hazard_expires', 'hazard_sources'):
            setattr(self, name, getattr(self, name)[keep])
        if self.hazard_bounds is not None:
            self.hazard_bounds = self.hazard_bounds[keep]
        used = set(np.unique(self.sources).tolist())
        used.update(np.unique(self.hazard_sources[self.hazard_expires >= now]).tolist())
        self.evidence = {key: value for key, value in self.evidence.items() if key in used}

    def update(self, free, hits, meta, now, free_ttl, occupied_ttl, obstacle_hits=None,
               refined_hazard_bounds=None):
        if not len(free) and not len(hits):
            return
        old_count = len(self)
        obstacle_hits = hits if obstacle_hits is None else obstacle_hits
        if refined_hazard_bounds is not None:
            refined_hazard_bounds = np.asarray(refined_hazard_bounds, dtype=float)
            if refined_hazard_bounds.shape != (len(obstacle_hits), 2, 3):
                raise ValueError('one finer hazard envelope per obstacle voxel required')
            complete = np.isfinite(refined_hazard_bounds).all(axis=(1, 2))
            missing = np.isnan(refined_hazard_bounds).all(axis=(1, 2))
            if (not np.all(complete | missing) or np.any(
                    refined_hazard_bounds[complete, 0] > refined_hazard_bounds[complete, 1])):
                raise ValueError('finer envelopes must be finite ordered XYZ boxes or entirely unknown')
        keys = np.concatenate((self.keys, free, hits, obstacle_hits))
        minimum = keys.min(axis=0)
        extent = keys.max(axis=0) - minimum + 1
        if math.prod(map(int, extent)) >= np.iinfo(np.int64).max:
            raise ValueError('rolling voxel span exceeds integer address space')
        relative = keys - minimum
        packed = (relative[:, 0] * extent[1] + relative[:, 1]) * extent[2] + relative[:, 2]
        _, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
        count = len(first)
        scores, stamps, expires = np.zeros(count, dtype=np.int8), np.full(count, -np.inf), np.full(count, -np.inf)
        sources = np.full(count, -1, dtype=np.int64)
        hazard_expires = np.full(count, -np.inf)
        hazard_sources = np.full(count, -1, dtype=np.int64)
        hazard_bounds = (np.full((count, 2, 3), np.nan) if self.hazard_bounds is not None
            or (refined_hazard_bounds is not None and np.isfinite(refined_hazard_bounds).any()) else None)
        continuing_hazard = np.zeros(count, dtype=bool)
        old = inverse[:old_count]
        scores[old], stamps[old], expires[old], sources[old] = self.scores, self.stamps, self.expires, self.sources
        hazard_expires[old] = self.hazard_expires
        hazard_sources[old] = self.hazard_sources
        continuing_hazard[old] = ((self.scores > 0) & (self.expires >= now) & (self.hazard_expires >= now))
        if hazard_bounds is not None and self.hazard_bounds is not None:
            live_old = continuing_hazard[old]
            hazard_bounds[old[live_old]] = self.hazard_bounds[live_old]
        self.serial += 1
        for indices, amount, ttl in ((inverse[old_count:old_count + len(free)], -1, free_ttl),
                                      (inverse[old_count + len(free):old_count + len(free) + len(hits)], 2, occupied_ttl)):
            valid_until = min(meta.valid_until_s, meta.capture_s + ttl)
            if valid_until < now:
                continue
            indices = indices[stamps[indices] <= meta.capture_s]
            new_scores = np.clip(scores[indices] + amount, -2, 3)
            if amount > 0:
                new_scores = np.maximum(new_scores, 2)
            deadlines = np.full(len(indices), valid_until)
            if amount > 0:
                retaining_hazard = continuing_hazard[indices]
                deadlines[retaining_hazard] = np.maximum(deadlines[retaining_hazard],
                    hazard_expires[indices[retaining_hazard]])
            if amount < 0:
                retaining_hit = (new_scores > 0) & np.isfinite(expires[indices])
                deadlines[retaining_hit] = np.minimum(deadlines[retaining_hit], expires[indices[retaining_hit]])
            scores[indices], stamps[indices], expires[indices], sources[indices] = new_scores, meta.capture_s, deadlines, self.serial
        hazard_indices = inverse[old_count + len(free) + len(hits):]
        hazard_deadline = min(meta.valid_until_s, meta.capture_s + occupied_ttl)
        accepted_hazards = (stamps[hazard_indices] <= meta.capture_s) & (hazard_deadline >= now)
        hazard_indices = hazard_indices[accepted_hazards]
        if hazard_bounds is not None:
            incoming = (refined_hazard_bounds[accepted_hazards].copy() if refined_hazard_bounds is not None
                        else np.full((len(hazard_indices), 2, 3), np.nan))
            continuing = continuing_hazard[hazard_indices]
            # NaN propagates: one unrefined contributor prevents later scans
            # from presenting the continuous aggregate as complete. Retaining
            # even expired contributors until this aggregate clears is safe
            # and preserves the existing conservative coarse-hazard lifetime.
            incoming[continuing, 0] = np.minimum(incoming[continuing, 0],
                                                hazard_bounds[hazard_indices[continuing], 0])
            incoming[continuing, 1] = np.maximum(incoming[continuing, 1],
                                                hazard_bounds[hazard_indices[continuing], 1])
            hazard_bounds[hazard_indices] = incoming
        # A shorter new observation cannot expire a still-live old hazard or
        # claim its longer deadline. Keep the actual longest-lived source.
        replace_source = (~continuing_hazard[hazard_indices]
                          | (hazard_expires[hazard_indices] <= hazard_deadline))
        hazard_expires[hazard_indices[replace_source]] = hazard_deadline
        hazard_sources[hazard_indices[replace_source]] = self.serial
        hazard_expires[scores <= 0] = -np.inf
        hazard_sources[scores <= 0] = -1
        if hazard_bounds is not None:
            hazard_bounds[scores <= 0] = np.nan
        keep = expires >= now
        self.keys, self.scores = keys[first][keep], scores[keep]
        self.stamps, self.expires, self.sources = stamps[keep], expires[keep], sources[keep]
        self.hazard_expires = hazard_expires[keep]
        self.hazard_sources = hazard_sources[keep]
        self.hazard_bounds = None if hazard_bounds is None else hazard_bounds[keep]
        self.evidence = {**self.evidence, self.serial: meta}

    def state(self, key, now):
        match = np.flatnonzero(np.all(self.keys == key, axis=1))
        if not len(match):
            return 'UNKNOWN'
        index = match[0]
        if self.expires[index] < now or self.scores[index] == 0:
            return 'UNKNOWN'
        return 'OCCUPIED' if self.scores[index] > 0 else 'FREE'


def _cell_groups(cells):
    """Lexicographic groups and first observations, using numeric scalar sort."""
    if not len(cells):
        return cells.copy(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    minimum = cells.min(axis=0)
    extent = cells.max(axis=0) - minimum + 1
    if math.prod(map(int, extent)) >= np.iinfo(np.int64).max:
        return np.unique(cells, axis=0, return_index=True, return_inverse=True)
    relative = cells - minimum
    keys = relative[:, 0].copy()
    for axis in range(1, cells.shape[1]):
        keys = keys * extent[axis] + relative[:, axis]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return cells[first], first, inverse


def _unique_cells(cells):
    return _cell_groups(cells)[0]


def _volume_regions(regions):
    """Validate caller-supplied odom prisms, retaining polygon holes."""
    import shapely
    from shapely.geometry import Polygon

    if not isinstance(regions, (list, tuple)) or not regions:
        raise ValueError('volume regions must be a nonempty finite list')
    parsed = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict) or not isinstance(region.get('part_id'), str) or not region['part_id'].strip():
            raise ValueError('each volume region requires a part_id')
        try:
            rings = [np.asarray(region['polygon_xy'], dtype=float)]
            holes = region.get('holes_xy', [])
            if not isinstance(holes, (list, tuple)):
                raise ValueError('holes_xy must be a list of rings')
            rings.extend(np.asarray(hole, dtype=float) for hole in holes)
            if any(ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3
                   or not np.isfinite(ring).all() for ring in rings):
                raise ValueError('volume polygon rings require finite XY vertices')
            z_min, z_max, offset = (float(region[name]) for name in ('z_min_m', 'z_max_m', 'offset_s'))
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('invalid volume region') from error
        if not np.isfinite([z_min, z_max, offset]).all() or z_min >= z_max or offset < 0:
            raise ValueError('volume height and arrival offset must be finite and ordered')
        polygon = Polygon(rings[0], rings[1:])
        if polygon.is_empty or not polygon.is_valid or not math.isfinite(polygon.area) or polygon.area <= 0:
            raise ValueError('volume polygon must be valid with positive area')
        shapely.prepare(polygon)
        parsed.append({'region_index': index, 'part_id': region['part_id'], 'polygon': polygon,
            'bounds': polygon.bounds, 'z_min_m': z_min, 'z_max_m': z_max, 'offset_s': offset})
    return parsed


def _volume_voxels(region, resolution, budget):
    """All closed voxel boxes intersecting a prism, including face contacts.

    The budget also bounds bounding-box candidates, so a thin or perforated
    polygon cannot cause unbounded enumeration. Such a query stays UNKNOWN.
    """
    import shapely

    xmin, ymin, xmax, ymax = region['bounds']
    lower = (xmin, ymin, region['z_min_m'])
    upper = (xmax, ymax, region['z_max_m'])
    try:
        first = [math.floor(math.nextafter(value / resolution, -math.inf)) for value in lower]
        last = [math.floor(math.nextafter(value / resolution, math.inf)) for value in upper]
    except (ValueError, OverflowError) as error:
        raise ValueError('volume exceeds voxel address space') from error
    if any(value <= np.iinfo(np.int64).min or value >= np.iinfo(np.int64).max for value in first + last):
        raise ValueError('volume exceeds voxel address space')
    count = math.prod(end - start + 1 for start, end in zip(first, last))
    if count > budget:
        return None, count
    x, y, z = [np.arange(start, end + 1, dtype=np.int64) for start, end in zip(first, last)]
    xx, yy = np.repeat(x, len(y)), np.tile(y, len(x))
    intersects = shapely.intersects(region['polygon'], shapely.box(
        xx * resolution, yy * resolution, (xx + 1) * resolution, (yy + 1) * resolution))
    xy = np.column_stack((xx[intersects], yy[intersects]))
    z = z[(z * resolution <= upper[2]) & ((z + 1) * resolution >= lower[2])]
    keys = np.column_stack((np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))))
    return keys, count


def _voxel_indices(table, keys):
    """Batch lookup in the table's lexicographic order; -1 means unobserved."""
    dtype = np.dtype([(axis, np.int64) for axis in ('x', 'y', 'z')])
    stored, requested = table.keys.view(dtype).reshape(-1), keys.view(dtype).reshape(-1)
    positions = np.searchsorted(stored, requested)
    found = positions < len(stored)
    found[found] = stored[positions[found]] == requested[found]
    return np.where(found, positions, -1)


def _finer_scale(scale, scales):
    """Fixed adjacent finer scale, independent of which observations are live."""
    return max((item for item in scales if item.resolution_m < scale.resolution_m),
               key=lambda item: item.resolution_m, default=None)


def _refined_hit_bounds(points, keys, scale, finer, fine_table, center, config, meta):
    """Certify full finer XYZ voxel bounds from this exact integrated scan.

    Every non-Ground return in a coarse cell must have been accepted as a hit
    by the finer table with this same Evidence object and full hazard deadline.
    The complete coarse cell must be inside that scan's finer workspace. The
    result never authorizes a fine FREE cell or makes a point-only collision.
    """
    if finer is None or fine_table is None or not len(keys):
        return None
    bounds = np.full((len(keys), 2, 3), np.nan)
    coarse_keys = np.floor(points / scale.resolution_m).astype(np.int64)
    grouped, _, inverse = _cell_groups(coarse_keys)
    if not np.array_equal(grouped, keys):
        raise ValueError('coarse hazard inputs do not match complete original returns')
    fine_keys = np.floor(points / finer.resolution_m).astype(np.int64)
    indices = _voxel_indices(fine_table, fine_keys)
    deadline = min(meta.valid_until_s, meta.capture_s + config.occupancy_ttl_s)
    accepted_sources = [serial for serial, evidence in fine_table.evidence.items() if evidence is meta]
    accepted = indices >= 0
    valid_indices = indices[accepted]
    accepted[accepted] = ((fine_table.scores[valid_indices] > 0)
        & (fine_table.expires[valid_indices] >= deadline)
        & (fine_table.hazard_expires[valid_indices] >= deadline)
        & np.isin(fine_table.hazard_sources[valid_indices], accepted_sources))
    complete = np.ones(len(keys), dtype=bool)
    np.logical_and.at(complete, inverse, accepted)
    lower, upper = keys * scale.resolution_m, (keys + 1) * scale.resolution_m
    workspace_lower = center + [-finer.radius_m, -finer.radius_m, config.z_min_m]
    workspace_upper = center + [finer.radius_m, finer.radius_m, config.z_max_m]
    complete &= np.all((lower >= workspace_lower) & (upper < workspace_upper), axis=1)
    fine_lower, fine_upper = fine_keys * finer.resolution_m, (fine_keys + 1) * finer.resolution_m
    accumulated_lower, accumulated_upper = np.full((len(keys), 3), np.inf), np.full((len(keys), 3), -np.inf)
    np.minimum.at(accumulated_lower, inverse, fine_lower)
    np.maximum.at(accumulated_upper, inverse, fine_upper)
    # Non-integer scale ratios may put part of a fine voxel outside its coarse
    # parent. Keep the intersection; other parents/fine hazards retain theirs.
    bounds[complete, 0] = np.maximum(lower[complete], accumulated_lower[complete])
    bounds[complete, 1] = np.minimum(upper[complete], accumulated_upper[complete])
    return bounds


def _hazard_boxes(scale, table, indices, scales, center, config, version, *, allow_refinement=True):
    """Conservative boxes for live hazards, falling back for missing coverage."""
    lower = table.keys[indices] * scale.resolution_m
    upper = (table.keys[indices] + 1) * scale.resolution_m
    finer = _finer_scale(scale, scales)
    refined = np.zeros(len(indices), dtype=bool)
    if not allow_refinement or finer is None or table.hazard_bounds is None:
        return lower, upper, refined
    envelopes = table.hazard_bounds[indices]
    refined = np.isfinite(envelopes).all(axis=(1, 2))
    workspace_lower = center + [-finer.radius_m, -finer.radius_m, config.z_min_m]
    workspace_upper = center + [finer.radius_m, finer.radius_m, config.z_max_m]
    refined &= np.all((lower >= workspace_lower) & (upper < workspace_upper), axis=1)
    valid_sources = [serial for serial, meta in table.evidence.items()
        if meta.origin == 'MEASUREMENT' and meta.support == 'DIRECT' and meta.metric_valid
        and meta.motion_epoch == version['motion_epoch'] and meta.calibration_version == version['calibration_version']]
    refined &= np.isin(table.hazard_sources[indices], valid_sources)
    lower[refined], upper[refined] = envelopes[refined, 0], envelopes[refined, 1]
    return lower, upper, refined


def ray_voxels(origin, endpoints, resolution):
    """Exact traversal in bounded batches; endpoint voxels are never free."""
    endpoints = np.asarray(endpoints, dtype=float)
    origin = np.asarray(origin,dtype=float)
    if (origin.shape!=(3,) or endpoints.ndim!=2 or endpoints.shape[1]!=3
            or not np.isfinite(origin).all() or not np.isfinite(endpoints).all()
            or not math.isfinite(resolution) or resolution<=0):
        raise ValueError('invalid ray traversal inputs')
    if traverse is not None and len(endpoints):
        targets=np.floor(endpoints/resolution).astype(np.int64)
        start=np.floor(origin/resolution).astype(np.int64)
        lower=np.minimum(targets.min(axis=0),start)
        extent=np.maximum(targets.max(axis=0),start)-lower+1
        # A bounded bitmap deduplicates traversed cells while tracing, instead
        # of allocating and sorting millions of repeated cell coordinates.
        if math.prod(map(int,extent)) <= 2_000_000:
            return traverse(origin,endpoints,resolution,lower,extent)
    parts = [_ray_voxels_chunk(origin, endpoints[start:start + 512], resolution)
             for start in range(0, len(endpoints), 512)]
    nonempty = [part for part in parts if len(part)]
    return _unique_cells(np.concatenate(nonempty)) if nonempty else np.empty((0, 3), dtype=np.int64)


def _ray_voxels_chunk(origin, endpoints, resolution):
    endpoints = np.asarray(endpoints, dtype=float)
    if not len(endpoints):
        return np.empty((0, 3), dtype=np.int64)
    start = np.floor(origin / resolution).astype(np.int64)
    cell = np.repeat(start[None, :], len(endpoints), axis=0)
    target = np.floor(endpoints / resolution).astype(np.int64)
    delta = endpoints - origin
    step = np.sign(delta).astype(np.int64)
    with np.errstate(divide='ignore', invalid='ignore'):
        stride = np.where(delta != 0, resolution / np.abs(delta), np.inf)
        boundary = (cell + (step > 0)) * resolution
        crossing = np.where(delta != 0, (boundary - origin) / delta, np.inf)
    chunks = []
    active = np.any(cell != target, axis=1)
    # A DDA crosses at most the Manhattan distance to the endpoint.
    for _ in range(int(np.abs(target - start).sum(axis=1).max()) + 1):
        if not active.any():
            break
        chunks.append(cell[active].copy())
        minimum = crossing.min(axis=1)
        before_endpoint = minimum < 1.-1e-12
        axes = (crossing <= minimum[:, None] + 1e-12) & active[:, None] & before_endpoint[:,None]
        cell += axes * step
        crossing = np.where(axes, crossing + stride, crossing)
        active &= before_endpoint & np.any(cell != target, axis=1)
    if not chunks:
        return np.empty((0, 3), dtype=np.int64)
    cells = np.concatenate(chunks)
    # Numeric scalar sorting avoids NumPy's much slower structured row sort.
    # Translation by the minimum keeps this independent of the odom origin.
    return _unique_cells(cells)


@dataclass
class RollingGrid:
    scale: Scale
    cells: VoxelTable = field(default_factory=VoxelTable)


class LocalWorld:
    LAYERS = ('objects', 'geometry', 'ground', 'surface_semantics', 'boundaries', 'hazards')

    def __init__(self, config=None, epoch=0, calibration='unknown'):
        self.config = config or WorldConfig()
        self.lock = RLock()
        # Only map writers serialize here. Fast measurements and consumers do
        # not wait for ray traversal, table joins, or map maintenance.
        self._geometry_writer = Lock()
        self.reset(epoch, calibration)

    def reset(self, epoch, calibration):
        if not isinstance(epoch, int) or epoch < 0 or not calibration:
            raise ValueError('explicit epoch and calibration required')
        with self.lock:
            self.epoch, self.calibration = epoch, calibration
            self.grids = [RollingGrid(s) for s in self.config.scales]
            self.layers = {k: {} for k in self.LAYERS if k not in ('geometry', 'ground')}
            self.seen = {}
            self.now, self.center = -math.inf, np.zeros(3)
            self.map_from_odom, self.map_version = np.eye(4), 'unversioned'
            self.last_sources = {}
            self.motion_quality, self.motion_covariance_valid = 'LOST', False
            self._generation = object()
            self.revision = self.fast_revision = self.geometry_revision = 0
            self.ground_observations = {}
            self._ground_indices = {}
            self._ground_revision = 0
            self._ground_cache = OrderedDict()
            self._query_stats = dict(ground_index_builds=0, ground_batches=0, ground_queries=0,
                ground_candidate_points=0, ground_cache_hits=0, ground_hull_builds=0,
                ground_plane_fits=0, ground_cpu_s=0., ground_wall_s=0.,
                surface_batches=0, surface_queries=0, surface_inverse_builds=0,
                surface_cpu_s=0., surface_wall_s=0.)
            self.current_scans = {}

    def set_motion_health(self, quality, covariance_valid=False):
        if quality not in ('OK', 'DEGRADED', 'LOST', 'INITIALIZING'):
            raise ValueError('invalid local motion quality')
        with self.lock:
            if (self.motion_quality, self.motion_covariance_valid) != (quality, bool(covariance_valid)):
                self.revision += 1
            self.motion_quality, self.motion_covariance_valid = quality, bool(covariance_valid)

    def set_map_correction(self, map_from_odom, version):
        matrix = rigid(map_from_odom)
        if not version:
            raise ValueError('map version required')
        with self.lock:
            if self.map_version != version or not np.array_equal(self.map_from_odom, matrix):
                self.revision += 1
            self.map_from_odom, self.map_version = matrix, version

    def advance(self, now, odom_from_base, *, maintain=True):
        """Advance validity and rolling center; async callers defer table trimming.

        ``integrate_scan`` trims the numeric map on its maintenance worker. The
        synchronous default keeps the archived replay/component API unchanged.
        """
        pose = rigid(odom_from_base)
        with self.lock:
            if not math.isfinite(now) or now < self.now:
                raise ValueError('clock reversal requires an explicit epoch reset')
            if now != self.now or not np.array_equal(self.center, pose[:3, 3]):
                self.revision += 1
            self.now, self.center = now, pose[:3, 3].copy()
            generation, center = self._generation, self.center.copy()
            tables = [(grid, grid.cells) for grid in self.grids]
            live_ground = {key: value for key, value in self.ground_observations.items()
                if min(value[0].valid_until_s, value[0].capture_s + self.config.ground_ttl_s) >= now}
            if len(live_ground) != len(self.ground_observations):
                self._ground_revision += 1
                self._ground_cache.clear()
                self._ground_indices = {key: self._ground_indices[key] for key in live_ground}
            self.ground_observations = live_ground
            self.current_scans = {key: value for key, value in self.current_scans.items()
                if min(value[0].valid_until_s, value[0].capture_s + self.config.occupancy_ttl_s) >= now}
            for layer in self.layers.values():
                for key in list(layer):
                    if layer[key][0].valid_until_s < now:
                        del layer[key]
            self.seen = {k: v for k, v in self.seen.items() if v >= now}
        if not maintain:
            return
        # Numeric tables are copy-on-write. A reader keeps a valid old version
        # while a writer trims or replaces it; no arrays are edited in place.
        trimmed = []
        for grid, original in tables:
            table = copy.copy(original)
            table.trim(now, center, grid.scale.resolution_m, grid.scale.radius_m)
            trimmed.append((grid, original, table))
        with self.lock:
            if generation is self._generation:
                for grid, original, table in trimmed:
                    if grid.cells is original:
                        grid.cells = table

    def _accept(self, layer, meta):
        if meta.motion_epoch != self.epoch or meta.calibration_version != self.calibration:
            raise ValueError('evidence epoch or calibration mismatch')
        if meta.state_s > self.now + 1e-8 or meta.valid_until_s < self.now:
            return False
        key = (layer, meta.correlation_group)
        if key in self.seen:
            return False
        self.seen[key] = meta.valid_until_s
        self.last_sources[layer] = max(meta.last_direct_s, self.last_sources.get(layer, -math.inf))
        return True

    def _measured_scan(self, endpoints_odom, origin_odom, ground_mask, meta):
        points, origin = np.asarray(endpoints_odom, dtype=float), np.asarray(origin_odom, dtype=float)
        ground = np.asarray(ground_mask, dtype=bool)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or origin.shape != (3,) or not np.isfinite(origin).all() or ground.shape != (len(points),)):
            raise ValueError('invalid measured rays')
        if meta.origin != 'MEASUREMENT' or not meta.metric_valid or meta.support != 'DIRECT':
            raise ValueError('only direct metric measurements may update ray geometry')
        return points, origin, ground

    def observe_current(self, endpoints_odom, origin_odom, ground_mask, meta):
        """Publish measured Ground and current obstacles without traversing rays.

        Raw supports are independent of voxel size. Each scan retains its own
        optical origin and provenance; empty/non-ground scans never supply a
        ground plane. The caller may enqueue integrate_scan on a slow worker.
        """
        points, origin, ground = self._measured_scan(endpoints_odom, origin_odom, ground_mask, meta)
        points, origin, ground = points.copy(), origin.copy(), ground.copy()
        for array in (points, origin, ground):
            array.setflags(write=False)
        measured_ground = points[ground]
        measured_ground.setflags(write=False)
        ground_index = (GroundIndex(measured_ground, self.config.ground_cell_size_m)
                        if len(measured_ground) else None)
        with self.lock:
            if ground_index is not None:
                self._query_stats['ground_index_builds'] += 1
            if meta.capture_s + max(self.config.occupancy_ttl_s, self.config.ground_ttl_s) < self.now:
                return False
            if not self._accept('current_geometry', meta):
                return False
            if ground.any():
                self.ground_observations[meta.observation_id] = (meta, measured_ground)
                self._ground_indices[meta.observation_id] = ground_index
                self._ground_revision += 1
                self._ground_cache.clear()
                self.last_sources['ground'] = max(meta.capture_s, self.last_sources.get('ground', -math.inf))
            sensor = meta.sensor_id or 'lidar'
            old = self.current_scans.get(sensor)
            if old is None or old[0].capture_s <= meta.capture_s:
                self.current_scans[sensor] = (meta, points, origin, ground)
            self.fast_revision += 1
            self.revision += 1
        return True

    def observe_scan(self, endpoints_odom, origin_odom, ground_mask, meta):
        """Synchronous adapter used by component tests and archived replay."""
        if not self.observe_current(endpoints_odom, origin_odom, ground_mask, meta):
            return False
        return self.integrate_scan(endpoints_odom, origin_odom, ground_mask, meta)

    def integrate_scan(self, endpoints_odom, origin_odom, ground_mask, meta):
        """Compute ray evidence outside the state lock, then commit one version."""
        points, origin, ground = self._measured_scan(endpoints_odom, origin_odom, ground_mask, meta)
        with self._geometry_writer:
            with self.lock:
                if meta.motion_epoch != self.epoch or meta.calibration_version != self.calibration:
                    raise ValueError('evidence epoch or calibration mismatch')
                if (meta.state_s > self.now + 1e-8 or meta.valid_until_s < self.now
                        or meta.capture_s + self.config.occupancy_ttl_s < self.now
                        or ('geometry', meta.correlation_group) in self.seen):
                    return False
                generation, now, center = self._generation, self.now, self.center.copy()
                grids = [(grid, copy.copy(grid.cells)) for grid in self.grids]
            integrated = {}
            for grid, table in sorted(grids, key=lambda item: item[0].scale.resolution_m):
                res, radius = grid.scale.resolution_m, grid.scale.radius_m
                table.trim(now, center, res, radius)
                lower = center + [-radius, -radius, self.config.z_min_m]
                upper = center + [radius, radius, self.config.z_max_m]
                if not np.all((origin >= lower) & (origin < upper)):
                    continue
                delta = points - origin
                with np.errstate(divide='ignore', invalid='ignore'):
                    exits = np.where(delta > 0, (upper - origin) / delta,
                                     np.where(delta < 0, (lower - origin) / delta, np.inf))
                fractions = np.minimum(1., exits.min(axis=1))
                clipped = origin + delta * fractions[:, None]
                # Trace one actual ray per 3-D terminal voxel. This can leave
                # additional unknown volume, but cannot invent free evidence.
                _, representatives, _ = _cell_groups(np.floor(clipped / res).astype(np.int64))
                free = ray_voxels(origin, clipped[representatives], res)
                inside = np.all((points >= lower) & (points < upper), axis=1)
                hits = _unique_cells(np.floor(points[inside] / res).astype(np.int64))
                obstacle_points = points[inside & ~ground]
                obstacles = _unique_cells(np.floor(obstacle_points / res).astype(np.int64))
                finer = _finer_scale(grid.scale, self.config.scales)
                bounds = _refined_hit_bounds(obstacle_points, obstacles, grid.scale, finer,
                    integrated.get(finer.name) if finer else None, center, self.config, meta)
                table.update(free, hits, meta, now, self.config.free_ttl_s,
                             self.config.occupancy_ttl_s, obstacle_hits=obstacles,
                             refined_hazard_bounds=bounds)
                integrated[grid.scale.name] = table
            with self.lock:
                if (generation is not self._generation or meta.valid_until_s < self.now
                        or meta.capture_s + self.config.occupancy_ttl_s < self.now):
                    return False
                if not self._accept('geometry', meta):
                    return False
                for grid, table in grids:
                    grid.cells = table
                self.geometry_revision += 1
                self.revision += 1
        return True

    def put(self, layer, identity, payload, meta):
        if layer not in self.layers or not identity:
            raise ValueError('expected a typed vector/object layer and identity')
        if layer == 'objects':
            center, dimensions = np.asarray(payload.get('center_xyz'), dtype=float), np.asarray(payload.get('dimensions'), dtype=float)
            if (center.shape != (3,) or dimensions.shape != (3,) or not np.isfinite(center).all()
                    or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
                raise ValueError('object requires finite center and positive metric dimensions')
            if 'velocity_xyz' in payload:
                velocity = np.asarray(payload['velocity_xyz'], dtype=float)
                if velocity.shape != (3,) or not np.isfinite(velocity).all():
                    raise ValueError('object velocity must be finite odom XYZ')
        if layer == 'boundaries':
            vertices = np.asarray(payload.get('polyline_xyz'), dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2 or not np.isfinite(vertices).all():
                raise ValueError('boundary requires a finite metric polyline')
        with self.lock:
            old = self.layers[layer].get(identity)
            if old and old[0].state_s > meta.state_s:
                return False
            # A group may contain multiple objects; deduplicate by identity too.
            if not self._accept(layer + ':' + identity, meta):
                return False
            import copy
            self.layers[layer][identity] = (meta, copy.deepcopy(payload))
            self.last_sources[layer] = max(meta.last_direct_s, self.last_sources.get(layer, -math.inf))
            self.revision += 1
            return True

    def voxel_state(self, xyz, scale='near'):
        point = np.asarray(xyz, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('invalid query point')
        with self.lock:
            grid = next(g for g in self.grids if g.scale.name == scale)
            if (np.max(np.abs(point[:2] - self.center[:2])) > grid.scale.radius_m
                    or not self.center[2] + self.config.z_min_m <= point[2] < self.center[2] + self.config.z_max_m):
                return 'UNKNOWN'
            return grid.cells.state(np.floor(point / grid.scale.resolution_m).astype(np.int64), self.now)

    def ground_support(self, xy, radius_m=.35, *, include_points=False):
        """Plane query only inside the convex hull of actual nearby returns."""
        query = np.asarray(xy, dtype=float)
        if query.shape != (2,):
            raise ValueError('invalid ground query')
        return self.ground_support_many(query[None], radius_m, include_points=include_points)[0]

    def ground_support_many(self, xy, radius_m=.35, *, include_points=False):
        """One version snapshot, exact radius supports, bounded shared fit cache.

        Scalar radius/include_points apply to every row; vectors preserve the
        existing query transport's per-row options. Indexing never subsamples.
        """
        begin, cpu_begin = time.perf_counter(), time.thread_time()
        queries = np.asarray(xy, dtype=float)
        if queries.ndim != 2 or queries.shape[1] != 2 or not np.isfinite(queries).all():
            raise ValueError('ground queries require finite Nx2 coordinates')
        radii = np.broadcast_to(np.asarray(radius_m, dtype=float), (len(queries),))
        includes = np.broadcast_to(np.asarray(include_points, dtype=bool), (len(queries),))
        if not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError('ground query radii must be finite and positive')
        with self.lock:
            now, generation, version = self.now, self._generation, self._version_locked()
            ground_revision, center = self._ground_revision, self.center[:2].copy()
            workspace = max(s.radius_m for s in self.config.scales)
            observations = [(meta, self._ground_indices[identity],
                             min(meta.valid_until_s, meta.capture_s+self.config.ground_ttl_s))
                            for identity, (meta, _) in self.ground_observations.items()
                            if min(meta.valid_until_s, meta.capture_s+self.config.ground_ttl_s) >= now]
            cache = dict(self._ground_cache)
        results, touched = [], OrderedDict()
        candidates = hits = hulls = fits = 0
        for query, radius, include in zip(queries, radii, includes):
            result = {'supported': False, 'reason': 'NO_RETURNS', 'version': version}
            results.append(result)
            if np.max(np.abs(query-center)) > workspace:
                result['reason'] = 'OUTSIDE_WORKSPACE'
                continue
            selected, sources, deadlines, identity_parts = [], [], [], []
            for meta, index, deadline in observations:
                indices, examined = index.nearby(query, radius)
                candidates += examined
                if len(indices):
                    selected.append((index, indices))
                    sources.append(meta)
                    deadlines.append(deadline)
                    identity_parts.append((meta.observation_id, indices.tobytes()))
            if not selected:
                continue
            key = (float(radius), tuple(identity_parts))
            fit = cache.get(key)
            if fit is None:
                points = np.concatenate([index.points[indices] for index, indices in selected])
                fit = GroundFit.build(points)
                hulls += int(len(points) >= 3)
                fits += int(fit.coefficients is not None)
                cache[key] = fit
            else:
                hits += 1
            touched[key] = fit
            result.update(return_count=len(fit.points), covariance_valid=False,
                observation_ids=[meta.observation_id for meta in sources],
                source_age_s=max(now-meta.capture_s for meta in sources), valid_until_s=min(deadlines))
            result.update(fit.at(query, bool(include)))
        with self.lock:
            changed = ground_revision != self._ground_revision
            for result in results:
                if generation is not self._generation or self.now > result.get('valid_until_s', math.inf):
                    result.update(supported=False, reason='EXPIRED_DURING_QUERY')
                elif changed:
                    result.update(supported=False, reason='GROUND_CHANGED_DURING_QUERY')
            if generation is self._generation:
                if not changed:
                    for key, fit in touched.items():
                        self._ground_cache[key] = fit
                        self._ground_cache.move_to_end(key)
                    while len(self._ground_cache) > self.config.ground_cache_entries:
                        self._ground_cache.popitem(last=False)
                counts = dict(ground_batches=1, ground_queries=len(queries), ground_candidate_points=candidates,
                    ground_cache_hits=hits, ground_hull_builds=hulls, ground_plane_fits=fits,
                    ground_cpu_s=time.thread_time()-cpu_begin, ground_wall_s=time.perf_counter()-begin)
                for name, amount in counts.items():
                    self._query_stats[name] += amount
        return results

    def query_stats(self):
        with self.lock:
            return {**self._query_stats, 'ground_cached_fits': len(self._ground_cache),
                    'ground_index_count': len(self._ground_indices)}

    def boundary_permission(self, identity, *, contact_interval_m=None, nominal_contact_m=None,
                            response_time_s=None, time_to_contact_s=None, safety_valid_until_s=None):
        """Explicit work budget; time-to-contact is measured from last direct input.

        A complete supplied joint bound replaces the former fixed sigma gate.
        The caller supplies the collision/safety horizon for the actual sweep.
        This permits nominal geometry only, never physical cleaning credit.
        """
        from dg202604.qloop.curb_execution import nominal_boundary_permission

        with self.lock:
            entry = self.layers['boundaries'].get(identity)
            if entry is None:
                return False
            meta, data = entry
            return nominal_boundary_permission(
                now_s=self.now, last_direct_s=meta.last_direct_s, valid_until_s=meta.valid_until_s,
                metric_valid=meta.metric_valid and self.motion_quality == 'OK', support=meta.support,
                total_error_bound_m=data.get('total_error_bound_m'),
                error_bound_complete=data.get('error_bound_complete'), ground_valid=data.get('ground_valid'),
                boundary_type=data.get('boundary_type', 'UNKNOWN'),
                motion_bound_valid=data.get('motion_bound_valid'), contact_interval_m=contact_interval_m,
                nominal_contact_m=nominal_contact_m, response_time_s=response_time_s,
                time_to_contact_s=time_to_contact_s, safety_valid_until_s=safety_valid_until_s)

    def observe_surface_grid(self, labels, odom_from_model, meta, *, scores=None, model_version='unspecified'):
        """Store HEAL's native class tensor without inferring free/ground support."""
        labels, scores = validate_grid(labels, scores)
        pose = rigid(odom_from_model)
        if meta.origin != 'MODEL' or not isinstance(model_version, str) or not model_version:
            raise ValueError('expected native HEAL semantic grid and MODEL provenance')
        # One owning copy of each compact array. Generic put() deep-copies other
        # payloads, but must not copy this already-owned semantic volume again.
        owned = np.array(labels, dtype=np.int8, order='C', copy=True)
        owned.setflags(write=False)
        confidence = None if scores is None else scores.copy(order='C')
        if confidence is not None:
            confidence.setflags(write=False)
        inverse = np.linalg.inv(pose)
        for array in (pose, inverse):
            array.setflags(write=False)
        grid = dict(labels=owned, scores=confidence, odom_from_model=pose,
                    model_from_odom=inverse, model_version=model_version)
        with self.lock:
            old = self.layers['surface_semantics'].get('heal')
            if old and old[0].state_s > meta.state_s:
                return False
            if not self._accept('surface_semantics:heal', meta):
                return False
            self.layers['surface_semantics']['heal'] = (meta, grid)
            self.last_sources['surface_semantics'] = max(meta.last_direct_s,
                self.last_sources.get('surface_semantics', -math.inf))
            self._query_stats['surface_inverse_builds'] += 1
            self.revision += 1
        return True

    def surface_at(self, xyz_odom):
        point = np.asarray(xyz_odom, dtype=float)
        if point.shape != (3,):
            raise ValueError('surface query requires an XYZ point')
        return self.surface_at_many(point[None])[0]

    def surface_at_many(self, xyz_odom, *, layer_offsets=(0,)):
        """Query one semantic version, with optional native Z-layer neighbors.

        Rows are returned point-major, then in ``layer_offsets`` order. Offsets
        change the model grid's Z index, not odom Z or neighboring XY cells.
        """
        begin, cpu_begin = time.perf_counter(), time.thread_time()
        points = np.asarray(xyz_odom, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('surface queries require finite Nx3 points')
        offsets = np.asarray(layer_offsets)
        if offsets.ndim != 1 or not len(offsets) or offsets.dtype.kind not in 'iu' or np.any(np.abs(offsets) > 1):
            raise ValueError('surface layer offsets must be integer neighbors -1, 0 or 1')
        with self.lock:
            entry = self.layers['surface_semantics'].get('heal')
            now, generation, version = self.now, self._generation, self._version_locked()
        results = [dict(label=None, reason='UNAVAILABLE', version=version,
                        free_space_permission=False, ground_support_permission=False)
                   for _ in range(len(points)*len(offsets))]
        if entry is not None and entry[0].valid_until_s >= now:
            meta, grid = entry
            model = transform(points, grid['model_from_odom'])
            coordinates = np.floor((model-GRID_ORIGIN_M)/GRID_RESOLUTION_M)
            coordinates = np.repeat(coordinates, len(offsets), axis=0)
            coordinates[:, 2] += np.tile(offsets, len(points))
            inside = np.all((coordinates >= 0) & (coordinates < [200, 200, 16]), axis=1)
            selected = np.flatnonzero(inside)
            cells = coordinates[inside].astype(np.int64)
            indices = (0, cells[:, 0], cells[:, 1], cells[:, 2])
            labels = grid['labels'][indices]
            scores = None if grid['scores'] is None else grid['scores'][indices]
            for result in results:
                result.update(reason='OUTSIDE_GRID', observation_id=meta.observation_id,
                    source_age_s=now-meta.capture_s, valid_until_s=meta.valid_until_s,
                    model_version=grid['model_version'])
            for i, position in enumerate(selected):
                label = int(labels[i])
                score = None if scores is None else int(scores[i])
                results[position].update(label=None if label in (0, 17) else label,
                    native_label=label, class_name=CLASS_NAMES[label], reason='MODEL_ONLY',
                    score_q=score, score=None if score is None else score*SCORE_SCALE)
        with self.lock:
            if generation is not self._generation:
                for result in results:
                    result.update(label=None, native_label=None, score=None, score_q=None, reason='EXPIRED_DURING_QUERY')
            elif entry is not None:
                expired = entry[0].valid_until_s < self.now
                replaced = self.layers['surface_semantics'].get('heal') is not entry
                if expired or replaced:
                    for result in results:
                        result.update(label=None, native_label=None, score=None, score_q=None,
                            reason='EXPIRED_DURING_QUERY' if expired else 'REPLACED_DURING_QUERY')
            if generation is self._generation:
                self._query_stats['surface_batches'] += 1
                self._query_stats['surface_queries'] += len(results)
                self._query_stats['surface_cpu_s'] += time.thread_time()-cpu_begin
                self._query_stats['surface_wall_s'] += time.perf_counter()-begin
        return results

    def ground_surface_many(self, xy_odom, radius_m=.35, *, minimum_score=.5):
        """Work-area semantics at measured Ground height, at most three Z reads.

        The centre sample decides the category; adjacent layers expose height
        ambiguity. Missing Ground does not get a guessed height from Occ.
        """
        if not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
            raise ValueError('semantic score threshold must be between zero and one')
        points = np.asarray(xy_odom, dtype=float)
        ground = self.ground_support_many(points, radius_m)
        results = [dict(xy_odom=xy.tolist(), ground=item, ground_supported=item['supported'],
            label=None, surface_class='unknown', reason='UNKNOWN_HEIGHT', neighborhood=[],
            version=item['version'], fine_work_permission=False, drive_permission=False)
            for xy, item in zip(points, ground)]
        selected = [i for i, item in enumerate(ground) if item['supported']]
        if not selected:
            return results
        xyz = np.array([[*points[i], ground[i]['z_m']] for i in selected])
        semantic = self.surface_at_many(xyz, layer_offsets=(-1, 0, 1))
        for j, i in enumerate(selected):
            band = semantic[j*3:j*3+3]
            centre = band[1]
            result = results[i]
            result.update(neighborhood=band, reason=centre['reason'])
            if any(item['version'] != result['version'] for item in band):
                result.update(ground_supported=False, reason='QUERY_VERSION_CHANGED')
                continue
            label, score = centre['label'], centre.get('score')
            if score is None:
                result['reason'] = 'SCORE_UNAVAILABLE'
            elif score < minimum_score:
                result['reason'] = 'LOW_SCORE'
            elif label is not None:
                nearby = {item['label'] for item in band if item['label'] in (11, 12, 13, 14)
                    and item.get('score') is not None and item['score'] >= minimum_score}
                if len(nearby) > 1:
                    result['reason'] = 'HEIGHT_SEMANTIC_CONFLICT'
                else:
                    result.update(label=label, surface_class={11: 'road', 13: 'sidewalk', 14: 'terrain'}.get(label, 'other'),
                        reason='MODEL_ON_MEASURED_GROUND', score=score, score_q=centre['score_q'],
                        model_version=centre['model_version'], observation_id=centre['observation_id'],
                        valid_until_s=min(ground[i]['valid_until_s'], centre['valid_until_s']))
        with self.lock:
            version = self._version_locked()
            for result in results:
                if result['version'] != version or (result.get('valid_until_s', self.now) < self.now):
                    result.update(label=None, surface_class='unknown', ground_supported=False, reason='QUERY_VERSION_CHANGED')
        return results

    def volume_clearance(self, regions, *, scale, required_until_s,
                         expected_version=None, max_voxels=200000):
        """Check complete caller-supplied continuous swept prisms in odom.

        Every intersecting closed voxel at the requested scale must carry
        DIRECT metric ray FREE evidence through required_until_s + offset_s.
        All current non-Ground returns and all scales' live hazards take
        precedence over missing evidence. Ground and self geometry never fill
        unknown cells. max_voxels also caps bounding-box candidate enumeration.
        This geometric certificate grants no operation or driving permission.
        """
        import shapely

        parsed = _volume_regions(regions)
        if not isinstance(scale, str) or not scale:
            raise ValueError('volume clearance requires an explicit scale')
        if isinstance(max_voxels, (bool, np.bool_)) or not isinstance(max_voxels, (int, np.integer)) or max_voxels <= 0:
            raise ValueError('max_voxels must be a positive integer')
        try:
            required_until_s = float(required_until_s)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('required_until_s must be a finite absolute time') from error
        if (not math.isfinite(required_until_s)
                or any(not math.isfinite(required_until_s + region['offset_s']) for region in parsed)):
            raise ValueError('volume validity horizon must be finite')
        if expected_version is not None and not isinstance(expected_version, dict):
            raise ValueError('expected_version must be a complete version dictionary')
        with self.lock:
            now, center, version = self.now, self.center.copy(), self._version_locked()
            generation, config = self._generation, self.config
            grids = tuple((grid.scale, grid.cells) for grid in self.grids)
            scans = tuple(self.current_scans.values())
        requested = next(((item, table) for item, table in grids if item.name == scale), None)
        result = {'state': 'UNKNOWN', 'reason': None, 'version': version, 'frame_id': 'odom',
            'scale': scale, 'resolution_m': requested[0].resolution_m if requested else None,
            'stamp_s': now if math.isfinite(now) else None, 'required_until_s': required_until_s,
            'valid_until_s': None, 'observation_ids': [], 'sensor_ids': [],
            'source_time_range_s': None, 'source_age_s': None,
            'first_unknown': None, 'first_collision': None, 'checked_voxels': 0,
            'candidate_voxels': 0, 'max_voxels': int(max_voxels),
            'evidence_domain': 'DIRECT_METRIC_RAY_VOXELS',
            'geometry_only': True, 'operation_valid': False, 'drive_permission': False}
        sources = {}

        def add_sources(metas):
            changed = False
            for meta in metas:
                if meta is not None:
                    key = (meta.observation_id, meta.sensor_id, meta.capture_s)
                    if key not in sources:
                        sources[key], changed = meta, True
            if changed:
                captures = [meta.capture_s for meta in sources.values()]
                result.update(observation_ids=sorted({meta.observation_id for meta in sources.values()}),
                    sensor_ids=sorted({meta.sensor_id or 'lidar' for meta in sources.values()}),
                    source_time_range_s=[min(captures), max(captures)], source_age_s=now - min(captures))

        def location(region, key=None, **details):
            return {'part_id': region['part_id'], 'region_index': region['region_index'],
                    'voxel_key': None if key is None else [int(value) for value in key],
                    'offset_s': region['offset_s'], **details}

        def unknown(reason, region, key=None, **details):
            result.update(state='UNKNOWN', reason=reason, valid_until_s=None,
                first_unknown=location(region, key, reason=reason, **details))
            return self._query_result(result, generation)

        if expected_version is not None and expected_version != version:
            return unknown('VERSION_MISMATCH', parsed[0])
        if not math.isfinite(now):
            return unknown('CLOCK_UNINITIALIZED', parsed[0])

        # Check every prism for hazards before returning any UNKNOWN from a
        # different prism, missing scale, workspace bound, or enumeration cap.
        for meta, measured, _, ground in scans:
            deadline = min(meta.valid_until_s, meta.capture_s + config.occupancy_ttl_s)
            if deadline < now:
                continue
            points = measured[~ground]
            for region in parsed:
                xmin, ymin, xmax, ymax = region['bounds']
                candidates = points[(points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                    & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
                    & (points[:, 2] >= region['z_min_m']) & (points[:, 2] <= region['z_max_m'])]
                inside = shapely.covers(region['polygon'], shapely.points(candidates[:, :2]))
                if inside.any():
                    add_sources([meta])
                    result.update(state='OCCUPIED', reason='CURRENT_NON_GROUND_RETURN',
                        valid_until_s=deadline - region['offset_s'],
                        first_collision=location(region, layer='current_geometry',
                            point_odom=candidates[np.flatnonzero(inside)[0]].tolist(),
                            observation_id=meta.observation_id, sensor_id=meta.sensor_id or 'lidar',
                            expires_s=deadline))
                    return self._query_result(result, generation)
        for hazard_scale, table in grids:
            indices = np.flatnonzero((table.scores > 0) & (table.expires >= now) & (table.hazard_expires >= now))
            lower, upper, _ = _hazard_boxes(hazard_scale, table, indices, config.scales,
                center, config, version, allow_refinement=(requested is not None
                    and requested[0].resolution_m < hazard_scale.resolution_m))
            for region in parsed:
                xmin, ymin, xmax, ymax = region['bounds']
                selected = np.flatnonzero((lower[:, 0] <= xmax) & (upper[:, 0] >= xmin)
                    & (lower[:, 1] <= ymax) & (upper[:, 1] >= ymin)
                    & (lower[:, 2] <= region['z_max_m']) & (upper[:, 2] >= region['z_min_m']))
                inside = shapely.intersects(region['polygon'], shapely.box(
                    lower[selected, 0], lower[selected, 1], upper[selected, 0], upper[selected, 1]))
                if inside.any():
                    index = int(indices[selected[np.flatnonzero(inside)[0]]])
                    meta = table.evidence.get(int(table.hazard_sources[index]))
                    deadline = float(min(table.expires[index], table.hazard_expires[index]))
                    add_sources([meta])
                    result.update(state='OCCUPIED', reason='UNEXPIRED_GEOMETRY_HAZARD',
                        valid_until_s=deadline - region['offset_s'],
                        first_collision=location(region, table.keys[index], layer='geometry',
                            scale=hazard_scale.name, resolution_m=hazard_scale.resolution_m,
                            observation_id=meta.observation_id if meta else None,
                            sensor_id=(meta.sensor_id or 'lidar') if meta else None, expires_s=deadline))
                    return self._query_result(result, generation)

        if requested is None:
            return unknown('SCALE_UNAVAILABLE', parsed[0])
        requested_scale, table = requested
        valid_until = math.inf
        for region in parsed:
            xmin, ymin, xmax, ymax = region['bounds']
            radius = requested_scale.radius_m
            if (xmin < center[0] - radius or xmax > center[0] + radius
                    or ymin < center[1] - radius or ymax > center[1] + radius
                    or region['z_min_m'] < center[2] + config.z_min_m
                    or region['z_max_m'] >= center[2] + config.z_max_m):
                return unknown('OUTSIDE_WORKSPACE', region)
            keys, candidates = _volume_voxels(region, requested_scale.resolution_m,
                                               max_voxels - result['candidate_voxels'])
            if keys is None:
                return unknown('VOXEL_BUDGET_EXCEEDED', region, required_candidate_voxels=candidates)
            result['candidate_voxels'] += candidates
            if not len(keys):
                return unknown('NO_INTERSECTING_VOXELS', region)
            indices = _voxel_indices(table, keys)
            needed_until = max(now, required_until_s + region['offset_s'])
            # Rows are bounded by max_voxels. Preserve the first failed cell in
            # deterministic lexicographic order, including missing height rows.
            for key, index in zip(keys, indices):
                result['checked_voxels'] += 1
                if index < 0:
                    return unknown('NO_VOXEL_EVIDENCE', region, key)
                deadline, score = float(table.expires[index]), int(table.scores[index])
                meta = table.evidence.get(int(table.sources[index]))
                add_sources([meta])
                if deadline < now:
                    return unknown('EXPIRED_EVIDENCE', region, key, expires_s=deadline)
                if score == 0:
                    return unknown('NEUTRAL_EVIDENCE', region, key, expires_s=deadline)
                if score > 0:
                    return unknown('NOT_FREE', region, key, expires_s=deadline)
                if (meta is None or meta.origin != 'MEASUREMENT' or meta.support != 'DIRECT'
                        or not meta.metric_valid or meta.motion_epoch != version['motion_epoch']
                        or meta.calibration_version != version['calibration_version']):
                    return unknown('INVALID_PROVENANCE', region, key)
                if deadline < needed_until:
                    return unknown('INSUFFICIENT_VALIDITY', region, key,
                        expires_s=deadline, needed_until_s=needed_until)
                valid_until = min(valid_until, deadline - region['offset_s'])
        result.update(state='FREE', reason='ALL_INTERSECTING_VOXELS_MEASURED_FREE', valid_until_s=valid_until)
        return self._query_result(result, generation)

    def swept_collision(self, poses, front_m=1.5941, rear_m=1.0, half_width_m=1.0915,
                        z_min_m=0., z_max_m=2.2, offsets_s=None):
        """Conservative collision at supplied dense trajectory samples.

        Returns UNKNOWN when no hit exists: sparse beams cannot certify a whole
        swept volume as free. Trajectory sampling/vehicle dynamics belong to the
        planner; this query never grants driving permission by itself.
        """
        poses = [rigid(p) for p in poses]
        offsets = np.zeros(len(poses)) if offsets_s is None else np.asarray(offsets_s, dtype=float)
        if (not poses or not np.isfinite([front_m,rear_m,half_width_m,z_min_m,z_max_m]).all()
                or min(front_m, rear_m, half_width_m) <= 0 or z_min_m >= z_max_m):
            raise ValueError('invalid swept vehicle volume')
        if offsets.shape != (len(poses),) or not np.isfinite(offsets).all() or np.any(offsets < 0) or np.any(np.diff(offsets) < 0):
            raise ValueError('trajectory times must be finite nondecreasing offsets')
        inverse_poses = [np.linalg.inv(pose) for pose in poses]
        with self.lock:
            now, generation, version = self.now, self._generation, self._version_locked()
            center, config = self.center.copy(), self.config
            grids = [(grid.scale, grid.cells) for grid in self.grids]
            scans = tuple(self.current_scans.values())
            objects = tuple(self.layers['objects'].items())
            hazard_count = sum(meta.valid_until_s >= now for meta, _ in self.layers['hazards'].values())
        # A current low return must constrain the consumer even while the slow
        # ray worker has not committed a map. Ground returns are kept separately.
        for meta, measured, _, ground in scans:
            if min(meta.valid_until_s, meta.capture_s + config.occupancy_ttl_s) < now:
                continue
            obstacles = measured[~ground]
            for index, inverse in enumerate(inverse_poses):
                p = obstacles @ inverse[:3, :3].T + inverse[:3, 3]
                inside = ((p[:, 0] >= -rear_m) & (p[:, 0] <= front_m)
                          & (np.abs(p[:, 1]) <= half_width_m)
                          & (p[:, 2] >= z_min_m) & (p[:, 2] <= z_max_m))
                if inside.any():
                    return self._query_result({'state': 'OCCUPIED', 'layer': 'current_geometry', 'sample': index,
                            'observation_id': meta.observation_id, 'sensor_id': meta.sensor_id,
                            'point_odom': obstacles[np.flatnonzero(inside)[0]].tolist(), 'version': version}, generation)
        for scale, table in grids:
            indices = np.flatnonzero((table.scores > 0) & (table.expires >= now) & (table.hazard_expires >= now))
            if not len(indices):
                continue
            lower, upper, _ = _hazard_boxes(scale, table, indices, config.scales, center,
                                            config, version)
            centers, half_extents = (lower + upper) / 2, (upper - lower) / 2
            # Only complete same-observation support envelopes can refine a
            # coarse hazard. Missing coverage retains its entire coarse box.
            for index, pose in enumerate(poses):
                inverse = inverse_poses[index]
                # Poses were validated once above and centers belong to the
                # numeric table; avoid repeated validation of this same data.
                p = centers @ inverse[:3, :3].T + inverse[:3, 3]
                padding = half_extents @ np.abs(pose[:3, :3])
                inside = ((p[:, 0] >= -rear_m - padding[:, 0]) & (p[:, 0] <= front_m + padding[:, 0])
                          & (np.abs(p[:, 1]) <= half_width_m + padding[:, 1])
                          & (p[:, 2] > z_min_m - padding[:, 2] + 1e-9) & (p[:, 2] < z_max_m + padding[:, 2]))
                if inside.any():
                    return self._query_result({'state': 'OCCUPIED', 'layer': 'geometry', 'scale': scale.name, 'sample': index,
                            'voxel_center_odom': ((table.keys[indices[np.flatnonzero(inside)[0]]] + .5)
                                * scale.resolution_m).tolist(), 'version': version}, generation)
        unresolved_prediction = None
        for identity, (meta, obj) in objects:
            if meta.valid_until_s < now or not meta.metric_valid:
                continue
            center = np.asarray(obj['center_xyz'])
            dimensions = np.asarray(obj['dimensions'])
            # Circumscribed object sphere is deliberately conservative.
            radius = float(np.linalg.norm(dimensions) / 2)
            for index, pose in enumerate(poses):
                velocity = np.asarray(obj.get('velocity_xyz', [0., 0., 0.]))
                age = now + offsets[index] - meta.state_s
                if age > obj.get('prediction_horizon_s', 0.) and np.any(velocity):
                    unresolved_prediction = identity
                    continue
                predicted = center + velocity * age
                inverse = inverse_poses[index]
                p = inverse[:3, :3] @ predicted + inverse[:3, 3]
                if (-rear_m - radius <= p[0] <= front_m + radius
                        and abs(p[1]) <= half_width_m + radius and z_min_m - radius <= p[2] <= z_max_m + radius):
                    return self._query_result({'state': 'OCCUPIED', 'layer': 'objects', 'identity': identity, 'sample': index,
                            'offset_s': float(offsets[index]), 'version': version}, generation)
        if hazard_count:
            return self._query_result({'state': 'UNRESOLVED_HAZARD', 'count': hazard_count, 'version': version}, generation)
        if unresolved_prediction is not None:
            return self._query_result({'state': 'UNKNOWN', 'reason': 'PREDICTION_HORIZON', 'identity': unresolved_prediction, 'version': version}, generation)
        return self._query_result({'state': 'UNKNOWN', 'version': version}, generation)

    def _query_result(self, result, generation):
        """A lock-free computation is usable only for the captured version."""
        with self.lock:
            current = self._version_locked()
            if generation is not self._generation or current != result['version']:
                result.update(state='UNKNOWN', reason='VERSION_MISMATCH', current_version=current)
                if 'valid_until_s' in result:
                    result.update(valid_until_s=None, first_collision=None,
                        first_unknown={'part_id': None, 'region_index': None, 'voxel_key': None,
                            'offset_s': None, 'reason': 'VERSION_MISMATCH'})
        return result

    def _version_locked(self):
        return {'motion_epoch': self.epoch, 'calibration_version': self.calibration,
                'map_version': self.map_version, 'revision': self.revision,
                'fast_revision': self.fast_revision, 'geometry_revision': self.geometry_revision}

    def current_snapshot(self):
        """Immutable raw returns for current Ground and layered collision consumers."""
        with self.lock:
            rows = []
            for sensor, (meta, points, origin, ground) in self.current_scans.items():
                deadline = min(meta.valid_until_s, meta.capture_s + self.config.occupancy_ttl_s)
                if deadline < self.now:
                    continue
                rows.append({'sensor_id': sensor, 'observation_id': meta.observation_id,
                    'capture_s': meta.capture_s, 'valid_until_s': deadline,
                    'points_odom': points, 'origin_odom': origin, 'ground_mask': ground})
            return {'stamp_s': self.now, 'version': self._version_locked(), 'scans': tuple(rows)}

    def snapshot(self):
        with self.lock:
            now, center = self.now, self.center.copy()
            grids = [(grid.scale, grid.cells) for grid in self.grids]
            result = {'stamp_s': self.now, 'motion_epoch': self.epoch, 'frame_id': 'odom',
                    'version': self._version_locked(),
                    'motion_quality': self.motion_quality, 'motion_covariance_valid': self.motion_covariance_valid,
                    'calibration_version': self.calibration, 'map_version': self.map_version,
                    'layer_counts': {k: len(v) for k, v in self.layers.items()},
                    'source_age_s': {k: self.now - v for k, v in self.last_sources.items() if ':' not in k},
                    'ground_observations': len(self.ground_observations),
                    'ground_returns': sum(len(points) for _, points in self.ground_observations.values()),
                    'current_sensor_ids': sorted(self.current_scans),
                    # Work permission requires the caller's explicit contact,
                    # response and sweep-safety budget, absent in a summary.
                    'fine_work_permission': False}
        # Count one immutable version without holding up new Ground/obstacles.
        result['scales'] = {}
        for scale, table in grids:
            valid = ((table.expires >= now) & (np.max(np.abs(
                (table.keys[:, :2] + .5) * scale.resolution_m - center[:2]), axis=1)
                <= scale.radius_m + scale.resolution_m))
            scores = table.scores[valid]
            result['scales'][scale.name] = {'resolution_m': scale.resolution_m,
                'radius_m': scale.radius_m, 'occupied': int(np.count_nonzero(scores > 0)),
                'free': int(np.count_nonzero(scores < 0)),
                'ambiguous': int(np.count_nonzero(scores == 0))}
        return result
