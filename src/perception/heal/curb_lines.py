"""Connect bounded image candidates into local, direction-independent segments.

Pixel segments are model observations. They carry no metric or work permission;
measured ground, camera calibration and the existing curb geometry stay downstream.
"""
import math
import numpy as np


def connect_candidates(xy, scores, valid, *, truncated=False, max_gap_px=6., max_turn_degrees=40.):
    """Connect adjacent stride-4 peaks, preserving holes and source slot indices.

    Each view contains at most 256 slots. Ambiguous local direction is left
    unconnected; no polynomial fit, extrapolation, or closing of missing spans.
    ``max_gap_px=6`` accepts neighboring diagonal cells but not a missing cell
    in a straight line. These pixel thresholds are initial development values.
    """
    xy, scores, valid = np.asarray(xy), np.asarray(scores), np.asarray(valid)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) > 256
            or scores.shape != (len(xy),) or valid.shape != scores.shape or valid.dtype != np.bool_
            or not np.isfinite(xy).all() or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
            or not math.isfinite(max_gap_px) or max_gap_px <= 0
            or not 0 < max_turn_degrees < 90):
        raise ValueError('expected at most 256 finite pixel candidates, scores and a boolean mask')
    source = np.flatnonzero(valid)
    points = xy[source].astype(float)
    count = len(points)
    delta = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(delta, axis=2)
    adjacent = (distances > 1e-6) & (distances <= max_gap_px)
    tangents, supported = np.zeros_like(points), np.zeros(count, dtype=bool)
    for i in range(count):
        vectors = delta[i, adjacent[i]]
        if not len(vectors):
            continue
        directions = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(directions.T @ directions)
        # Corners/junctions/close parallel lines may be ambiguous. Splitting
        # here preserves uncertainty rather than inventing a continuation.
        supported[i] = values[-1] >= .8 * values.sum()
        tangents[i] = vectors[:, -1]
    cosine = math.cos(math.radians(max_turn_degrees))
    pairs = np.column_stack(np.nonzero(np.triu(adjacent, 1)))
    pairs = sorted(pairs.tolist(), key=lambda pair: (distances[tuple(pair)], *pair))
    neighbors = [[] for _ in points]
    for i, j in pairs:
        if not supported[i] or not supported[j] or len(neighbors[i]) >= 2 or len(neighbors[j]) >= 2:
            continue
        direction = delta[i, j] / distances[i, j]
        if min(abs(tangents[i] @ direction), abs(tangents[j] @ direction)) < cosine:
            continue
        # A second edge must continue in the opposite direction at a vertex.
        if any(delta[a, existing] @ delta[a, b] > -cosine * distances[a, existing] * distances[a, b]
               for a, b in ((i, j), (j, i)) for existing in neighbors[a]):
            continue
        neighbors[i].append(j)
        neighbors[j].append(i)
    seen, lines = set(), []
    starts = sorted(range(count), key=lambda i: (len(neighbors[i]) == 2, i))
    for start in starts:
        if start in seen or not neighbors[start]:
            continue
        path, current, previous = [], start, None
        while current not in seen:
            seen.add(current)
            path.append(current)
            next_points = [i for i in neighbors[current] if i != previous]
            if not next_points:
                break
            previous, current = current, next_points[0]
        if len(path) >= 3:
            indices = source[path]
            lines.append(dict(candidate_indices=indices.tolist(), xy=xy[indices].tolist(),
                              scores=scores[indices].tolist(), closed=current == start and len(neighbors[start]) == 2))
    used = {index for line in lines for index in line['candidate_indices']}
    return dict(lines=lines, unconnected_count=count-len(used), truncated=bool(truncated),
                coordinate_frame='model_input_pixels', metric_valid=False,
                fine_work_permission=False, max_gap_px=float(max_gap_px))
