"""HEAL's native semantic grid contract; predictions never grant free space."""
import math
import numpy as np

GRID_SHAPE = (1, 200, 200, 16)
GRID_ORIGIN_M = (-40., -40., -5.)
GRID_RESOLUTION_M = (.4, .4, .5)
SCORE_SCALE = 1. / 127.
CLASS_NAMES = ('other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation', 'unmapped_17')
STATIC_SURFACE_CLASSES = (11, 12, 13, 14)


def validate_grid(labels, scores=None):
    """Scores are signed byte top-1 probabilities, round(p*127), zero point 0."""
    labels = np.asarray(labels)
    if (labels.shape != GRID_SHAPE or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES))):
        raise ValueError('expected native HEAL integer labels with shape 1x200x200x16')
    if scores is not None:
        scores = np.asarray(scores)
        if (scores.shape != GRID_SHAPE or scores.dtype != np.int8
                or np.any(scores < 0)):
            raise ValueError('semantic scores require nonnegative int8 with scale 1/127')
    return labels, scores


def save_grid(path, labels, scores, *, timestamp_ns, source_from_model, model_version, calibration_version):
    """PC file handoff; only compact arrays and their coordinate/source contract."""
    from ..local_world.motion import rigid
    labels, scores = validate_grid(labels, scores)
    if (scores is None or isinstance(timestamp_ns, (bool, np.bool_))
            or not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0
            or not model_version or not calibration_version):
        raise ValueError('compact grid requires scores, source time and model/calibration versions')
    np.savez(path, schema=np.asarray('heal-surface-v1'), labels=labels.astype(np.int8, copy=False),
        scores=scores, timestamp_ns=np.asarray(timestamp_ns, dtype=np.int64),
        source_from_model=rigid(source_from_model), model_version=np.asarray(model_version),
        calibration_version=np.asarray(calibration_version), score_scale=np.asarray(SCORE_SCALE))


def load_grid(path):
    from ..local_world.motion import rigid
    with np.load(path, allow_pickle=False) as data:
        if str(data['schema'].item()) != 'heal-surface-v1' or not math.isclose(float(data['score_scale']), SCORE_SCALE):
            raise ValueError('unsupported HEAL surface schema or score scale')
        labels, scores = validate_grid(data['labels'], data['scores'])
        stamp = data['timestamp_ns']
        if stamp.shape != () or stamp.dtype.kind not in 'iu' or int(stamp) < 0:
            raise ValueError('invalid surface source time')
        model, calibration = str(data['model_version'].item()), str(data['calibration_version'].item())
        if not model or not calibration:
            raise ValueError('surface versions are required')
        return dict(labels=labels, scores=scores, timestamp_ns=int(stamp),
            source_from_model=rigid(data['source_from_model']), model_version=model,
            calibration_version=calibration)
