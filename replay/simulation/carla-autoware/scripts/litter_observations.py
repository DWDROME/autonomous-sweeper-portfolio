"""One admission point for map-frame litter observations used by task planning."""
from copy import deepcopy
from hashlib import sha256
import json
import math

import numpy as np
from shapely.geometry import Point, Polygon


SCHEMA = 'horizon-litter-observations-v1'


def observation_packet(payload):
    body = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(payload=payload, sha256=sha256(body).hexdigest())


def _polygon(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 64 or not np.isfinite(points).all():
        raise ValueError('invalid_observation_polygon')
    shape = Polygon(points)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError('invalid_observation_polygon')
    return shape


class LitterObservations:
    def __init__(self, *, allow_scenario=False, max_age_s=2.):
        self.allow_scenario = allow_scenario
        self.max_age_s = max_age_s
        self.reset()

    def reset(self):
        self.packet = None
        self.reason = 'waiting_for_observations'

    def ingest(self, packet, *, now, session_id, map_id):
        try:
            payload = packet['payload']
            if packet.get('sha256') != observation_packet(payload)['sha256']:
                raise ValueError('observation_hash_mismatch')
            fields = {'schema', 'source', 'sessionId', 'mapVersionId', 'frameId', 'sourceTime', 'observedArea', 'targets'}
            if set(payload) != fields or payload['schema'] != SCHEMA or payload['frameId'] != 'map':
                raise ValueError('observation_contract_mismatch')
            if payload['source'] not in ('perception', 'carla_scenario'):
                raise ValueError('unsupported_observation_source')
            if payload['source'] == 'carla_scenario' and not self.allow_scenario:
                raise ValueError('scenario_observations_disabled')
            if payload['sessionId'] != session_id or payload['mapVersionId'] != map_id:
                raise ValueError('observation_context_mismatch')
            stamp = payload['sourceTime']
            if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                    or now is None or not -1e-8 <= now-stamp <= self.max_age_s):
                raise ValueError('observation_time_invalid')
            if self.packet and stamp < self.packet['payload']['sourceTime']:
                raise ValueError('observation_time_reversed')
            observed = _polygon(payload['observedArea'])
            targets = payload['targets']
            if not isinstance(targets, list) or len(targets) > 256:
                raise ValueError('invalid_target_count')
            identities = set()
            for target in targets:
                common = {'id', 'kind', 'class', 'visible'}
                field = 'point' if target['kind'] == 'point' else 'polygon'
                if target['kind'] not in ('point', 'area') or set(target) != common | {field}:
                    raise ValueError('invalid_target_geometry')
                if (not isinstance(target['id'], str) or not 1 <= len(target['id']) <= 128
                        or target['id'] in identities or not isinstance(target['visible'], bool)
                        or not isinstance(target['class'], str) or not target['class']):
                    raise ValueError('invalid_target_identity_or_visibility')
                identities.add(target['id'])
                if field == 'point':
                    xy = np.asarray(target[field], dtype=float)
                    if xy.shape != (2,) or not np.isfinite(xy).all():
                        raise ValueError('invalid_target_point')
                    shape = Point(xy)
                else:
                    shape = _polygon(target[field])
                if target['visible'] and not observed.covers(shape):
                    raise ValueError('target_outside_observed_area')
            self.packet = deepcopy(packet)
            self.reason = 'available'
            return dict(accepted=True, targets=len(targets), source=payload['source'])
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            self.packet = None
            self.reason = str(error)
            return dict(accepted=False, reason=self.reason)

    def current(self, now):
        if self.packet is None:
            return None
        if now is None or not -1e-8 <= now-self.packet['payload']['sourceTime'] <= self.max_age_s:
            self.packet = None
            self.reason = 'observations_expired'
        return self.packet


def task_targets(packet, polygon, target_ids=None):
    """Select admitted observations within the assigned work area."""
    if packet is None:
        return []
    area = Polygon(polygon)
    selected = []
    for target in packet['payload']['targets']:
        if target_ids is not None and target['id'] not in target_ids:
            continue
        shape = Point(target['point']) if target['kind'] == 'point' else Polygon(target['polygon'])
        if area.covers(shape):
            selected.append(deepcopy(target))
    return selected
