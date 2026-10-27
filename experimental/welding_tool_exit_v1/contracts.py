"""Tool-frame straight extraction and conservative separated-interval checks."""
import itertools
import math
import numpy as np


def direction_bank():
    """Fixed orientation; translation rays only, on a 5-degree azimuth grid."""
    rays = [np.eye(3)[i]*sign for i in (2, 0, 1) for sign in (-1, 1)]
    for elevation in (-5, -15, -30, -45, -60, -75, 0, 5, 15, 30, 45, 60, 75):
        e = math.radians(elevation)
        for azimuth in range(0, 360, 5):
            a = math.radians(azimuth)
            rays.append(np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)]))
    return list({tuple(np.round(d, 12)): d for d in rays}.values())


def certify_translation(distance, start_mm, end_mm, resolution_mm=.01,
                        clearance_mm=0., max_queries=20000):
    """1-Lipschitz surface distance for pure translation in millimetres.

    If midpoint distance exceeds half interval length, the entire interval
    is separated. Unresolved intervals fail conservatively. Contact release
    is deliberately outside this strictly separated certificate.
    """
    if not 0 <= start_mm <= end_mm or resolution_mm <= 0:
        raise ValueError('Invalid translation interval or resolution')
    cache = {}; intervals = []; minimum = float('inf')
    def query(s):
        nonlocal minimum
        if s not in cache:
            cache[s] = float(distance(s))
            minimum = min(minimum, cache[s])
        return cache[s]
    pending = [(float(start_mm), float(end_mm))]
    for s in (float(start_mm), float(end_mm)):
        d = query(s)
        if not np.isfinite(d) or d <= clearance_mm + 1e-8:
            return dict(passed=False, reason='collision_or_zero_clearance', failure_mm=s,
                        queries=len(cache), minimum_sample_distance_mm=minimum)
    while pending:
        a, b = pending.pop(); mid = (a+b)/2.; d = query(mid)
        if not np.isfinite(d) or d <= clearance_mm + 1e-8:
            return dict(passed=False, reason='collision_or_zero_clearance', failure_mm=mid,
                        queries=len(cache), minimum_sample_distance_mm=minimum)
        lower = d-(b-a)/2.
        if lower > clearance_mm + 1e-8:
            intervals.append((a, b, lower))
            continue
        if b-a <= resolution_mm or len(cache) >= max_queries:
            return dict(passed=False, reason='unresolved_clearance', failure_mm=mid,
                        queries=len(cache), minimum_sample_distance_mm=minimum)
        pending.extend([(mid, b), (a, mid)])
    return dict(passed=True, reason='translation_distance_lower_bound', queries=len(cache),
                minimum_sample_distance_mm=minimum,
                minimum_interval_lower_bound_mm=min((x[2] for x in intervals), default=minimum),
                certified_intervals=len(intervals))


class ContactTracker:
    def __init__(self, initial_electrodes, max_release_mm=5.):
        self.allowed = set(initial_electrodes)
        self.separated = set()
        self.max_release = float(max_release_mm)
        self.previous = 0.

    def advance(self, touching, travel_mm):
        touching = set(touching)
        if (travel_mm < self.previous-1e-8 or not touching <= self.allowed or
            touching & self.separated or (touching and travel_mm > self.max_release)):
            return False
        self.separated |= self.allowed-touching
        self.previous = float(travel_mm)
        return True


def nearest_equivalent(model, q, previous):
    """Choose actual bounded joint coordinates; never wrap interpolation edges."""
    q, previous = np.asarray(q, float), np.asarray(previous, float)
    options = []
    for i, angle in enumerate(q):
        low, high = model.bounds[:, i]
        ks = range(math.ceil((low-angle)/(2*np.pi)), math.floor((high-angle)/(2*np.pi))+1)
        options.append(sorted((angle+2*np.pi*k for k in ks), key=lambda a: abs(a-previous[i]))[:3])
    legal = [np.asarray(state) for state in itertools.product(*options)
             if model.limits_valid(np.asarray(state))]
    if not legal:
        raise ValueError('No source-legal equivalent joint representation')
    return min(legal, key=lambda state: float(np.linalg.norm(state-previous)))
