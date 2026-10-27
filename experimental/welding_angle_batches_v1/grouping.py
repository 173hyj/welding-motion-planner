"""Exact static angle batching; feasibility is supplied by the caller.

Angles are bounded axis coordinates in degrees, never circular bearings. These
plans do not certify collision-free transitions or estimate robot motion time.
"""
from math import ceil, isfinite


def _inputs(availability, required):
    available = {float(a): set(welds) for a, welds in availability.items()}
    if any(not isfinite(a) for a in available):
        raise ValueError('angles must be finite')
    needed = set().union(*available.values()) if required is None else set(required)
    return available, needed


def minimal_angle_covers(availability, required=None, max_covers=10000):
    """Return ALL minimum-cardinality angle tuples, sorted deterministically.

    availability maps angle to an iterable of weld IDs. required defaults to
    their union. An unavailable required weld returns []; empty requirements
    return [()]. Exact branch-and-bound preserves equal-coverage angle choices.
    If enumeration exceeds max_covers, raise instead of silently truncating.
    """
    if max_covers < 1:
        raise ValueError('max_covers limit must be positive')
    available, needed = _inputs(availability, required)
    if not needed:
        return [()]
    if not needed <= set().union(*available.values()):
        return []
    ids = sorted(needed)
    masks = []
    angles = []
    for angle in sorted(available):
        mask = sum(1 << i for i, wid in enumerate(ids) if wid in available[angle])
        if mask:
            angles.append(angle)
            masks.append(mask)
    full = (1 << len(ids)) - 1
    options = [[j for j, mask in enumerate(masks) if mask & (1 << i)] for i in range(len(ids))]
    for limit in range(1, len(ids) + 1):
        answers = set()
        visited = set()

        def search(chosen, covered):
            if chosen in visited:
                return
            visited.add(chosen)
            if covered == full:
                answers.add(tuple(angles[j] for j in sorted(chosen)))
                if len(answers) > max_covers:
                    raise ValueError('minimum cover enumeration exceeds max_covers limit')
                return
            remaining = full ^ covered
            slots = limit - len(chosen)
            if slots <= 0:
                return
            gain = max((mask & remaining).bit_count() if hasattr(int, 'bit_count') else bin(mask & remaining).count('1') for mask in masks)
            count = bin(remaining).count('1')
            if ceil(count / gain) > slots:
                return
            target = min((i for i in range(len(ids)) if remaining & (1 << i)), key=lambda i: len(options[i]))
            for j in options[target]:
                search(chosen | {j}, covered | masks[j])

        search(frozenset(), 0)
        if answers:
            return sorted(answers)
    return []


def evaluate_plan(groups, start_angle=0):
    """Evaluate the given ordered groups, with no return-home move."""
    previous = float(start_angle)
    travel = 0.0
    rotations = 0
    stops = 0
    if not isfinite(previous):
        raise ValueError('start_angle must be finite')
    for group in groups:
        angle = float(group['angle'])
        if not isfinite(angle):
            raise ValueError('angles must be finite')
        distance = abs(angle - previous)
        travel += distance
        rotations += int(distance > 0)
        stops += 1
        previous = angle
    return {'stops': stops, 'rotations_count': rotations, 'total_travel_deg': travel}


def propose_batches(availability, start_angle=0, stop_penalty_s=0, required=None):
    """Return minimum-stop plans ranked by open-axis travel, then angle tuple.

    Each weld is assigned once, to the first supporting stop in the route.
    stop_penalty_s reports stop overhead only, NOT total cycle time; travel in
    degrees cannot be converted to seconds without a validated motion model.
    """
    if not isfinite(float(stop_penalty_s)) or stop_penalty_s < 0:
        raise ValueError('stop_penalty_s must be finite and nonnegative')
    available, needed = _inputs(availability, required)
    plans = []
    for cover in minimal_angle_covers(available, needed):
        # On a nonwrapping one-dimensional axis, a shortest open tour visits
        # one extreme then the other. Visiting the starting stop first is free.
        prefix = (float(start_angle),) if float(start_angle) in cover else ()
        remaining = tuple(a for a in cover if a not in prefix)
        routes = {prefix + remaining, prefix + tuple(reversed(remaining))}
        route = min(routes, key=lambda r: (evaluate_plan([{'angle': a} for a in r], start_angle)['total_travel_deg'], r))
        unassigned = needed.copy()
        groups = []
        for angle in route:
            assigned = unassigned & available[angle]
            groups.append({'angle': angle, 'weld_ids': sorted(assigned)})
            unassigned -= assigned
        metrics = evaluate_plan(groups, start_angle)
        plans.append(dict(metrics, angles=route, groups=groups, stop_penalty_s=len(route) * float(stop_penalty_s)))
    return sorted(plans, key=lambda p: (p['stops'], p['total_travel_deg'], p['angles']))
