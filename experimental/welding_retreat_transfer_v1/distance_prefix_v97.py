"""Extract an exact saved endpoint; the new endpoint needs fresh certification."""
import copy
import math


def crop_exact_prefix(path, distance):
    distance = float(distance)
    if (not math.isfinite(distance) or distance <= 0. or
            not path.get('passed') or distance > path['length_mm']):
        raise ValueError('Invalid passed-path prefix distance')
    frames = path['frames']
    travels = [float(f['travel_mm']) for f in frames]
    if any(not math.isfinite(s) for s in travels) or any(
            b < a - 1e-5 for a, b in zip(travels, travels[1:])):
        raise ValueError('Invalid nonmonotonic saved path')
    indices = [i for i, s in enumerate(travels) if abs(s - distance) <= 1e-6]
    if not indices:
        raise ValueError('Requested endpoint absent; refit rather than interpolate')
    result = copy.deepcopy(path)
    result['frames'] = result['frames'][:indices[-1] + 1]
    result['q_end_rad'] = result['frames'][-1]['q_rad'][:]
    result['length_mm'] = result['fixed_distance_mm'] = distance
    result['passed'] = False
    for key in list(result):
        if key.startswith('maximum_') or key in ('endpoint_clearance_mm', 'geometry_certificate'):
            result.pop(key)
    return result
