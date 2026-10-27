"""Preserve accepted analytic and prior numeric actuator states without wrapping.

This merge records observed candidates; it makes no completeness assertion.
"""
from copy import deepcopy
import math
from numbers import Real


def _index(rows):
    indexed = {}
    for row in rows:
        try:
            pose = row.get('pose', row)
            key = tuple(pose[k] for k in ('robot_id', 'roll_deg', 'axis_sign'))
            if key in indexed:
                raise ValueError('Duplicate pose key: %r' % (key,))
            if not isinstance(row['geometry']['passed'], bool):
                raise ValueError('Geometry passed flag must be boolean')
            accepted = row['accepted']
            if not isinstance(accepted, list):
                raise ValueError('Accepted candidates must be a list')
            if accepted and not row['geometry']['passed']:
                raise ValueError('Accepted candidate has rejected fixed geometry')
            for state in accepted:
                q = state['q_rad']
                if (len(q) != 6 or any(isinstance(v, bool) or not isinstance(v, Real)
                                      or not math.isfinite(v) for v in q)):
                    raise ValueError('Accepted candidate requires six finite actuator values')
                if (state['source_joint_limits_valid'] is not True
                        or state['acceptance']['passed'] is not True):
                    raise ValueError('Accepted candidate must pass acceptance and source limits')
            indexed[key] = row
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('Malformed candidate trial') from exc
    return indexed


def merge_trials(analytic_rows, numeric_rows, tolerance=1e-4):
    """Return deep-copied trial union and aggregate observed-candidate counts.

    Every analytic accepted state is preserved. Numeric accepted states within
    strictly ``tolerance`` raw actuator Euclidean distance of an analytic state
    mark that state as shared; all remaining numeric states are retained.
    Input rows must have identical unique pose coverage and fixed-geometry flags.
    """
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Tolerance must be positive and finite')
    analytic = _index(analytic_rows)
    numeric = _index(numeric_rows)
    if analytic.keys() != numeric.keys():
        raise ValueError('Analytic and numeric pose coverage differs')
    stats = dict(numeric_candidates=0, analytic_candidates=0, matched_numeric=0,
                 numeric_retained=0, new_analytic=0, union_candidates=0)
    merged = []
    for key, original in analytic.items():
        old = numeric[key]
        if original['geometry']['passed'] != old['geometry']['passed']:
            raise ValueError('Fixed geometry passed flags differ for pose %r' % (key,))
        row = deepcopy(original)
        accepted = deepcopy(original['accepted'])
        row['accepted'] = accepted
        for state in accepted:
            state['origin'] = 'analytic'
        shared = set()
        retained = []
        for state in old['accepted']:
            matches = [i for i, candidate in enumerate(accepted)
                       if math.dist(state['q_rad'], candidate['q_rad']) < tolerance]
            if matches:
                stats['matched_numeric'] += 1
                shared.update(matches)
            else:
                copy = deepcopy(state)
                copy['origin'] = 'numeric_retained'
                retained.append(copy)
        for i in shared:
            accepted[i]['origin'] = 'analytic_and_numeric'
        stats['analytic_candidates'] += len(accepted)
        stats['numeric_candidates'] += len(old['accepted'])
        stats['new_analytic'] += len(accepted) - len(shared)
        stats['numeric_retained'] += len(retained)
        accepted.extend(retained)
        row['numeric_retained'] = deepcopy(retained)
        row['numeric_retained_count'] = len(retained)
        if accepted:
            row['status'] = 'candidates_found'
        stats['union_candidates'] += len(accepted)
        merged.append(row)
    return merged, stats


