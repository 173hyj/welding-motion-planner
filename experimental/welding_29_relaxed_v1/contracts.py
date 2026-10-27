"""Source-order, branch and process contracts for the relaxed workflow."""
import numpy as np


def quintic(t):
    return 10*t**3 - 15*t**4 + 6*t**5


def legal_charts(model, q, reference):
    charts = []
    for shift in (-2*np.pi, 0., 2*np.pi):
        candidate = np.asarray(q, dtype=float).copy()
        candidate[5] += shift
        if model.limits_valid(candidate):
            charts.append(candidate)
    return sorted(charts, key=lambda x: np.linalg.norm(x - reference))


def workflow_phases(has_next):
    phases = ['ingress', 'close', 'weld', 'open', 'retreat']
    return phases + ['transfer'] if has_next else phases


def validate_workflow(source_ids, records):
    if [r['id'] for r in records] != list(source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError('missing or reordered source welds; expected all 29 source tasks')
    for i, record in enumerate(records):
        if record['phases'] != workflow_phases(i < len(records)-1):
            raise ValueError(f'incomplete workflow at {record["id"]}')
        for key in ['entry_q_rad', 'exit_q_rad']:
            if np.asarray(record[key]).shape != (6,) or not np.all(np.isfinite(record[key])):
                raise ValueError(f'invalid {key}')
    return True
