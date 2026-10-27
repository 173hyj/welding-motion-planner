"""Deterministic, cost-gated shortcuts of a previously certified seed path.

Only newly introduced segments are certified here. Retained segments inherit
the caller's baseline certification; this helper is not a full-path validator.
"""

import numbers

import numpy as np


def greedy_shortcut(original, certify, cost, max_trials=80):
    """Return a source-corner subsequence and optimization statistics.

    ``cost`` evaluates the entire path (e.g. display time including FK).
    Callback failures on candidate paths count as attempts and are skipped.
    Invalid or failed baseline costs raise ValueError because comparisons would
    otherwise be meaningless. Input dtype and endpoint bytes are preserved.
    """
    path = np.asarray(original)
    if (path.ndim != 2 or path.shape[1] != 15 or len(path) < 2
            or not np.issubdtype(path.dtype, np.number)
            or np.iscomplexobj(path) or not np.all(np.isfinite(path))):
        raise ValueError('original must be a finite real Nx15 path with N >= 2')
    if (isinstance(max_trials, bool) or not isinstance(max_trials, numbers.Integral)
            or max_trials <= 0):
        raise ValueError('max_trials must be a positive integer')
    path = path.copy()
    try:
        original_cost = float(cost(path.copy()))
    except Exception as exc:
        raise ValueError('baseline cost evaluation failed') from exc
    if not np.isfinite(original_cost):
        raise ValueError('baseline cost must be finite')
    current_cost = original_cost
    source_indices = list(range(len(path)))
    attempts = accepted = 0
    limit_reached = False
    while True:
        # Longest source spans first; left-to-right tie breaking is stable.
        candidates = sorted(
            ((i, j) for i in range(len(path)) for j in range(i + 2, len(path))),
            key=lambda ij: (-(source_indices[ij[1]] - source_indices[ij[0]]), ij[0]))
        improved = False
        for i, j in candidates:
            if attempts >= max_trials:
                limit_reached = True
                break
            attempts += 1
            try:
                if not bool(certify(path[i].copy(), path[j].copy())):
                    continue
                trial = np.concatenate((path[:i + 1], path[j:]), axis=0)
                trial_cost = float(cost(trial.copy()))
            except Exception:
                continue
            if np.isfinite(trial_cost) and current_cost - trial_cost > 1e-9:
                path = trial
                source_indices = source_indices[:i + 1] + source_indices[j:]
                current_cost = trial_cost
                accepted += 1
                improved = True
                break
        if limit_reached or not improved:
            break
    return path, {
        'original_cost': original_cost, 'new_cost': current_cost,
        'attempts': attempts, 'accepted': accepted,
        'limit_reached': limit_reached,
    }


# Short name for callers following the module's operation name.
shortcut = greedy_shortcut
