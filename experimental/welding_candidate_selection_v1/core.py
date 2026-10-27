"""Pure layered candidate selection on bounded, unwrapped raw configurations.

States contain twelve robot joints and one positioner angle in radians, then
two gun drivers in mm. These functions do not certify collision feasibility.
"""

import numpy as np


def _cost_array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {result.shape}')
    if np.isnan(result).any() or (result < 0).any():
        raise ValueError(f'{name} must contain nonnegative costs or positive infinity')
    return result


def shortest_path(layers, edge_costs, start_cost=None, end_cost=None):
    """Return minimum-cost indices and total cost, or None when infeasible.

    Layer entries are opaque; matrices have shape (len(layer_i), len(layer_i+1)).
    Positive infinity forbids an edge or endpoint. Equal-cost solutions choose
    the lexicographically smallest complete index sequence. Inputs are unchanged.
    Empty layers are infeasible; an empty graph has cost zero.
    """
    sizes = [len(layer) for layer in layers]
    if len(edge_costs) != max(0, len(sizes) - 1):
        raise ValueError('one edge matrix required per consecutive layer pair')
    matrices = [_cost_array(e, (a, b), f'edge_costs[{i}]')
                for i, (e, a, b) in enumerate(zip(edge_costs, sizes[:-1], sizes[1:]))]
    if not sizes:
        if start_cost is not None:
            _cost_array(start_cost, (0,), 'start_cost')
        if end_cost is not None:
            _cost_array(end_cost, (0,), 'end_cost')
        return {'indices': [], 'cost': 0.}
    start = np.zeros(sizes[0]) if start_cost is None else _cost_array(start_cost, (sizes[0],), 'start_cost')
    end = np.zeros(sizes[-1]) if end_cost is None else _cost_array(end_cost, (sizes[-1],), 'end_cost')
    if 0 in sizes:
        return None
    # Retain paths to resolve ties by the complete sequence, not only by the
    # immediate predecessor (which need not preserve lexicographic order).
    paths = [(i,) if np.isfinite(c) else None for i, c in enumerate(start)]
    costs = start.copy()
    for matrix in matrices:
        next_costs = np.full(matrix.shape[1], np.inf)
        next_paths = [None] * matrix.shape[1]
        for j in range(matrix.shape[1]):
            choices = [(float(costs[i] + matrix[i, j]), paths[i] + (j,))
                       for i in range(matrix.shape[0])
                       if paths[i] is not None and np.isfinite(costs[i] + matrix[i, j])]
            if choices:
                next_costs[j], next_paths[j] = min(choices)
        costs, paths = next_costs, next_paths
    choices = [(float(c + e), p) for c, e, p in zip(costs, end, paths)
               if p is not None and np.isfinite(c + e)]
    if not choices:
        return None
    cost, path = min(choices)
    return {'indices': list(path), 'cost': cost}


def shortest_path_with_time(layers, edge_costs, edge_times, start_cost=None,
                            end_cost=None, start_time=None, end_time=None,
                            time_budget=np.inf, max_labels=100000):
    """Exact layered resource-constrained shortest path with Pareto labels.

    Return indices, cost and time, or None. Endpoint arrays follow the scalar
    solver's conventions. All costs/times must be nonnegative or positive inf.
    The time budget is a strict upper bound without numerical slack. Equal-cost
    results prefer the lexicographically smallest complete index sequence.

    max_labels bounds the total retained frontier across each layer and raises
    RuntimeError rather than silently approximating. Dominance uses exact float
    comparisons, preserving a lexicographically better equal-cost prefix even
    when slower, because that prefix may win the final tie within budget.
    """
    if not np.isscalar(time_budget) or np.isnan(time_budget) or time_budget < 0:
        raise ValueError('time_budget must be nonnegative or positive infinity')
    if isinstance(max_labels, bool) or not isinstance(max_labels, (int, np.integer)) or max_labels < 1:
        raise ValueError('max_labels must be a positive integer')
    sizes = [len(layer) for layer in layers]
    count = max(0, len(sizes) - 1)
    if len(edge_costs) != count or len(edge_times) != count:
        raise ValueError('one cost and time matrix required per consecutive layer pair')
    costs = [_cost_array(e, (a, b), f'edge_costs[{i}]')
             for i, (e, a, b) in enumerate(zip(edge_costs, sizes[:-1], sizes[1:]))]
    times = [_cost_array(e, (a, b), f'edge_times[{i}]')
             for i, (e, a, b) in enumerate(zip(edge_times, sizes[:-1], sizes[1:]))]
    first, last = (sizes[0], sizes[-1]) if sizes else (0, 0)
    sc, ec, st, et = [np.zeros(n) if value is None else _cost_array(value, (n,), name)
                      for value, n, name in ((start_cost, first, 'start_cost'),
                                             (end_cost, last, 'end_cost'),
                                             (start_time, first, 'start_time'),
                                             (end_time, last, 'end_time'))]
    if not sizes:
        return {'indices': [], 'cost': 0., 'time': 0.}
    if 0 in sizes:
        return None

    def dominates(a, b):
        return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[2] <= b[2])

    frontier = [[(float(c), float(t), (i,))] if np.isfinite(c) and np.isfinite(t) and t <= time_budget else []
                for i, (c, t) in enumerate(zip(sc, st))]
    if sum(map(len, frontier)) > max_labels:
        raise RuntimeError('Pareto frontier exceeds max_labels')
    for cost_matrix, time_matrix in zip(costs, times):
        next_frontier = []
        retained = 0
        for j in range(cost_matrix.shape[1]):
            labels = []
            for i, predecessors in enumerate(frontier):
                edge_cost, edge_time = cost_matrix[i, j], time_matrix[i, j]
                if not np.isfinite(edge_cost) or not np.isfinite(edge_time):
                    continue
                for c, t, path in predecessors:
                    candidate = (float(c + edge_cost), float(t + edge_time), path + (j,))
                    if not np.isfinite(candidate[0]) or not np.isfinite(candidate[1]) or candidate[1] > time_budget:
                        continue
                    if any(dominates(existing, candidate) for existing in labels):
                        continue
                    labels = [existing for existing in labels if not dominates(candidate, existing)]
                    labels.append(candidate)
            retained += len(labels)
            if retained > max_labels:
                raise RuntimeError('Pareto frontier exceeds max_labels')
            next_frontier.append(labels)
        frontier = next_frontier
    options = [(float(c + ec[i]), path, float(t + et[i]))
               for i, labels in enumerate(frontier) for c, t, path in labels
               if np.isfinite(c + ec[i]) and np.isfinite(t + et[i]) and t + et[i] <= time_budget]
    if not options:
        return None
    cost, path, time = min(options)
    return {'indices': list(path), 'cost': cost, 'time': time}


def raw_features(a, b):
    """Measure raw deltas without angle wrapping or equivalent-turn substitution.

    Joint summaries concern the twelve robot joints; wrist axes are zero-based
    3,4,5,9,10,11. The simultaneous-axis time lower bound includes positioner
    and drivers, with speeds 30 deg/s, 15 deg/s and 80 mm/s respectively.
    """
    states = [np.asarray(state, dtype=float) for state in (a, b)]
    if any(state.shape != (15,) or not np.isfinite(state).all() for state in states):
        raise ValueError('each configuration must be a finite raw 15-vector')
    delta = np.abs(states[1] - states[0])
    degrees = np.rad2deg(delta[:12])
    return {
        'max_joint_delta_deg': float(np.max(degrees)),
        'sum_joint_delta_deg': float(np.sum(degrees)),
        'wrist_delta_deg': float(np.sum(degrees[[3, 4, 5, 9, 10, 11]])),
        'time_lower_bound_s': float(max(np.max(degrees) / 30.,
                                       np.rad2deg(delta[12]) / 15.,
                                       np.max(delta[13:]) / 80.)),
    }


def configuration_cost(a, b, weights=None):
    """Weighted raw motion proxy, not a planned trajectory cost.

    Defaults are motion=1, wrist=.5, time=.05. Motion and wrist multiply summed
    robot deltas in radians; time multiplies seconds. Supplied mapping entries
    override these defaults and must be finite and nonnegative.
    """
    selected = {'motion': 1., 'wrist': .5, 'time': .05}
    if weights is not None:
        if set(weights) - set(selected):
            raise ValueError('unknown configuration weight')
        selected.update(weights)
    for name, weight in selected.items():
        if not np.isscalar(weight) or not np.isfinite(weight) or weight < 0:
            raise ValueError(f'{name} weight must be finite and nonnegative')
    features = raw_features(a, b)
    return float(selected['motion'] * np.deg2rad(features['sum_joint_delta_deg'])
                 + selected['wrist'] * np.deg2rad(features['wrist_delta_deg'])
                 + selected['time'] * features['time_lower_bound_s'])
