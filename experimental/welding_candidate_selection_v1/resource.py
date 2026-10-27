"""Exact resource-constrained selection on an already certified layered graph.

This module consumes additive nonnegative measurements; it never wraps angles,
changes graph states, or certifies physical feasibility of graph edges.
"""
import numpy as np


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or np.isnan(array).any() or (array < 0).any():
        raise ValueError(f'{name} must have shape {shape} and nonnegative values or positive infinity')
    return array


def shortest_path_with_resources(layers, edge_costs, edge_resources, resource_limits,
                                 start_cost=None, end_cost=None, start_resources=None,
                                 end_resources=None, max_labels=100000):
    """Minimize scalar cost subject to every additive resource upper bound.

    edge_resources[i] has shape (N_i,N_next,R); resource_limits has shape (R,).
    Endpoint resource arrays have shape (N_endpoint,R), defaulting to zero.
    Endpoint costs have shape (N_endpoint,), defaulting to zero. Positive inf
    forbids an edge/endpoint even when its corresponding limit is infinite.
    Return {'indices': list, 'cost': float, 'resources': list} or None.

    Feasibility uses exact <= comparisons with no tolerance. Equal-cost feasible
    paths prefer the lexicographically smallest complete index sequence, even
    if it consumes more resources. Therefore equal-cost dominance additionally
    requires a lexicographically no-worse prefix. max_labels caps the total
    retained frontier per layer; exceeding it raises RuntimeError, never a beam
    approximation. Empty layers are infeasible; an empty graph has zero totals.
    """
    limits = np.asarray(resource_limits, dtype=float)
    if limits.ndim != 1 or np.isnan(limits).any() or (limits < 0).any():
        raise ValueError('resource_limits must be a nonnegative vector or positive infinity')
    if isinstance(max_labels, bool) or not isinstance(max_labels, (int, np.integer)) or max_labels < 1:
        raise ValueError('max_labels must be a positive integer')
    sizes = [len(layer) for layer in layers]
    count = max(0, len(sizes) - 1)
    if len(edge_costs) != count or len(edge_resources) != count:
        raise ValueError('one cost/resource matrix required per consecutive layer pair')
    rcount = len(limits)
    costs = [_array(e, (a, b), f'edge_costs[{i}]')
             for i, (e, a, b) in enumerate(zip(edge_costs, sizes[:-1], sizes[1:]))]
    resources = [_array(e, (a, b, rcount), f'edge_resources[{i}]')
                 for i, (e, a, b) in enumerate(zip(edge_resources, sizes[:-1], sizes[1:]))]
    first, last = (sizes[0], sizes[-1]) if sizes else (0, 0)
    sc, ec, sr, er = [np.zeros(shape) if value is None else _array(value, shape, name)
                      for value, shape, name in ((start_cost, (first,), 'start_cost'),
                                                (end_cost, (last,), 'end_cost'),
                                                (start_resources, (first, rcount), 'start_resources'),
                                                (end_resources, (last, rcount), 'end_resources'))]
    if not sizes:
        return {'indices': [], 'cost': 0., 'resources': np.zeros(rcount).tolist()}
    if 0 in sizes:
        return None

    def dominates(a, b):
        return a[0] <= b[0] and np.all(a[1] <= b[1]) and (a[0] < b[0] or a[2] <= b[2])

    frontier = [[(float(c), r.copy(), (i,))]
                if np.isfinite(c) and np.isfinite(r).all() and np.all(r <= limits) else []
                for i, (c, r) in enumerate(zip(sc, sr))]
    if sum(map(len, frontier)) > max_labels:
        raise RuntimeError('Pareto frontier exceeds max_labels')
    for cost_matrix, resource_matrix in zip(costs, resources):
        next_frontier, retained = [], 0
        for j in range(cost_matrix.shape[1]):
            labels = []
            for i, predecessors in enumerate(frontier):
                edge_cost, edge_resource = cost_matrix[i, j], resource_matrix[i, j]
                if not np.isfinite(edge_cost) or not np.isfinite(edge_resource).all():
                    continue
                for c, r, path in predecessors:
                    new_cost, new_resource = float(c + edge_cost), r + edge_resource
                    if not np.isfinite(new_cost) or not np.isfinite(new_resource).all() or np.any(new_resource > limits):
                        continue
                    candidate = (new_cost, new_resource, path + (j,))
                    if any(dominates(existing, candidate) for existing in labels):
                        continue
                    labels = [existing for existing in labels if not dominates(candidate, existing)]
                    labels.append(candidate)
            retained += len(labels)
            if retained > max_labels:
                raise RuntimeError('Pareto frontier exceeds max_labels')
            next_frontier.append(labels)
        frontier = next_frontier
    best = None
    for i, labels in enumerate(frontier):
        for c, r, path in labels:
            cost, resource = float(c + ec[i]), r + er[i]
            if not np.isfinite(cost) or not np.isfinite(resource).all() or np.any(resource > limits):
                continue
            if best is None or (cost, path) < best[:2]:
                best = (cost, path, resource)
    if best is None:
        return None
    cost, path, resource = best
    return {'indices': list(path), 'cost': cost, 'resources': resource.tolist()}
