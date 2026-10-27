"""Exact layered shortest paths excluding complete sequences, never nodes.

Use lower-bound weights for uncertified edges and positive infinity for failed
edges. This solver does not certify edges or impose any candidate/beam cutoff.
"""
import numpy as np


def _costs(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or np.isnan(array).any() or (array < 0).any():
        raise ValueError(f'{name} must have shape {shape}, with nonnegative values or positive infinity')
    return array


def shortest_path_avoiding(layers, edge_costs, start_cost=None, end_cost=None,
                           forbidden_paths=()):
    """Return {'indices': list, 'cost': float}, or None if no finite path.

    Layers contain opaque items. Consecutive cost matrices have shape
    (N_i,N_next); endpoint vectors default to zero. Costs must be nonnegative;
    +inf forbids an edge/endpoint. Forbidden paths are complete index sequences.
    Ties prefer the lexicographically smallest sequence. Inputs are unchanged.

    A prefix trie tracks whether a partial path still matches a forbidden
    sequence. All prefixes that have escaped that trie merge into one state per
    graph node. Thus this product-DAG DP avoids enumerating all possible paths:
    per-layer states are bounded by node count plus forbidden-prefix count.
    Repeated calls with each prior result forbidden enumerate exact k-best paths.
    """
    sizes = [len(layer) for layer in layers]
    if len(edge_costs) != max(0, len(sizes) - 1):
        raise ValueError('one edge matrix required per consecutive layer pair')
    matrices = [_costs(e, (a, b), f'edge_costs[{k}]')
                for k, (e, a, b) in enumerate(zip(edge_costs, sizes[:-1], sizes[1:]))]
    first, last = (sizes[0], sizes[-1]) if sizes else (0, 0)
    start = np.zeros(first) if start_cost is None else _costs(start_cost, (first,), 'start_cost')
    end = np.zeros(last) if end_cost is None else _costs(end_cost, (last,), 'end_cost')
    # Trie nodes are dictionaries mapping candidate index to next trie node.
    children, terminal = [{}], set()
    for sequence in forbidden_paths:
        path = tuple(sequence)
        if len(path) != len(sizes) or any(isinstance(i, (bool, np.bool_)) or
                not isinstance(i, (int, np.integer)) or i < 0 or i >= n
                for i, n in zip(path, sizes)):
            raise ValueError('forbidden path must be a complete valid index sequence')
        trie = 0
        for index in path:
            if index not in children[trie]:
                children[trie][index] = len(children)
                children.append({})
            trie = children[trie][index]
        terminal.add(trie)
    if not sizes:
        return None if 0 in terminal else {'indices': [], 'cost': 0.}
    if 0 in sizes:
        return None
    # -1 denotes an escaped prefix, which can never equal a forbidden path.
    frontier = {(i, children[0].get(i, -1)): (float(c), (i,))
                for i, c in enumerate(start) if np.isfinite(c)}
    for matrix in matrices:
        next_frontier = {}
        for (i, trie), (cost, path) in frontier.items():
            for j, edge_cost in enumerate(matrix[i]):
                total = float(cost + edge_cost)
                if not np.isfinite(total):
                    continue
                next_trie = children[trie].get(j, -1) if trie != -1 else -1
                key, candidate = (j, next_trie), (total, path + (j,))
                if key not in next_frontier or candidate < next_frontier[key]:
                    next_frontier[key] = candidate
        frontier = next_frontier
    options = [(float(cost + end[i]), path) for (i, trie), (cost, path) in frontier.items()
               if trie not in terminal and np.isfinite(cost + end[i])]
    if not options:
        return None
    cost, path = min(options)
    return {'indices': list(path), 'cost': cost}
