"""Exact directed open Hamiltonian order for small, fixed cost matrices."""
import numpy as np


def held_karp_open(costs):
    costs = np.asarray(costs, float)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or np.isnan(costs).any():
        raise ValueError('Square cost matrix without NaN required')
    n = len(costs)
    if n == 0:
        return []
    if n > 16:
        raise ValueError('Use heuristic ordering for more than 16 nodes')
    values = {(1 << i, i): 0. for i in range(n)}
    parents = {}
    for mask in range(1, 1 << n):
        for last in range(n):
            key = (mask, last)
            if key not in values:
                continue
            for nxt in range(n):
                if mask & (1 << nxt) or not np.isfinite(costs[last, nxt]):
                    continue
                dest = (mask | (1 << nxt), nxt)
                value = values[key] + costs[last, nxt]
                if value < values.get(dest, np.inf):
                    values[dest] = value
                    parents[dest] = key
    mask = (1 << n) - 1
    endings = [(values[(mask, i)], i) for i in range(n) if (mask, i) in values]
    if not endings:
        return None
    _, last = min(endings)
    key = (mask, last)
    order = [last]
    while key in parents:
        key = parents[key]
        order.append(key[1])
    return order[::-1]
