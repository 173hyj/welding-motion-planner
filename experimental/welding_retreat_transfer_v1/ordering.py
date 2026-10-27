"""Classic nearest-neighbour multistart and 2-opt for an open visit route."""
import numpy as np


def route_cost(route, costs):
    return float(sum(costs[a, b] for a, b in zip(route, route[1:])))


def improve_order(route, costs):
    route = list(route)
    best = route_cost(route, costs)
    while True:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                # Full recomputation also handles directed and nonmetric costs.
                value = route_cost(candidate, costs)
                if value < best - 1e-10:
                    route, best, improved = candidate, value, True
                    break
            if improved:
                break
        if not improved:
            return route


def visit_order(costs):
    costs = np.asarray(costs, float)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or np.isnan(costs).any():
        raise ValueError('Square, non-NaN cost matrix required')
    n = len(costs)
    if not n:
        return []
    best, best_value = None, np.inf
    for start in range(n):
        route, remaining = [start], set(range(n)) - {start}
        while remaining:
            candidate = min(remaining, key=lambda j: (costs[route[-1], j], j))
            if not np.isfinite(costs[route[-1], candidate]):
                break
            route.append(candidate)
            remaining.remove(candidate)
        if remaining:
            continue
        route = improve_order(route, costs)
        value = route_cost(route, costs)
        if value < best_value:
            best, best_value = route, value
    return best
