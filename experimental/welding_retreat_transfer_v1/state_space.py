"""Explicit physical-axis interpolation and dimension-independent RRT-Connect."""
import numpy as np


def interpolate(start, end, resolutions):
    start, end, resolutions = map(lambda x: np.asarray(x, float), (start, end, resolutions))
    if start.shape != end.shape or resolutions.shape != start.shape or np.any(resolutions <= 0):
        raise ValueError('Matching axis vectors and positive resolutions required')
    count = max(1, int(np.ceil(np.max(abs(end-start) / resolutions))))
    count += count % 2  # Always include the exact midpoint as well as both ends.
    return [start + (end-start)*u for u in np.linspace(0., 1., count+1)]


def edge_is_valid(start, end, resolutions, state_valid):
    return all(state_valid(q) for q in interpolate(start, end, resolutions))


def connect_rrt(start, goal, low, high, edge_valid, *, seed=20261009,
                max_iterations=1500, step=.06):
    start, goal, low, high = map(lambda x: np.asarray(x, float), (start, goal, low, high))
    if (any(x.shape != start.shape for x in (goal, low, high)) or np.any(high <= low) or
        np.any(start < low) or np.any(start > high) or np.any(goal < low) or np.any(goal > high) or
        not edge_valid(start, start) or not edge_valid(goal, goal)):
        return None, dict(reason='endpoint_invalid', iterations=0)
    scale = high-low
    rng = np.random.default_rng(seed)
    trees = [dict(nodes=[start], parents=[-1]), dict(nodes=[goal], parents=[-1])]

    def extend(tree, target):
        nearest = int(np.argmin([np.linalg.norm((q-target)/scale) for q in tree['nodes']]))
        origin = tree['nodes'][nearest]
        delta = target-origin
        distance = np.linalg.norm(delta/scale)
        if distance < 1e-12:
            return nearest, True
        new = target if distance <= step else origin + delta*(step/distance)
        if not edge_valid(origin, new):
            return None, False
        tree['nodes'].append(new)
        tree['parents'].append(nearest)
        return len(tree['nodes'])-1, distance <= step

    def chain(tree, index):
        result = []
        while index >= 0:
            result.append(tree['nodes'][index])
            index = tree['parents'][index]
        return result[::-1]

    for iteration in range(max_iterations):
        side = iteration % 2
        a, b = trees[side], trees[1-side]
        mode = rng.random()
        if mode < .2:
            target = b['nodes'][int(rng.integers(len(b['nodes'])))]
        elif mode < .7:
            base = a['nodes'][int(rng.integers(len(a['nodes'])))]
            target = np.clip(base + rng.normal(size=len(start))*scale*.05, low, high)
        else:
            target = low + rng.random(len(start))*scale
        ia, _ = extend(a, target)
        if ia is None:
            continue
        for _ in range(150):
            ib, reached = extend(b, a['nodes'][ia])
            if ib is None:
                break
            if reached:
                pa, pb = chain(a, ia), chain(b, ib)
                path = pa + pb[-2::-1] if side == 0 else pb + pa[-2::-1]
                return path, dict(reason='bidirectional_connected', iterations=iteration+1,
                                  nodes=sum(len(t['nodes']) for t in trees))
    return None, dict(reason='iteration_limit', iterations=max_iterations,
                      nodes=sum(len(t['nodes']) for t in trees))
