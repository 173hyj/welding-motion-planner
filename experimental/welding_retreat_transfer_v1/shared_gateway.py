"""Move to a qualified common state without forcing guns closed near the part."""
import numpy as np
from state_space import connect_rrt


def connect_shared(bridge, a, b, gateway, arm_order, seed, iterations):
    a, b, gateway = (np.asarray(x, float) for x in (a, b, gateway))
    current = a.copy()
    path, trials = [current.copy()], []
    actions = [(list(range(6*i, 6*i+6))+[13+i], gateway, 'park_arm_and_gun')
               for i in arm_order]
    actions += [([12], b, 'rotate_positioner')]
    actions += [(list(range(6*i, 6*i+6))+[13+i], b, 'deploy_arm_and_gun')
                for i in reversed(arm_order)]
    for free, source, phase in actions:
        target = current.copy(); target[free] = source[free]
        if np.array_equal(target, current):
            continue
        if bridge.edge_scaled(current, target):
            segment, stats = [current.copy(), target], dict(reason='direct_edge')
        elif iterations:
            def unpack(part):
                full = current.copy(); full[free] = part
                return full
            partial, stats = connect_rrt(current[free], target[free],
                (bridge.low/bridge.scale)[free], (bridge.high/bridge.scale)[free],
                lambda x, y: bridge.edge_scaled(unpack(x), unpack(y)),
                seed=seed+len(trials), max_iterations=iterations, step=.012)
            segment = [unpack(p) for p in partial] if partial is not None else None
        else:
            segment, stats = None, dict(reason='direct_edge_blocked')
        trials.append(dict(phase=phase, free_axes=free, **stats))
        if segment is None:
            return None, trials
        path.extend(segment[1:]); current = target
    if not np.array_equal(current, b):
        raise RuntimeError('Shared gateway did not restore exact destination')
    return path, trials
