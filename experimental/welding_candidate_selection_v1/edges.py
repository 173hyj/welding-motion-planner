"""Raw-source edge planning; coarse paths never authorize a new saved edge."""
from pathlib import Path
import sys
import time
import numpy as np

SCALE = np.r_[np.ones(13), 100., 100.]
FREE = [0, 1, 2, 3, 4, 5, 13]
FIXED = [6, 7, 8, 9, 10, 11, 12, 14]


def _dependencies():
    folder = Path(__file__).resolve().parent.parent / 'welding_retreat_transfer_v1'
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from world_bridge_v2 import SourceBridge
    from state_space import connect_rrt, interpolate
    return SourceBridge, connect_rrt, interpolate


def path_length(states):
    return float(np.linalg.norm(np.diff(np.asarray(states)/SCALE, axis=0), axis=1).sum())


def measured_duration(states, engine):
    """Display duration: simultaneous TCP, joint and driver speed constraints."""
    states = np.asarray(states, float)
    if len(states) < 2:
        return 0.
    speed = np.r_[np.full(12, np.deg2rad(30)), np.deg2rad(15), 80., 80.]
    times = np.max(np.abs(np.diff(states, axis=0))/speed, axis=1)
    for i, rid in enumerate(sorted(engine.world.models)):
        model = engine.world.models[rid]
        xyz = np.array([model.tcp(q[6*i:6*i+6])[:3, 3] for q in states])
        times = np.maximum(times, np.linalg.norm(np.diff(xyz, axis=0), axis=1)/200.)
    return float(times.sum())


def accept_candidate(original, candidate, engine, fine_passed):
    candidate = np.asarray(candidate, float)
    if candidate.ndim != 2 or candidate.shape[1] != 15 or not np.isfinite(candidate).all():
        return False, 'invalid_candidate'
    if not np.array_equal(original[[0, -1]], candidate[[0, -1]]):
        return False, 'endpoint_changed'
    if not np.all(candidate[:, FIXED] == original[0, FIXED]):
        return False, 'fixed_axes_changed'
    if not fine_passed:
        return False, 'fine_source_failed'
    if path_length(candidate) > path_length(original)+1e-8:
        return False, 'normalized_length_regression'
    if measured_duration(candidate, engine) > measured_duration(original, engine)+1e-8:
        return False, 'duration_regression'
    return True, 'fine_source_and_motion_metrics_accepted'


def _shortcut(path, bridge):
    output, i = [path[0]], 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1 and not bridge.edge_scaled(path[i], path[j]):
            j -= 1
        output.append(path[j]); i = j
    return np.asarray(output)


def _dense(path, bridge, interpolate):
    rows = []
    for a, b in zip(path, path[1:]):
        samples = interpolate(a, b, bridge.resolutions/bridge.scale)
        rows.extend(samples if not rows else samples[1:])
    result = np.asarray(rows)*bridge.scale
    result[0] = path[0]*bridge.scale
    result[-1] = path[-1]*bridge.scale
    return result


def compress_collinear(states, tolerance=1e-10):
    """Remove only redundant straight-line samples; keep actual polyline corners."""
    states = np.asarray(states, float)
    keep = [0]
    for i in range(1, len(states)-1):
        first = (states[i]-states[keep[-1]])/SCALE
        second = (states[i+1]-states[i])/SCALE
        norm = np.linalg.norm(first)
        if norm <= tolerance:
            continue
        direction = first/norm
        projection = float(second@direction)
        if projection < -tolerance or np.linalg.norm(second-projection*direction) > tolerance:
            keep.append(i)
    keep.append(len(states)-1)
    return states[keep].copy()


def _trajopt(states, engine, fine, proposal, mesh_bounds, interpolate, seed_path=None):
    stats = dict(attempted=True, accepted=False)
    try:
        from experimental.trajopt_transfer_v1.solver import optimize, Options
        from experimental.trajopt_transfer_v1.run_optimize import refine_seed
        from experimental.trajopt_transfer_v1.proximity import SourceProximity
        corners = compress_collinear(states)/fine.scale if seed_path is None else np.asarray(seed_path)
        if len(corners) > 40:
            return states, dict(attempted=False, accepted=False, reason='too_many_polyline_corners')
        if len(corners) <= 2:
            return states, dict(attempted=False, accepted=False, reason='direct_polyline')
        seed = refine_seed(_shortcut(corners, proposal), 8)
        low = np.broadcast_to(fine.low/fine.scale, seed.shape).copy()
        high = np.broadcast_to(fine.high/fine.scale, seed.shape).copy()
        constant = np.flatnonzero(np.all(seed == seed[0], axis=0))
        low[:, constant] = high[:, constant] = seed[0, constant]
        oracle = SourceProximity(fine, mesh_bounds=mesh_bounds)
        result = optimize(seed, low, high, oracle.oracle,
                          Options(max_iterations=6, feasibility_callback=oracle.sparse_feasibility))
        candidate = result.trajectory
        fine_ok = all(fine.edge_scaled(a, b) for a, b in zip(candidate, candidate[1:]))
        dense = _dense(candidate, fine, interpolate)
        dense[[0, -1]] = states[[0, -1]]
        accepted, reason = accept_candidate(states, dense, engine, fine_ok)
        stats.update(accepted=accepted, reason=reason, status=result.status,
                     objective_before=result.initial_objective, objective_after=result.objective,
                     duration_before=measured_duration(states, engine), duration_after=measured_duration(dense, engine),
                     length_before=path_length(states), length_after=path_length(dense))
        return (dense if accepted else states), stats
    except Exception as exc:
        stats.update(status='fallback', reason=type(exc).__name__+': '+str(exc))
        return states, stats


def connect(engine, a15, b15, baseline_states=None, seconds=60, seed=1,
            mesh_bounds=None, do_trajopt=False):
    """Return physical dense states, preserving inherited baseline rotation frames."""
    a, b = np.asarray(a15, float), np.asarray(b15, float)
    if a.shape != (15,) or b.shape != (15,) or not np.isfinite([a, b]).all():
        raise ValueError('Finite physical 15-axis endpoints required')
    baseline = None if baseline_states is None else np.asarray(baseline_states, float)
    if baseline is not None and (baseline.ndim != 2 or baseline.shape[1] != 15 or len(baseline)<2 or not np.isfinite(baseline).all()):
        raise ValueError('Baseline must be finite physical Nx15 frames')
    # The caller supplies the sealed original NPZ frames, not proposal samples.
    if baseline is not None and np.array_equal(a, baseline[0]) and np.array_equal(b, baseline[-1]):
        stats = dict(strategy='inherited_baseline', inherited_source_validation=True)
        states = baseline.copy()
        if do_trajopt:
            if not np.all(baseline[:, FIXED] == baseline[0, FIXED]):
                stats['trajopt'] = dict(attempted=False, accepted=False, reason='moving_fixed_axes_or_positioner')
            else:
                corners = compress_collinear(baseline)
                if len(corners) <= 2 or len(corners) > 40:
                    stats['trajopt'] = dict(attempted=False, accepted=False,
                        reason='direct_polyline' if len(corners)<=2 else 'too_many_polyline_corners')
                else:
                    SourceBridge, _, interpolate = _dependencies()
                    fine, proposal = SourceBridge(engine), SourceBridge(engine)
                    proposal.resolutions[:12] = np.deg2rad(.5)
                    states, stats['trajopt'] = _trajopt(states, engine, fine, proposal,
                        mesh_bounds, interpolate, seed_path=corners/fine.scale)
        return dict(passed=True, states=states, stats=stats)
    if not np.array_equal(a[6:12], b[6:12]) or a[14] != b[14]:
        return dict(passed=False, states=None, stats=dict(reason='inactive_arm_changed'))
    if a[12] != b[12]:
        return _baseline_adapters(engine, a, b, baseline, seconds, seed, mesh_bounds, do_trajopt)
    SourceBridge, connect_rrt, interpolate = _dependencies()
    deadline = time.perf_counter()+seconds
    class BudgetExpired(RuntimeError):
        pass
    class ProposalBridge(SourceBridge):
        def valid_scaled(self, state):
            if time.perf_counter() >= deadline:
                raise BudgetExpired()
            return super().valid_scaled(state)
    fine, proposal = SourceBridge(engine), ProposalBridge(engine)
    proposal.resolutions[:12] = np.deg2rad(.5)
    x, y = a/fine.scale, b/fine.scale
    stats = dict(inherited_source_validation=False)
    path = None
    # Fine direct is cheap in knot count and does not depend on proposal acceptance.
    if fine.edge_scaled(x, y):
        path = np.array([x, y]); stats['strategy'] = 'direct'
    else:
        try:
            def unpack(q):
                full = x.copy(); full[FREE] = q
                return full
            partial, rrtstats = connect_rrt(x[FREE], y[FREE], (proposal.low/proposal.scale)[FREE],
                (proposal.high/proposal.scale)[FREE], lambda p,q: proposal.edge_scaled(unpack(p), unpack(q)),
                seed=seed, max_iterations=900, step=.012)
            stats['rrt'] = rrtstats
            if partial is not None:
                original = np.array([unpack(q) for q in partial])
                try:
                    shortened = _shortcut(original, proposal)
                except BudgetExpired:
                    shortened = original
                for candidate in (shortened, original):
                    if all(fine.edge_scaled(p,q) for p,q in zip(candidate,candidate[1:])):
                        path = candidate; stats['strategy'] = 'rrt'; break
        except BudgetExpired:
            stats['reason'] = 'proposal_budget_expired'
    stats['fine_checks'] = fine.checks
    if path is None:
        if baseline is not None:
            return _baseline_adapters(engine, a, b, baseline, seconds, seed, mesh_bounds, do_trajopt)
        return dict(passed=False, states=None, stats=stats)
    states = _dense(path, fine, interpolate)
    states[[0,-1]] = [a,b]
    if do_trajopt and len(path)>2:
        optimization_proposal = SourceBridge(engine)
        optimization_proposal.resolutions[:12] = np.deg2rad(.5)
        states, stats['trajopt'] = _trajopt(states, engine, fine, optimization_proposal,
            mesh_bounds, interpolate, seed_path=path)
    elif do_trajopt:
        stats['trajopt'] = dict(attempted=False, accepted=False, reason='direct_polyline')
    return dict(passed=True, states=states, waypoints_scaled=path, stats=stats)


def _baseline_adapters(engine, a, b, baseline, seconds, seed, mesh_bounds, do_trajopt):
    if baseline is None or a[12] != baseline[0,12] or b[12] != baseline[-1,12]:
        return dict(passed=False, states=None, stats=dict(reason='baseline_angle_profile_required'))
    if not np.all(baseline[:,6:12] == a[6:12]) or not np.all(baseline[:,14] == a[14]):
        return dict(passed=False, states=None, stats=dict(reason='baseline_inactive_arm_changed'))
    left = connect(engine,a,baseline[0],seconds=seconds/2,seed=seed,mesh_bounds=mesh_bounds,do_trajopt=do_trajopt)
    if not left['passed']:
        return left
    right = connect(engine,baseline[-1],b,seconds=seconds/2,seed=seed+1,mesh_bounds=mesh_bounds,do_trajopt=do_trajopt)
    if not right['passed']:
        return right
    return dict(passed=True, states=np.vstack([left['states'][:-1],baseline,right['states'][1:]]),
                stats=dict(strategy='inherited_baseline_with_adapters',inherited_source_validation=True,
                           entry=left['stats'],exit=right['stats']))

