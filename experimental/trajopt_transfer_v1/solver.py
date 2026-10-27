"""Sequential convex optimization of normalized trajectories.

oracle(x, linearize=True) returns (g, J): g >= 0 is safe, with the
clearance margin already subtracted; J has shape (len(g), x.size).
Rows may change between calls. On nonlinear evaluations J may be None.
The optional callback must return a bool for independently sampled feasibility.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from scipy import sparse


@dataclass
class SolverOptions:
    velocity_weight: float = 1.0
    acceleration_weight: float = 1.0
    penalty: float = 100.0
    max_penalty: float = 1e5
    trust_radius: float = 0.2
    min_trust_radius: float = 1e-5
    max_trust_radius: float = 1.0
    max_iterations: int = 80
    feasibility_tolerance: float = 1e-5
    improvement_tolerance: float = 1e-8
    acceptance_ratio: float = 0.1
    feasibility_callback: Optional[Callable] = None
    progress_callback: Optional[Callable] = None


Options = SolverOptions


def objective(trajectory, options=None):
    """Weighted squared first and second differences in normalized state."""
    opt = options if isinstance(options, SolverOptions) else SolverOptions(**(options or {}))
    x = np.asarray(trajectory, float)
    return float(opt.velocity_weight*np.square(np.diff(x, axis=0)).sum()
                 + opt.acceleration_weight*np.square(np.diff(x, n=2, axis=0)).sum())


@dataclass
class OptimizeResult:
    trajectory: np.ndarray
    success: bool
    feasible: bool
    status: str
    objective: float
    initial_objective: float
    max_violation: float
    history: list


def _qp(H, x, g, J, low, high, penalty):
    """Solve in increments with hinge slacks, preferring OSQP."""
    n, m = x.size, g.size
    P = sparse.block_diag((2 * H, sparse.csc_matrix((m, m))), format='csc')
    q = np.concatenate((2 * H @ x, np.full(m, penalty)))
    A = sparse.vstack((sparse.eye(n + m),
                       sparse.hstack((sparse.csc_matrix(J), sparse.eye(m)))), format='csc')
    l = np.concatenate((low, np.zeros(m), -g))
    u = np.concatenate((high, np.full(m, np.inf), np.full(m, np.inf)))
    import osqp
    problem = osqp.OSQP()
    problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=False,
                  eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, polishing=True)
    result = problem.solve()
    return result.x[:n] if result.info.status_val in (1, 2) else None


def optimize(seed, lower, upper, oracle, options=None):
    """Return the best accepted trajectory; success always requires feasibility."""
    opt = options if isinstance(options, SolverOptions) else SolverOptions(**(options or {}))
    x = np.asarray(seed, dtype=float).copy()
    if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError('seed must be a finite NxD array with N >= 2')
    lo = np.broadcast_to(np.asarray(lower, float), x.shape).copy()
    hi = np.broadcast_to(np.asarray(upper, float), x.shape).copy()
    if np.any(lo > hi) or np.any(x < lo) or np.any(x > hi):
        raise ValueError('seed must satisfy consistent bounds')
    if opt.penalty <= 0 or opt.trust_radius <= 0:
        raise ValueError('penalty and trust radius must be positive')
    N, D = x.shape
    B1 = np.diff(np.eye(N), axis=0)
    B2 = np.diff(np.eye(N), n=2, axis=0)
    H = np.kron(opt.velocity_weight*(B1.T @ B1) + opt.acceleration_weight*(B2.T @ B2), np.eye(D))
    objective = lambda a: float(a.ravel() @ H @ a.ravel())
    def evaluate(a, linearize):
        g, J = oracle(a, linearize=linearize)
        g = np.asarray(g, float).reshape(-1)
        if not np.all(np.isfinite(g)):
            raise ValueError('oracle returned nonfinite clearance')
        if linearize:
            J = J.toarray() if sparse.issparse(J) else np.asarray(J, float)
            if J.shape != (len(g), a.size) or not np.all(np.isfinite(J)):
                raise ValueError('oracle Jacobian has incorrect shape or nonfinite entries')
        return g, J
    initial = objective(x)
    history = []
    radius = opt.trust_radius
    penalty = opt.penalty
    penalty_limit = max(penalty, min(opt.max_penalty, penalty*1000.))
    status = 'max_iterations'
    for iteration in range(opt.max_iterations):
        g, J = evaluate(x, True)
        flat = x.ravel()
        low = np.maximum(lo.ravel()-flat, -radius)
        high = np.minimum(hi.ravel()-flat, radius)
        low[:D] = high[:D] = low[-D:] = high[-D:] = 0
        step = _qp(H, flat, g, J, low, high, penalty)
        if step is None:
            status = 'qp_failed'
            break
        candidate = np.clip(x + step.reshape(x.shape), lo, hi)
        candidate[[0, -1]] = x[[0, -1]]
        step = (candidate-x).ravel()
        candidate_g, _ = evaluate(candidate, False)
        old_merit = objective(x) + penalty*np.maximum(-g, 0).sum()
        model_merit = objective(candidate) + penalty*np.maximum(-(g+J@step), 0).sum()
        new_merit = objective(candidate) + penalty*np.maximum(-candidate_g, 0).sum()
        predicted = float(old_merit-model_merit)
        actual = float(old_merit-new_merit)
        ratio = actual/predicted if predicted > opt.improvement_tolerance else 0.
        callback_ok = opt.feasibility_callback is None or bool(opt.feasibility_callback(candidate))
        accepted = actual > 0 and ratio >= opt.acceptance_ratio and callback_ok
        history.append(dict(iteration=iteration, accepted=accepted, trust_radius=radius,
                            predicted_reduction=predicted, actual_reduction=actual, ratio=ratio,
                            penalty=penalty))
        if opt.progress_callback is not None:
            opt.progress_callback(history[-1].copy())
        if accepted:
            x = candidate
            if ratio > .75:
                radius = min(opt.max_trust_radius, radius*1.5)
        else:
            radius *= .5
        if predicted <= opt.improvement_tolerance and callback_ok:
            current_g = candidate_g if accepted else g
            violation = float(np.maximum(-current_g, 0).max()) if current_g.size else 0.
            if violation > opt.feasibility_tolerance:
                if penalty < penalty_limit:
                    penalty = min(penalty*10., penalty_limit)
                    radius = max(radius, opt.trust_radius)
                    continue
                status = 'penalty_limit'
                break
            status = 'converged'
            break
        if radius < opt.min_trust_radius:
            status = 'trust_region_exhausted'
            break
    final_g, _ = evaluate(x, False)
    violation = float(np.maximum(-final_g, 0).max()) if final_g.size else 0.
    feasible = violation <= opt.feasibility_tolerance and (opt.feasibility_callback is None or bool(opt.feasibility_callback(x)))
    return OptimizeResult(x, bool(feasible), bool(feasible), status, objective(x), initial, violation, history)
