"""Lazy full-budget IK: callers certify each solution before requesting another."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from refinement import PoseProblem


def legal_sobol(model, count):
    if not count:
        return []
    unit = qmc.Sobol(6, scramble=True, seed=20261008).random_base2(int(np.ceil(np.log2(max(128, count*4)))))
    random = model.bounds[0]+unit*(model.bounds[1]-model.bounds[0])
    return [q for q in random if model.limits_valid(q)][:count]


def nearest_seeds(states, angle, limit=4):
    ranked = sorted(states, key=lambda state: abs(state[0]-angle))
    return [np.asarray(q) for _, q in ranked[:limit]]


def iter_solutions(model, target, seeds, random_count=16, stats=None, max_nfev=220):
    stats = stats if stats is not None else {}
    stats.update(starts=0, converged_pose_count=0, illegal_solutions=0, best_residual=float('inf'),
                 nfev=0, solve_s=0.)
    problem = PoseProblem(model, target)
    starts = [np.asarray(q, float) for q in seeds]+[model.q_initial, np.zeros(6)]+legal_sobol(model, random_count)
    starts = list({tuple(np.round(q, 7)): q for q in starts}.values())
    seen = []
    for seed in starts:
        stats['starts'] += 1
        started = time.perf_counter()
        fit = least_squares(lambda q: problem.evaluate(q)[0],
            np.clip(seed, model.bounds[0]+1e-8, model.bounds[1]-1e-8),
            jac=lambda q: problem.evaluate(q)[1], bounds=(model.bounds[0], model.bounds[1]),
            max_nfev=max_nfev, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        error = problem.evaluate(fit.x)[0]
        stats['solve_s'] += time.perf_counter()-started
        stats['nfev'] += fit.nfev
        stats['best_residual'] = min(stats['best_residual'], float(np.linalg.norm(error)))
        if np.linalg.norm(error[:3])*1000. >= .05 or np.linalg.norm(error[3:]) >= 1e-5:
            continue
        stats['converged_pose_count'] += 1
        if not model.limits_valid(fit.x):
            stats['illegal_solutions'] += 1
            continue
        if any(np.linalg.norm(np.arctan2(np.sin(fit.x-q), np.cos(fit.x-q))) < 1e-4 for q in seen):
            continue
        seen.append(fit.x.copy())
        yield dict(q_rad=fit.x.tolist(), position_error_mm=float(np.linalg.norm(error[:3])*1000.),
                   rotation_error_rad=float(np.linalg.norm(error[3:])))
