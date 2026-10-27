"""Local source-Jacobian IK and actual joint interpolation, without branch jumps."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experimental/welding_endpoints_v1'))
from refinement import PoseProblem
from contracts import nearest_equivalent


def line_error(position, origin, direction):
    delta = np.asarray(position)-np.asarray(origin); direction = np.asarray(direction)
    direction = direction/np.linalg.norm(direction)
    return float(np.linalg.norm(delta-(delta@direction)*direction))


def interpolation_probes(qa, qb, max_joint_step_deg=.25):
    qa, qb = np.asarray(qa), np.asarray(qb)
    count = max(2,int(np.ceil(np.max(abs(qb-qa))/np.deg2rad(max_joint_step_deg))))
    # Even count always includes the midpoint of the actual coordinate edge.
    count += count % 2
    return [qa+(qb-qa)*u for u in np.linspace(0.,1.,count+1)]


def continuous_ik(model, target, previous, max_joint_change_deg=5.):
    previous = np.asarray(previous); problem = PoseProblem(model,target)
    fit = least_squares(lambda q:problem.evaluate(q)[0],
        np.clip(previous,model.bounds[0]+1e-9,model.bounds[1]-1e-9),
        jac=lambda q:problem.evaluate(q)[1],bounds=(model.bounds[0],model.bounds[1]),
        max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    q = fit.x
    if not model.limits_valid(q): return dict(passed=False,reason='source_joint_limits')
    q = nearest_equivalent(model,q,previous)
    error = problem.evaluate(q)[0]
    position_error = float(np.linalg.norm(error[:3])*1000.)
    rotation_error_deg = float(np.degrees(np.linalg.norm(error[3:])))
    jump = float(np.degrees(np.max(abs(q-previous))))
    passed = position_error < .05 and rotation_error_deg < .01 and jump <= max_joint_change_deg
    return dict(passed=passed,q_rad=q.tolist(),position_error_mm=position_error,
        rotation_error_deg=rotation_error_deg,maximum_joint_change_deg=jump,nfev=fit.nfev,
        reason='local_continuous_ik' if passed else 'pose_error_or_joint_jump')
