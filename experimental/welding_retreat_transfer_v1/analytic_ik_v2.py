"""Source-specific spherical-wrist IK with the original q2/q3 coupling."""
import itertools
import math
import weakref
import numpy as np
from scipy.spatial.transform import Rotation


def representations(angle, low, high):
    lo = math.ceil((low-angle-1e-10)/(2*np.pi))
    hi = math.floor((high-angle+1e-10)/(2*np.pi))
    return [angle+2*np.pi*k for k in range(lo, hi+1)]


class FanucAnalyticIK:
    def __init__(self, model):
        self.model = model
        self.shoulder = np.array([312., 0., 670.])
        self.elbow = np.array([312., 0., 1745.])
        self.center = np.array([1592., 0., 1970.])
        axes = np.array([[0, 0, 1], [0, 1, 0], [0, -1, 0],
                         [-1, 0, 0], [0, -1, 0], [-1, 0, 0]], float)
        points = [np.zeros(3), self.shoulder, self.elbow,
                  self.center, self.center, self.center]
        for i, (joint, expected, point) in enumerate(zip(model.joints[:6], axes, points)):
            p = np.asarray(joint['axis1'], float)
            vector = np.asarray(joint['axis2'], float)-p
            axis = vector/np.linalg.norm(vector)
            if (joint['type'] != 0 or not np.allclose(axis, expected, atol=1e-12, rtol=0.)
                    or np.linalg.norm(np.cross(p-point, expected)) > 1e-8):
                raise ValueError('Unsupported source axis geometry')
            keys = [model.joints[1]['key']] if i == 2 else []
            factors = [1.] if i == 2 else []
            if joint['leadingJoints'] != keys or joint['leadingFactors'] != factors:
                raise ValueError('Unsupported source coupling')
        self.l1 = 1075.
        self.l2 = math.hypot(1280., 225.)
        self.alpha = math.atan2(225., 1280.)
        self.base_inverse = np.linalg.inv(model.base)
        self.tool_inverse = np.linalg.inv(model.tcp_home)

    def solve(self, target, reference=None):
        target = np.asarray(target, float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError('Finite 4x4 target required')
        transform = self.base_inverse@target@self.tool_inverse
        center = (transform@np.r_[self.center, 1.])[:3]
        radius = math.hypot(center[0], center[1])
        azimuth = math.atan2(center[1], center[0])
        low, high = self.model.bounds
        solutions = []
        for q1, radial in ((azimuth, radius), (azimuth+np.pi, -radius)):
            x, z = radial-312., center[2]-670.
            cosine = (x*x+z*z-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
            if abs(cosine) > 1.+1e-10:
                continue
            delta_abs = math.acos(np.clip(cosine, -1., 1.))
            for delta in (delta_abs, -delta_abs):
                angle1 = math.atan2(z, x)-math.atan2(self.l2*math.sin(delta), self.l1+self.l2*math.cos(delta))
                q2 = np.pi/2-angle1
                q3 = angle1+delta-self.alpha
                for arm in itertools.product(*(representations(q, low[i], high[i])
                                               for i, q in enumerate((q1, q2, q3)))):
                    q1a, _, q3a = arm
                    arm_rotation = Rotation.from_euler('z', q1a).as_matrix()@Rotation.from_euler('y', -q3a).as_matrix()
                    wrist = arm_rotation.T@transform[:3, :3]
                    # XYX proper Euler decomposition. Avoid singular warning and
                    # preserve a reference q4 representative at q5=0.
                    b = math.atan2(math.hypot(wrist[1, 0], wrist[2, 0]), wrist[0, 0])
                    if abs(math.sin(b)) < 1e-12:
                        if wrist[0, 0] < 0.:
                            continue  # q5=pi is outside this source's +/-120deg.
                        a = -float(reference[3]) if reference is not None else 0.
                        total = math.atan2(wrist[2, 1], wrist[1, 1])
                        c = total-a
                        branches = [(a, 0., c)]
                    else:
                        a = math.atan2(wrist[1, 0], -wrist[2, 0])
                        c = math.atan2(wrist[0, 1], wrist[0, 2])
                        branches = [(a, b, c), (a+np.pi, -b, c+np.pi)]
                        if abs(math.sin(b)) < 1e-6 and reference is not None:
                            # Both independent Euler angles become ill-conditioned.
                            # A nearby reference representative avoids dividing
                            # roundoff by sin(b); the unchanged FK gate rejects
                            # it when the chosen reference cannot fit the target.
                            ar = -float(reference[3])
                            local = Rotation.from_euler('x', -ar).as_matrix()@wrist
                            br = math.atan2(-local[2, 0], local[0, 0])
                            last = Rotation.from_euler('y', -br).as_matrix()@local
                            cr = math.atan2(last[2, 1], last[1, 1])
                            branches.append((ar, br, cr))
                    for branch in branches:
                        for tail in itertools.product(*(representations(-v, low[i+3], high[i+3])
                                                        for i, v in enumerate(branch))):
                            q = np.r_[arm, tail]
                            if not self.model.limits_valid(q):
                                continue
                            if np.max(abs(self.model.tcp(q)-target)) > 1e-6:
                                continue
                            if not any(np.max(abs(q-s)) < 1e-7 for s in solutions):
                                solutions.append(q)
        if reference is not None:
            reference = np.asarray(reference)
            solutions.sort(key=lambda q: np.linalg.norm(q-reference))
        return solutions


_SOLVERS = weakref.WeakKeyDictionary()


def solver_for(model):
    if model not in _SOLVERS:
        _SOLVERS[model] = FanucAnalyticIK(model)
    return _SOLVERS[model]


def analytical_seed_bank(model, target, seeds, random_count=0):
    reference = np.asarray(seeds[0]) if seeds else None
    candidates = solver_for(model).solve(target, reference)
    solutions = []
    residual = float('inf')
    for q in candidates:
        actual = model.tcp(q)
        position = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
        rotation = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3].T@actual[:3, :3]).as_rotvec()))
        residual = min(residual, math.hypot(position/1000., rotation))
        solutions.append(dict(q_rad=q.tolist(), position_error_mm=position, rotation_error_rad=rotation))
    return solutions, dict(starts=0, converged_pose_count=len(solutions),
                           illegal_solutions=0, best_residual=residual,
                           solver='source_specific_analytic', singular_family_complete=False)


def analytical_continuous_ik(model, target, previous, max_joint_change_deg=5.):
    previous = np.asarray(previous)
    candidates = solver_for(model).solve(target, previous)
    if not candidates:
        return dict(passed=False, reason='source_analytic_branches_rejected')
    q = candidates[0]
    actual = model.tcp(q)
    position = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
    rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(target[:3, :3].T@actual[:3, :3]).as_rotvec())))
    jump = float(np.degrees(np.max(abs(q-previous))))
    return dict(passed=bool(position < .05 and rotation < .01 and jump <= max_joint_change_deg),
                q_rad=q.tolist(), position_error_mm=position, rotation_error_deg=rotation,
                maximum_joint_change_deg=jump, nfev=0,
                reason='source_analytic_continuation' if jump <= max_joint_change_deg else 'pose_error_or_joint_jump')
