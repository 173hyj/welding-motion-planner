import json
import sys
import unittest
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from welding_kinematics import RobotKinematics
from refinement import solve_seed_bank, roll_target, refinement_angles, PoseProblem


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        cls.model=RobotKinematics(scene,next(a for a in scene['assets'] if a['kind']=='robot'))

    def test_roll_keeps_position_and_normal_fixed(self):
        pose=self.model.tcp(self.model.q_initial)
        for angle in [0.,.25,35.5,359.75]:
            result=roll_target(pose,angle,1)
            np.testing.assert_allclose(result[:3,3],pose[:3,3],atol=1e-12)
            np.testing.assert_allclose(result[:3,2],pose[:3,2],atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(result[:3,:3]),1.)

    def test_reversed_axis_is_same_line_and_same_position(self):
        pose=self.model.tcp(self.model.q_initial)
        result=roll_target(pose,33.,-1)
        np.testing.assert_allclose(result[:3,2],-pose[:3,2],atol=1e-12)
        np.testing.assert_allclose(result[:3,3],pose[:3,3],atol=1e-12)

    def test_fine_angles_include_wrap_boundary_without_repeating_coarse_angles(self):
        angles=refinement_angles(.25)
        self.assertIn(.25,angles)
        self.assertIn(359.75,angles)
        self.assertEqual(len(angles),1080)
        self.assertFalse(any(abs(x-round(x))<1e-10 for x in angles))

    def test_seed_bank_recovers_actual_reachable_pose_with_raw_joint_limits(self):
        model=self.model
        q=model.q_initial.copy()
        target=model.tcp(q)
        solutions,stats=solve_seed_bank(model,target,[q],random_count=0)
        self.assertTrue(solutions)
        for solution in solutions:
            self.assertTrue(model.limits_valid(np.asarray(solution['q_rad'])))
            self.assertLess(solution['position_error_mm'],.1)
            self.assertLess(solution['rotation_error_rad'],1e-5)

    def test_analytic_jacobian_matches_independent_fk_finite_difference(self):
        model=self.model;q=model.q_initial.copy()+[.1,.02,-.02,.03,.04,.05]
        target=model.tcp(model.q_initial)
        problem=PoseProblem(model,target);actual=problem.evaluate(q)[1].copy()
        def error(q):
            from scipy.spatial.transform import Rotation
            pose=model.tcp(q)
            return np.r_[(pose[:3,3]-target[:3,3])/1000.,Rotation.from_matrix(target[:3,:3].T@pose[:3,:3]).as_rotvec()]
        step=1e-6
        expected=np.column_stack([(error(q+step*np.eye(6)[i])-error(q-step*np.eye(6)[i]))/(2*step) for i in range(6)])
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-6)


if __name__=='__main__':unittest.main()
