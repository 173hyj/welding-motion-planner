import importlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from welding_kinematics import RobotKinematics
from analytic_ik_v2 import solver_for


class SignedWristTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('signed_wrist_model_v40')
        except ModuleNotFoundError:
            self.fail('Per-robot continuous wrist/park domain missing')
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        self.models = {a['id']:RobotKinematics(scene,a) for a in scene['assets'] if a['kind']=='robot'}

    def test_negative_domain_leaves_source_bounds_and_fk_unchanged(self):
        source = next(iter(self.models.values()))
        bounds = source.bounds.copy()
        m = self.module.SignedWristModel(source, -1)
        q = np.array([-3.,-.8,.5,0.,.4,-np.pi/2])
        self.assertTrue(source.limits_valid(q))
        self.assertTrue(m.limits_valid(q))
        np.testing.assert_array_equal(source.bounds,bounds)
        np.testing.assert_array_equal(m.tcp(q),source.tcp(q))
        self.assertLess(m.bounds[1,5],0.)
        solutions = solver_for(m).solve(source.tcp(q),q)
        self.assertTrue(solutions)
        for solution in solutions:
            self.assertLess(solution[5],0.)
            self.assertTrue(source.limits_valid(solution))
            np.testing.assert_allclose(source.tcp(solution),source.tcp(q),atol=1e-6,rtol=0.)

    def test_wrong_sign_and_source_coupling_both_rejected(self):
        source = next(iter(self.models.values()))
        negative = self.module.SignedWristModel(source,-1)
        q = np.array([-3.,-.8,.5,0.,.4,np.pi/2])
        self.assertTrue(source.limits_valid(q))
        self.assertFalse(negative.limits_valid(q))
        q[5] = -np.pi/2; q[1] = -1.; q[2] = -2.
        self.assertFalse(source.limits_valid(q))
        self.assertFalse(negative.limits_valid(q))

    def test_each_other_park_matches_that_robots_own_domain(self):
        ids = sorted(self.models)
        signs = {ids[0]:-1,ids[1]:1}
        parks = self.module.branch_parks(self.models,signs)
        self.assertLess(parks[ids[0]][5],0.)
        self.assertGreater(parks[ids[1]][5],0.)
        for rid,q in parks.items():
            self.assertTrue(self.models[rid].limits_valid(q))
            self.assertTrue(self.module.SignedWristModel(self.models[rid],signs[rid]).limits_valid(q))
        with self.assertRaises(ValueError):
            self.module.branch_parks(self.models,{ids[0]:1})

    def test_invalid_sign_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.SignedWristModel(next(iter(self.models.values())),0)

    def test_path_sign_checks_all_frames_and_exact_inactive_park(self):
        ids = sorted(self.models); signs = {ids[0]:-1,ids[1]:1}
        parks = self.module.branch_parks(self.models,signs)
        q = parks[ids[1]].copy()
        path = dict(robot_id=ids[1],other_robot_id=ids[0],other_q_rad=parks[ids[0]],
                    q_start_rad=q,frames=[{'q_rad':q.copy()} for _ in range(3)])
        self.assertEqual(self.module.path_compatibility(path,signs,parks),(True,True))
        path['frames'][1]['q_rad'][5] *= -1
        self.assertEqual(self.module.path_compatibility(path,signs,parks),(False,True))
        path['frames'][1]['q_rad'][5] *= -1
        path['other_q_rad'] = parks[ids[0]].copy();path['other_q_rad'][5] *= -1
        self.assertEqual(self.module.path_compatibility(path,signs,parks),(True,False))


if __name__ == '__main__':
    unittest.main()
