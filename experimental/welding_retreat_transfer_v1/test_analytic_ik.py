import copy
import json
import unittest
from pathlib import Path
import numpy as np
from welding_kinematics import RobotKinematics
from analytic_ik import FanucAnalyticIK, analytical_continuous_ik


ROOT = Path(__file__).resolve().parents[2]


class AnalyticIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        cls.assets = [a for a in cls.scene['assets'] if a['kind'] == 'robot']

    def test_recovers_source_fk_of_random_legal_configurations(self):
        rng = np.random.default_rng(20261009)
        for asset in self.assets:
            model = RobotKinematics(self.scene, asset)
            solver = FanucAnalyticIK(model)
            checked = 0
            while checked < 50:
                q = rng.uniform(*model.bounds)
                if not model.limits_valid(q):
                    continue
                target = model.tcp(q)
                solutions = solver.solve(target, q)
                self.assertTrue(solutions, msg=q.tolist())
                self.assertTrue(any(np.max(abs(s-q)) < 1e-5 for s in solutions), msg=q.tolist())
                for s in solutions:
                    self.assertTrue(model.limits_valid(s))
                    self.assertLess(np.max(abs(model.tcp(s)-target)), 1e-6)
                checked += 1

    def test_unreachable_rejected(self):
        model = RobotKinematics(self.scene, self.assets[0])
        target = model.tcp(model.q_initial)
        target[:3, 3] += [100000., 0., 0.]
        self.assertEqual(FanucAnalyticIK(model).solve(target, model.q_initial), [])

    def test_singular_wrist_preserves_reference_q4(self):
        model = RobotKinematics(self.scene, self.assets[0])
        q = np.array([0., -.4, -.2, .7, 0., 1.3])
        self.assertTrue(model.limits_valid(q))
        solutions = FanucAnalyticIK(model).solve(model.tcp(q), q)
        self.assertTrue(any(np.max(abs(s-q)) < 1e-5 for s in solutions))

    def test_modified_source_geometry_rejected(self):
        scene = copy.deepcopy(self.scene)
        scene['joints'][4]['axis1'][2] += 5.
        with self.assertRaises(ValueError):
            FanucAnalyticIK(RobotKinematics(scene, self.assets[0]))

    def test_continuation_tracks_small_motion_and_rejects_branch_jump(self):
        model = RobotKinematics(self.scene, self.assets[0])
        q = np.array([0., -.4, -.2, .7, -.5, 1.3])
        next_q = q.copy(); next_q[0] += .001
        fit = analytical_continuous_ik(model, model.tcp(next_q), q)
        self.assertTrue(fit['passed'])
        self.assertLess(np.max(abs(np.array(fit['q_rad'])-next_q)), 1e-6)
        next_q[0] += .5
        self.assertFalse(analytical_continuous_ik(model, model.tcp(next_q), q)['passed'])


if __name__ == '__main__':
    unittest.main()
