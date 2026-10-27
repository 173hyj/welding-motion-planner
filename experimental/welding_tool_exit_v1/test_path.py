import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from welding_kinematics import RobotKinematics
try:
    from arm_path import continuous_ik, line_error, interpolation_probes
except ModuleNotFoundError:
    continuous_ik = line_error = interpolation_probes = None


class PathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        cls.model = RobotKinematics(scene,next(a for a in scene['assets'] if a['kind']=='robot'))

    def setUp(self):
        self.assertIsNotNone(continuous_ik,'continuous tool-path IK not implemented')

    def test_small_motion_keeps_same_joint_branch(self):
        q = self.model.q_initial.copy(); q[4] += .03
        goal = self.model.tcp(q); result = continuous_ik(self.model,goal,self.model.q_initial)
        self.assertTrue(result['passed'])
        self.assertLess(np.max(abs(np.asarray(result['q_rad'])-self.model.q_initial)),.04)

    def test_unreachable_motion_does_not_return_valid_path(self):
        goal = self.model.tcp(self.model.q_initial); goal[:3,3] += 100000.
        self.assertFalse(continuous_ik(self.model,goal,self.model.q_initial)['passed'])

    def test_line_error_detects_lateral_deviation(self):
        self.assertAlmostEqual(line_error([3,4,0],[0,0,0],[1,0,0]),4.)

    def test_interpolation_is_actual_bounded_joint_motion_not_wrapped_shortcut(self):
        qa = np.zeros(6); qb = np.zeros(6); qa[5]=3.13; qb[5]=-3.13
        probes = interpolation_probes(qa,qb,max_joint_step_deg=1.)
        self.assertTrue(any(abs(q[5])<.01 for q in probes))
        self.assertGreater(len(probes),300)


if __name__ == '__main__': unittest.main()
