import json
from pathlib import Path
import unittest
import numpy as np
from welding_kinematics import RobotKinematics, axis_matrix

ROOT = Path(__file__).resolve().parents[1]


class CounterbalanceFKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT / 'build/scene.json').read_text())
        cls.model = RobotKinematics(cls.scene, next(a for a in cls.scene['assets'] if a['kind'] == 'robot'))

    def test_source_crank_is_equivariant_under_robot_first_axis(self):
        q = self.model.q_initial.copy()
        q[0] = 0.
        baseline = self.model.links(q)
        first = self.scene['joints'][0]
        point, tip = np.array(first['axis1']), np.array(first['axis2'])
        for angle in (.25, .8, 1.5, 3.):
            moved = q.copy()
            moved[0] = angle
            expected_rotation = axis_matrix(point, tip - point, angle)
            actual = self.model.links(moved)
            for key in ('246', '247'):
                np.testing.assert_allclose(actual[key], expected_rotation @ baseline[key], atol=1e-9, rtol=0)

    def test_cylinder_and_rod_share_direction_of_transformed_source_pins(self):
        pin_a = np.array(self.scene['joints'][6]['axis1'])
        pin_b = np.array(self.scene['joints'][8]['axis1'])
        home = pin_a - pin_b
        for q1, q2 in ((0., -.8), (.8, -.8), (1.5, .4), (3., 1.)):
            q = self.model.q_initial.copy()
            q[:2] = q1, q2
            links = self.model.links(q)
            a = links['241'][:3, :3] @ pin_a + links['241'][:3, 3]
            b = links['240'][:3, :3] @ pin_b + links['240'][:3, 3]
            expected = (a - b) / np.linalg.norm(a - b)
            for key in ('246', '247'):
                actual = links[key][:3, :3] @ (home / np.linalg.norm(home))
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
