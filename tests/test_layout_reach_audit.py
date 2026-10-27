import unittest
import numpy as np

from scripts.audit_layout_reach_v17 import chain_radius, choose_robot_anchors
from welding_kinematics import axis_matrix


class LayoutReachAuditTests(unittest.TestCase):
    def test_radius_encloses_composed_arbitrary_rotations(self):
        anchors = np.array([[0., 0., 0.], [1., 0., 2.], [3., 0., 2.]])
        axes = np.eye(3)
        vertices = np.array([[4., 1., 3.], [3., -2., 1.]])
        radius = chain_radius(anchors, vertices)
        rng = np.random.default_rng(42)
        for q in rng.uniform(-20, 20, (100, 3)):
            pose = np.eye(4)
            for point, axis, value in zip(anchors, axes, q):
                pose = pose @ axis_matrix(point, axis, value)
            moved = vertices @ pose[:3, :3].T + pose[:3, 3]
            self.assertLessEqual(np.linalg.norm(moved, axis=1).max(), radius + 1e-12)

    def test_robot_anchors_remain_on_each_source_axis(self):
        import json
        from pathlib import Path
        scene = json.loads((Path(__file__).parents[1] / 'build/scene.json').read_text())
        anchors = choose_robot_anchors(scene['joints'][:6])
        for p, j in zip(anchors, scene['joints'][:6]):
            original = np.array(j['axis1'])
            direction = np.array(j['axis2']) - original
            self.assertLess(np.linalg.norm(np.cross(p - original, direction)) / np.linalg.norm(direction), 1e-8)


if __name__ == '__main__':
    unittest.main()
