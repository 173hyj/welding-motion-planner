import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SourceGunKinematicsTests(unittest.TestCase):
    def test_source_links_and_closed_pose_are_preserved(self):
        from gun_kinematics import GunKinematics
        for label in ('R1', 'R2'):
            raw = json.loads((ROOT / f'build/gun{label}_detail.json').read_text())
            gun = GunKinematics(raw)
            self.assertEqual(gun.limits_mm, (-185., 5.))
            self.assertEqual(len(gun.joints), 4)
            self.assertEqual(len(gun.link_nodes), 5)
            for transform in gun.transforms(0).values():
                np.testing.assert_allclose(transform, np.eye(4), atol=1e-10)
            with self.assertRaises(ValueError):
                gun.transforms(-186)

    def test_opening_follows_source_dependent_arm(self):
        from gun_kinematics import GunKinematics
        raw = json.loads((ROOT / 'build/gunR1_detail.json').read_text())
        gun = GunKinematics(raw)
        transforms = gun.transforms(-185)
        point = np.array([20., 0., 925., 1.])
        moved = transforms['510'] @ point
        self.assertAlmostEqual(moved[0], -165., places=4)
        self.assertGreater(925. - moved[2], 30.)
        self.assertAlmostEqual(gun.joint_values(-185)[1], np.arcsin(-185/485))

    def test_real_mesh_partition_matches_independent_original_gun(self):
        from scipy.spatial import cKDTree
        from build_scene import flatten
        from gun_kinematics import gun_mesh_parts
        for label, name in [('R1', 'DTMX-0545-ST1-R1'), ('R2', 'DTMX-0568-ST1-R2')]:
            raw = json.loads((ROOT / f'build/gun{label}_detail.json').read_text())
            original = json.loads((ROOT / f'build/{name}.json').read_text())
            model, parts = gun_mesh_parts(raw)
            part_vertices = np.vstack([p[2].vertices for p in parts.values()])
            self.assertEqual(sum(len(p[2].faces) for p in parts.values()), len(flatten(raw).faces))
            original_vertices = flatten(original).vertices
            self.assertLess(cKDTree(original_vertices).query(part_vertices)[0].max(), .001)
            self.assertLess(cKDTree(part_vertices).query(original_vertices)[0].max(), .001)
            self.assertEqual(sum(p[1] == 'electrode' for p in parts.values()), 2)


if __name__ == '__main__':
    unittest.main()
