import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from positioner import SourcePositioner, transform_weld, angle_grid


class PositionerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        cls.positioner=SourcePositioner(cls.scene)

    def test_zero_restores_source_and_source_axis_points_stay_fixed(self):
        model=self.positioner
        np.testing.assert_allclose(model.delta(0),np.eye(4),atol=1e-12)
        for angle in [-180,-45,5,90,180]:
            d=model.delta(angle)
            for p in model.axis_points_world:
                np.testing.assert_allclose(d[:3,:3]@p+d[:3,3],p,atol=1e-8)

    def test_weld_normal_and_position_follow_same_rigid_motion(self):
        model=self.positioner;weld=self.scene['welds'][0]
        moved,pose=transform_weld(self.scene,weld,model.delta(90))
        p=np.asarray(weld['world_xyz']);axis=model.axis_points_world[0];n=model.axis_world
        distance=np.linalg.norm(np.cross(p-axis,n))
        self.assertAlmostEqual(np.linalg.norm(np.cross(np.asarray(moved['world_xyz'])-axis,n)),distance,places=7)
        np.testing.assert_allclose(pose[:3,3],moved['world_xyz'],atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(pose[:3,:3]),1.)

    def test_actual_source_attachment_targets_moving_link(self):
        self.assertEqual(self.positioner.moving_link,'4655')
        self.assertEqual(set(self.positioner.attached_product_ids),{w['product_id'] for w in self.scene['welds']})

    def test_grid_is_exactly_five_degrees_with_no_duplicate_full_turn(self):
        angles=angle_grid()
        self.assertEqual(len(angles),72)
        self.assertEqual(angles[:5],[0,5,-5,10,-10])
        self.assertTrue(all(a%5==0 for a in angles))
        self.assertEqual(len(set(a%360 for a in angles)),len(angles))

    def test_outside_actual_source_limits_rejected(self):
        with self.assertRaises(ValueError):self.positioner.delta(365)


if __name__=='__main__':unittest.main()
