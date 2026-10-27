import unittest
import numpy as np
from scipy.spatial.transform import Rotation
try:
    from run_arm import resolve_exit_direction
except ImportError:
    resolve_exit_direction=None


class DirectionFrameTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(resolve_exit_direction,'exit direction frame conversion missing')
        self.delta=np.eye(4);self.delta[:3,:3]=Rotation.from_euler('x',65,degrees=True).as_matrix()
        self.local=np.array([0.,0.,1.]);self.world=self.delta[:3,:3]@self.local

    def test_already_rotated_direct_initial_direction_is_not_rotated_twice(self):
        result=resolve_exit_direction(dict(direction_world=self.world.tolist(),direction_frame='current_world'),self.delta)
        np.testing.assert_allclose(result,self.world,atol=1e-12)

    def test_zero_angle_geometry_direction_is_rotated_once(self):
        result=resolve_exit_direction(dict(direction_world=self.local.tolist()),self.delta)
        np.testing.assert_allclose(result,self.world,atol=1e-12)

    def test_unknown_coordinate_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_exit_direction(dict(direction_world=[1,0,0],direction_frame='unknown'),self.delta)


if __name__=='__main__':unittest.main()
