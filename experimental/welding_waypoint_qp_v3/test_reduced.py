import unittest
import numpy as np
try:
    from reduced import direction_joint_jacobian
except ImportError:
    direction_joint_jacobian=None


class ReducedTests(unittest.TestCase):
    def test_eliminated_joint_step_preserves_shared_line_direction(self):
        self.assertIsNotNone(direction_joint_jacobian,'reduced-QP Jacobian missing')
        J=np.diag([2.,3.,4.,1.,1.,1.]);B=np.array([[1.,0.],[0.,1.],[0.,0.]])
        mapping=direction_joint_jacobian(J,B,20.)
        delta=np.array([.1,.2])
        np.testing.assert_allclose(J@(mapping@delta),np.r_[20.*B@delta,np.zeros(3)],atol=1e-10)


if __name__=='__main__':unittest.main()
