import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
try:
    from qp_step import solve_step, tangent_basis, updated_direction, polygon_signed_distance
except ImportError:
    solve_step=tangent_basis=updated_direction=polygon_signed_distance=None


class QPTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(solve_step,'waypoint QP not implemented')

    def test_single_qp_changes_shared_direction_to_resolve_collision(self):
        # Two 1-axis toy waypoints follow the same angular direction u.
        pose=np.array([[1.,0.,-1.],[0.,1.,-2.]])
        collision=np.array([[0.,1.,0.]])
        fit=solve_step(pose,np.zeros(2),collision,np.array([.2]),
                       np.array([-1.,-1.,-.5]),np.array([1.,1.,.5]),n_joint_variables=2)
        self.assertTrue(fit['solved'])
        x=np.asarray(fit['step'])
        self.assertGreater(x[1],.19)
        self.assertLess(np.linalg.norm(pose@x),.002)
        self.assertLess(fit['maximum_collision_slack_mm'],.002)

    def test_direction_is_a_unit_vector_and_line_is_not_bent(self):
        d=np.array([0.,0.,1.]);basis=tangent_basis(d)
        np.testing.assert_allclose(basis.T@d,[0.,0.],atol=1e-12)
        result=updated_direction(d,basis,[.1,.2],.5)
        self.assertAlmostEqual(np.linalg.norm(result),1.)
        p=np.array([3.,4.,5.]);points=[p+s*result for s in [0.,10.,50.,100.]]
        for a,b in zip(points,points[1:]):self.assertLess(np.linalg.norm(np.cross(b-a,result)),1e-12)

    def test_nonconvex_source_polygon_is_not_treated_as_convex_hull(self):
        poly=[(0.,0.),(2.,0.),(2.,1.),(1.,1.),(1.,2.),(0.,2.)]
        self.assertGreater(polygon_signed_distance([.5,.5],poly),0.)
        self.assertLess(polygon_signed_distance([1.5,1.5],poly),0.)

    def test_infeasible_bounds_do_not_produce_an_accepted_step(self):
        with self.assertRaises(ValueError):
            solve_step(np.eye(2),np.zeros(2),np.empty((0,2)),np.zeros(0),np.ones(2),np.zeros(2),n_joint_variables=2)

    def test_fcl_proxy_sign_changes_at_separation(self):
        import fcl
        from proximity import mesh_proxy
        a=fcl.CollisionObject(fcl.Sphere(1.),fcl.Transform(np.eye(3),np.zeros(3)))
        b=fcl.CollisionObject(fcl.Sphere(1.),fcl.Transform(np.eye(3),np.array([1.5,0.,0.])))
        self.assertLess(mesh_proxy(a,b),0.)
        b.setTransform(fcl.Transform(np.eye(3),np.array([3.,0.,0.])))
        self.assertAlmostEqual(mesh_proxy(a,b),1.)


if __name__=='__main__':unittest.main()
