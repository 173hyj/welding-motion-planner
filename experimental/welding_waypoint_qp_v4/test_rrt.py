import unittest
import numpy as np
try:
    from rrt import choose_pre_collision, edge_samples, path_length
except ImportError:
    choose_pre_collision=edge_samples=path_length=None


class RRTTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(choose_pre_collision,'RRT waypoint helper missing')
    def test_waypoint_is_before_collision_with_margin(self):
        self.assertEqual(choose_pre_collision(43.2,2.),41.2)
        self.assertEqual(choose_pre_collision(.8,2.),0.)
    def test_joint_edges_are_sampled_and_length_is_reported(self):
        q=np.array([0.,0.]);r=np.array([.4,.2]);samples=edge_samples(q,r,.1)
        self.assertGreater(len(samples),2);np.testing.assert_allclose(samples[0],q);np.testing.assert_allclose(samples[-1],r)
        self.assertAlmostEqual(path_length([q,r]),np.sqrt(.2))


if __name__=='__main__':unittest.main()
