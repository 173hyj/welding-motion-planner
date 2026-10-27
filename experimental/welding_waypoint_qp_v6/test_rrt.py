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

    def test_bidirectional_rrt_detours_around_blocked_joint_region(self):
        from rrt import bidirectional_rrt
        class Model:
            bounds=np.array([[-1.]*6,[1.]*6])
            def limits_valid(self,q):return bool(np.all(q>=self.bounds[0]) and np.all(q<=self.bounds[1]))
        def edge_valid(a,b):
            return all(not (-.1<q[0]<.5 and q[1]<.3) for q in edge_samples(a,b,.01))
        start=np.array([-.5,0.,0.,0.,0.,0.]);goal=np.array([.8,0.,0.,0.,0.,0.])
        path,stats=bidirectional_rrt(start,goal,Model(),edge_valid,seed=7,max_iterations=3000,step_rad=.15)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0],start);np.testing.assert_allclose(path[-1],goal)
        self.assertTrue(all(edge_valid(a,b) for a,b in zip(path,path[1:])))


if __name__=='__main__':unittest.main()
