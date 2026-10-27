import importlib
import unittest
import numpy as np


class StateSpaceTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('state_space')
        except ModuleNotFoundError:
            self.fail('state_space module has not been implemented')

    def test_samples_include_both_arms_angle_and_openings(self):
        m = self.module()
        a = np.zeros(15)
        b = np.arange(1, 16) * .1
        samples = m.interpolate(a, b, np.full(15, .02))
        np.testing.assert_allclose(samples[0], a)
        np.testing.assert_allclose(samples[-1], b)
        self.assertLessEqual(np.max(abs(np.diff(samples, axis=0))), .0200000001)
        np.testing.assert_allclose(samples[len(samples)//2], (a+b)/2)

    def test_no_wrap_across_source_range(self):
        m = self.module()
        samples = m.interpolate(np.array([-3.]), np.array([3.]), np.array([.1]))
        self.assertAlmostEqual(samples[len(samples)//2][0], 0)
        self.assertTrue(all(-3 <= x[0] <= 3 for x in samples))

    def test_edge_midpoint_collision_rejected(self):
        m = self.module()
        self.assertFalse(m.edge_is_valid(np.array([0.]), np.array([1.]), np.array([.05]),
                                        lambda q: not .45 <= q[0] <= .55))

    def test_bidirectional_rrt_avoids_obstacle(self):
        m = self.module()
        start, end = np.array([-.8, 0.]), np.array([.8, 0.])
        def valid(q):
            return not (-.2 <= q[0] <= .2 and -.3 <= q[1] <= .3)
        def edge(a, b):
            return m.edge_is_valid(a, b, np.full(2, .02), valid)
        path, stats = m.connect_rrt(start, end, np.array([-1., -1.]), np.array([1., 1.]),
                                   edge, seed=4, max_iterations=1000, step=.15)
        self.assertIsNotNone(path, stats)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], end)
        self.assertTrue(all(edge(a, b) for a, b in zip(path, path[1:])))

    def test_invalid_endpoint_not_silently_changed(self):
        m = self.module()
        path, _ = m.connect_rrt(np.array([-2.]), np.array([0.]), np.array([-1.]),
                               np.array([1.]), lambda a,b: True)
        self.assertIsNone(path)


if __name__ == '__main__':
    unittest.main()
