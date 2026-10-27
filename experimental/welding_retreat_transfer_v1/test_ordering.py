import importlib
import unittest
import numpy as np


class OrderingTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('ordering')
        except ModuleNotFoundError:
            self.fail('ordering module has not been implemented')

    def test_multistart_visits_every_node_once(self):
        m = self.module()
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0]])
        costs = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        route = m.visit_order(costs)
        self.assertEqual(sorted(route), list(range(5)))
        self.assertLessEqual(m.route_cost(route, costs), 4.1)

    def test_improve_open_route_uncrosses_edges(self):
        m = self.module()
        points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        costs = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        route = m.improve_order([0, 1, 2, 3], costs)
        self.assertEqual(sorted(route), list(range(4)))
        self.assertLess(m.route_cost(route, costs), m.route_cost([0, 1, 2, 3], costs))

    def test_directed_cost_recomputed_after_reversal(self):
        m = self.module()
        costs = np.array([[0, 1, 30, 10], [20, 0, 1, 30], [30, 20, 0, 1], [30, 30, 20, 0]])
        route = m.improve_order([0, 1, 2, 3], costs)
        self.assertEqual(m.route_cost(route, costs), 3)

    def test_forbidden_edge_not_selected(self):
        m = self.module()
        costs = np.array([[0, np.inf, 1], [1, 0, 1], [1, 1, 0]])
        route = m.visit_order(costs)
        self.assertEqual(sorted(route), list(range(3)))
        self.assertTrue(np.isfinite(m.route_cost(route, costs)))

    def test_disconnected_graph_has_no_complete_route(self):
        m = self.module()
        costs = np.array([[0, 1, np.inf], [1, 0, np.inf], [np.inf, np.inf, 0]])
        self.assertIsNone(m.visit_order(costs))

    def test_single_or_empty_route(self):
        m = self.module()
        self.assertEqual(m.visit_order(np.zeros((0, 0))), [])
        self.assertEqual(m.visit_order(np.zeros((1, 1))), [0])


if __name__ == '__main__':
    unittest.main()
