import itertools
import unittest
import numpy as np
from exact_ordering import held_karp_open


class ExactOrderingTests(unittest.TestCase):
    def test_directed_open_order_matches_brute_force(self):
        costs = np.random.default_rng(5).uniform(1., 20., (6, 6))
        np.fill_diagonal(costs, np.inf)
        result = held_karp_open(costs)
        self.assertIsNotNone(result)
        value = lambda order: sum(costs[a, b] for a, b in zip(order, order[1:]))
        self.assertEqual(sorted(result), list(range(6)))
        self.assertAlmostEqual(value(result), min(value(p) for p in itertools.permutations(range(6))))

    def test_no_route_and_single_node(self):
        self.assertIsNone(held_karp_open(np.full((3, 3), np.inf)))
        self.assertEqual(held_karp_open(np.array([[np.inf]])), [0])
        self.assertEqual(held_karp_open(np.empty((0, 0))), [])

    def test_forbidden_edge_and_expensive_unplanned_edge(self):
        costs = np.full((4, 4), 50.)
        np.fill_diagonal(costs, np.inf)
        costs[0, 1] = costs[1, 2] = costs[2, 3] = 1.
        costs[1, 0] = np.inf
        self.assertEqual(held_karp_open(costs), [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
