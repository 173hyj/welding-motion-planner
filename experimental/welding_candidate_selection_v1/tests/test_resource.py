import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('candidate_resource', Path(__file__).resolve().parents[1] / 'resource.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_weighted_optimum_can_violate_wrist_while_another_is_feasible(self):
        result = self.module.shortest_path_with_resources(
            [[0], [0, 1, 2], [0]], [np.array([[0., 1., 2.]]), np.zeros((3, 1))],
            [np.array([[[1., 1., 11.], [2., 2., 9.], [3., 3., 8.]]]), np.zeros((3, 1, 3))],
            [4., 4., 10.])
        self.assertEqual(result, {'indices': [0, 1, 0], 'cost': 1., 'resources': [2., 2., 9.]})

    def test_strict_limits_endpoints_empty_and_ties(self):
        solve = self.module.shortest_path_with_resources
        self.assertEqual(solve([], [], [], [0., 0.]), {'indices': [], 'cost': 0., 'resources': [0., 0.]})
        self.assertIsNone(solve([[]], [], [], [1.]))
        self.assertEqual(solve([[0, 1]], [], [], [3., 4.], start_cost=[1., 1.], end_cost=[2., 2.],
            start_resources=[[1., 2.], [0., 0.]], end_resources=[[2., 2.], [0., 0.]]),
            {'indices': [0], 'cost': 3., 'resources': [3., 4.]})
        self.assertIsNone(solve([[0]], [], [], [1.], start_resources=[[1. + 1e-12]]))

    def test_preserves_lexicographic_equal_cost_slower_prefix(self):
        solve = self.module.shortest_path_with_resources
        self.assertEqual(solve([[0, 1], [0]], [np.zeros((2, 1))],
            [np.array([[[3.]], [[1.]]])], [4.]),
            {'indices': [0, 0], 'cost': 0., 'resources': [3.]})

    def test_validation_forbidden_and_explicit_cap(self):
        solve = self.module.shortest_path_with_resources
        for bad in (-1., np.nan, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                solve([[0], [0]], [[[0.]]], [[[[bad]]]], [1.])
        self.assertIsNone(solve([[0], [0]], [[[0.]]], [[[[np.inf]]]], [np.inf]))
        with self.assertRaises(ValueError):
            solve([[0]], [], [], [np.nan])
        with self.assertRaises(ValueError):
            solve([[0], [0]], [[[0.]]], [np.zeros((1, 1))], [1.])
        with self.assertRaises(ValueError):
            solve([[0]], [], [], [1.], max_labels=0)
        with self.assertRaises(RuntimeError):
            solve([[0, 1]], [], [], [1.], max_labels=1)

    def test_matches_exhaustive_random_graphs(self):
        rng = np.random.default_rng(88372)
        solve = self.module.shortest_path_with_resources
        for _ in range(100):
            sizes = rng.integers(1, 4, size=int(rng.integers(2, 5)))
            costs = [rng.integers(0, 6, (a, b)).astype(float) for a, b in zip(sizes[:-1], sizes[1:])]
            resources = [rng.integers(0, 7, e.shape + (3,)).astype(float) for e in costs]
            for e in costs:
                e[rng.random(e.shape) < .1] = np.inf
            sc, ec = rng.integers(0, 3, sizes[0]), rng.integers(0, 3, sizes[-1])
            sr, er = rng.integers(0, 3, (sizes[0], 3)), rng.integers(0, 3, (sizes[-1], 3))
            limits = rng.integers(3, 22, 3)
            options = []
            for p in itertools.product(*(range(n) for n in sizes)):
                c = float(sc[p[0]] + ec[p[-1]] + sum(e[a, b] for e, a, b in zip(costs, p[:-1], p[1:])))
                r = sr[p[0]] + er[p[-1]] + sum((e[a, b] for e, a, b in zip(resources, p[:-1], p[1:])), np.zeros(3))
                if np.isfinite(c) and np.all(r <= limits):
                    options.append((c, p, r.tolist()))
            actual = solve([list(range(n)) for n in sizes], costs, resources, limits,
                           sc, ec, sr, er)
            if not options:
                self.assertIsNone(actual)
            else:
                c, p, r = min(options)
                self.assertEqual(actual, {'indices': list(p), 'cost': c, 'resources': r})


if __name__ == '__main__':
    unittest.main()
