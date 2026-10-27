import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np

CORE = Path(__file__).resolve().parents[1] / 'core.py'


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('candidate_core', CORE)
        cls.core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.core)

    def test_global_path_beats_greedy(self):
        result = self.core.shortest_path([['s'], ['a', 'b'], ['t']],
                                         [np.array([[1., 2.]]), np.array([[100.], [1.]])])
        self.assertEqual(result, {'indices': [0, 1, 0], 'cost': 3.})

    def test_forbidden_edges_and_rerun(self):
        layers = [[0], [0, 1], [0]]
        edges = [np.array([[1., 2.]]), np.array([[1.], [1.]])]
        self.assertEqual(self.core.shortest_path(layers, edges)['indices'], [0, 0, 0])
        edges[0][0, 0] = np.inf
        self.assertEqual(self.core.shortest_path(layers, edges)['indices'], [0, 1, 0])
        edges[0][0, 1] = np.inf
        self.assertIsNone(self.core.shortest_path(layers, edges))

    def test_endpoints_single_empty_and_ties(self):
        self.assertEqual(self.core.shortest_path([], []), {'indices': [], 'cost': 0.})
        self.assertIsNone(self.core.shortest_path([[]], []))
        self.assertEqual(self.core.shortest_path([[0, 1]], [], [5., 0.], [0., 2.]),
                         {'indices': [1], 'cost': 2.})
        self.assertEqual(self.core.shortest_path([[0, 1], [0, 1]], [np.zeros((2, 2))]),
                         {'indices': [0, 0], 'cost': 0.})
        self.assertEqual(self.core.shortest_path([[0, 1], [0, 1]], [np.zeros((2, 2))], [9., 0.], [8., 0.]),
                         {'indices': [1, 1], 'cost': 0.})

    def test_cost_validation(self):
        for value in (np.nan, -1., -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.core.shortest_path([[0], [0]], [[[value]]])
        with self.assertRaises(ValueError):
            self.core.shortest_path([[0], [0]], [np.zeros((2, 1))])
        with self.assertRaises(ValueError):
            self.core.shortest_path([[0]], [], [np.nan])

    def test_matches_exhaustive_small_random_graphs(self):
        rng = np.random.default_rng(238)
        for _ in range(50):
            sizes = rng.integers(1, 4, size=4)
            edges = [rng.integers(0, 9, size=(a, b)).astype(float) for a, b in zip(sizes[:-1], sizes[1:])]
            for edge in edges:
                edge[rng.random(edge.shape) < .2] = np.inf
            start = rng.integers(0, 4, size=sizes[0])
            end = rng.integers(0, 4, size=sizes[-1])
            options = [(float(start[p[0]] + end[p[-1]] + sum(e[a, b] for e, a, b in zip(edges, p[:-1], p[1:]))), p)
                       for p in itertools.product(*(range(n) for n in sizes))]
            expected_cost, expected_path = min(options)
            actual = self.core.shortest_path([list(range(n)) for n in sizes], edges, start, end)
            if np.isinf(expected_cost):
                self.assertIsNone(actual)
            else:
                self.assertEqual(actual, {'indices': list(expected_path), 'cost': expected_cost})

    def test_raw_angles_never_wrap(self):
        a, b = np.zeros(15), np.zeros(15)
        a[5], b[5] = np.deg2rad([179., -179.])
        features = self.core.raw_features(a, b)
        self.assertAlmostEqual(features['max_joint_delta_deg'], 358.)
        self.assertAlmostEqual(features['sum_joint_delta_deg'], 358.)
        self.assertAlmostEqual(features['wrist_delta_deg'], 358.)
        self.assertAlmostEqual(features['time_lower_bound_s'], 358. / 30.)

    def test_external_axes_and_speed_lower_bound(self):
        a, b = np.zeros(15), np.zeros(15)
        b[12], b[13], b[14] = np.deg2rad(45.), 80., 320.
        features = self.core.raw_features(a, b)
        self.assertEqual(features['max_joint_delta_deg'], 0.)
        self.assertEqual(features['sum_joint_delta_deg'], 0.)
        self.assertEqual(features['wrist_delta_deg'], 0.)
        self.assertEqual(features['time_lower_bound_s'], 4.)

    def test_configuration_cost_and_validation(self):
        a, b = np.zeros(15), np.zeros(15)
        b[3] = np.deg2rad(180.)
        self.assertAlmostEqual(self.core.configuration_cost(a, b), 1.5 * np.pi + .3)
        self.assertAlmostEqual(self.core.configuration_cost(a, b, {'motion': 0., 'wrist': 0., 'time': 2.}), 12.)
        for bad in (np.zeros(14), np.full(15, np.nan)):
            with self.assertRaises(ValueError):
                self.core.raw_features(a, bad)
        for weights in ({'motion': -1.}, {'time': np.nan}, {'typo': 1.}):
            with self.assertRaises(ValueError):
                self.core.configuration_cost(a, b, weights)

    def test_time_budget_retains_more_expensive_faster_prefix(self):
        layers = [[0], [0, 1], [0]]
        costs = [np.array([[0., 2.]]), np.zeros((2, 1))]
        times = [np.array([[8., 2.]]), np.full((2, 1), 2.)]
        self.assertEqual(self.core.shortest_path(layers, costs)['indices'], [0, 0, 0])
        self.assertEqual(self.core.shortest_path_with_time(layers, costs, times, time_budget=5.),
                         {'indices': [0, 1, 0], 'cost': 2., 'time': 4.})
        self.assertIsNone(self.core.shortest_path_with_time(layers, costs, times, time_budget=3.))

    def test_time_endpoints_exact_boundary_and_lexicographic_tie(self):
        solve = self.core.shortest_path_with_time
        self.assertEqual(solve([], [], [], time_budget=0.), {'indices': [], 'cost': 0., 'time': 0.})
        self.assertIsNone(solve([[]], [], [], time_budget=1.))
        self.assertEqual(solve([[0, 1]], [], [], start_cost=[1., 1.], end_cost=[2., 2.],
                               start_time=[2., 0.], end_time=[1., 0.], time_budget=3.),
                         {'indices': [0], 'cost': 3., 'time': 3.})
        self.assertIsNone(solve([[0]], [], [], start_time=[1. + 1e-12], time_budget=1.))

    def test_time_cost_validation_and_frontier_cap(self):
        solve = self.core.shortest_path_with_time
        for bad in (-1., np.nan, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                solve([[0], [0]], [[[0.]]], [[[bad]]], time_budget=3.)
        with self.assertRaises(ValueError):
            solve([[0]], [], [], time_budget=np.nan)
        with self.assertRaises(ValueError):
            solve([[0]], [], [], max_labels=0)
        with self.assertRaises(RuntimeError):
            solve([[0, 1]], [], [], max_labels=1)
        self.assertIsNone(solve([[0], [0]], [[[0.]]], [[[np.inf]]]))

    def test_time_pareto_matches_exhaustive_random_graphs(self):
        rng = np.random.default_rng(1238)
        for _ in range(60):
            sizes = rng.integers(1, 4, size=int(rng.integers(2, 5)))
            costs = [rng.integers(0, 8, size=(a, b)).astype(float) for a, b in zip(sizes[:-1], sizes[1:])]
            times = [rng.integers(0, 8, size=e.shape).astype(float) for e in costs]
            start_cost, end_cost = rng.integers(0, 3, sizes[0]), rng.integers(0, 3, sizes[-1])
            start_time, end_time = rng.integers(0, 3, sizes[0]), rng.integers(0, 3, sizes[-1])
            budget = float(rng.integers(3, 22))
            options = []
            for p in itertools.product(*(range(n) for n in sizes)):
                c = float(start_cost[p[0]] + end_cost[p[-1]] + sum(e[a, b] for e, a, b in zip(costs, p[:-1], p[1:])))
                t = float(start_time[p[0]] + end_time[p[-1]] + sum(e[a, b] for e, a, b in zip(times, p[:-1], p[1:])))
                if t <= budget:
                    options.append((c, p, t))
            actual = self.core.shortest_path_with_time([list(range(n)) for n in sizes], costs, times,
                    start_cost, end_cost, start_time, end_time, time_budget=budget)
            if not options:
                self.assertIsNone(actual)
            else:
                c, p, t = min(options)
                self.assertEqual(actual, {'indices': list(p), 'cost': c, 'time': t})


if __name__ == '__main__':
    unittest.main()
