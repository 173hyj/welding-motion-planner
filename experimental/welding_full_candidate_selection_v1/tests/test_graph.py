import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('full_candidate_graph', Path(__file__).resolve().parents[1] / 'graph.py')
        cls.graph = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.graph)

    def test_banning_complete_sequence_keeps_its_nodes_available(self):
        layers = [[0, 1]] * 3
        edges = [np.array([[0., 2.], [2., 0.]]), np.array([[0., 1.], [1., 0.]])]
        self.assertEqual(self.graph.shortest_path_avoiding(layers, edges, forbidden_paths=[(0, 0, 0)]),
                         {'indices': [1, 1, 1], 'cost': 0.})
        self.assertEqual(self.graph.shortest_path_avoiding(layers, edges, forbidden_paths=[(0, 0, 0), (1, 1, 1)]),
                         {'indices': [0, 0, 1], 'cost': 1.})

    def test_no_topk_pruning(self):
        self.assertEqual(self.graph.shortest_path_avoiding([list(range(784)), [0]],
            [np.arange(784, dtype=float).reshape(-1, 1)], forbidden_paths=[(i, 0) for i in range(783)]),
            {'indices': [783, 0], 'cost': 783.})

    def test_endpoints_forbidden_edges_empty_and_exact_ties(self):
        solve = self.graph.shortest_path_avoiding
        self.assertEqual(solve([], []), {'indices': [], 'cost': 0.})
        self.assertIsNone(solve([], [], forbidden_paths=[()]))
        self.assertIsNone(solve([[]], []))
        self.assertEqual(solve([[0, 1]], [], [3., 1.], [0., 2.]), {'indices': [0], 'cost': 3.})
        self.assertEqual(solve([[0, 1]], [], [3., 1.], [0., 2.], [(0,)]), {'indices': [1], 'cost': 3.})
        self.assertIsNone(solve([[0], [0]], [[[np.inf]]]))
        self.assertIsNone(solve([[0]], [], forbidden_paths=[(0,)]))

    def test_validation(self):
        solve = self.graph.shortest_path_avoiding
        for bad in (-1., np.nan, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                solve([[0], [0]], [[[bad]]])
        with self.assertRaises(ValueError):
            solve([[0], [0]], [np.zeros((2, 1))])
        for path in ((0, 0), (1,), (.0,), (-1,)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                solve([[0]], [], forbidden_paths=[path])

    def test_random_graphs_against_exhaustive_forbidden_enumeration(self):
        rng = np.random.default_rng(9375)
        solve = self.graph.shortest_path_avoiding
        for _ in range(100):
            sizes = rng.integers(1, 4, size=int(rng.integers(1, 5)))
            layers = [list(range(n)) for n in sizes]
            edges = [rng.integers(0, 6, (a, b)).astype(float) for a, b in zip(sizes[:-1], sizes[1:])]
            for e in edges:
                e[rng.random(e.shape) < .1] = np.inf
            start, end = rng.integers(0, 4, sizes[0]), rng.integers(0, 4, sizes[-1])
            paths = list(itertools.product(*(range(n) for n in sizes)))
            forbidden = [p for p in paths if rng.random() < .5]
            expected = [(float(start[p[0]] + end[p[-1]] + sum(e[a, b] for e, a, b in zip(edges, p[:-1], p[1:]))), p)
                        for p in paths if p not in forbidden]
            expected = [item for item in expected if np.isfinite(item[0])]
            actual = solve(layers, edges, start, end, forbidden)
            if expected:
                cost, path = min(expected)
                self.assertEqual(actual, {'indices': list(path), 'cost': cost})
            else:
                self.assertIsNone(actual)


if __name__ == '__main__':
    unittest.main()
