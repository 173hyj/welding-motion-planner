import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullFinishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load('finish_helper', 'full_finish.py')
        cls.base = load('finish_base', 'optimize.py')
        cls.graph = load('finish_graph', 'graph.py')

    def test_prebuild_and_exploration_keep_all_nodes_records_unchanged(self):
        wrapped = self.helper.known_graph_builder(self.base.build_graph, 6, 2)
        layers = {1: {'arm': 0}}
        nodes = {1: [dict(id='a', endpoint=np.zeros(7), cost=2.),
                     dict(id='b', endpoint=np.zeros(7), cost=0.)]}
        records = {(0, 0, 'start', 'a'): {'passed': True, 'cost': 1.},
                   (0, 0, 'a', 'end'): {'passed': True, 'cost': 1.}}
        original_keys = set(records)
        args = (layers, nodes, records, [[1], []], [np.zeros(7)] * 2, [np.zeros(7)] * 2, 0)
        for _ in range(5):
            ordered, matrices, start, end, keys = wrapped(*args)
            self.assertEqual(ordered, [1])
            self.assertTrue(np.isfinite(start).all())
            self.assertTrue(np.isfinite(end).all())
            self.assertEqual(self.graph.shortest_path_avoiding([nodes[1]], matrices, start, end)['indices'], [1])
            self.assertEqual(set(records), original_keys)
        ordered, matrices, start, end, keys = wrapped(*args)
        self.assertEqual(len(start), 2)
        self.assertTrue(np.isinf(start[1]) and np.isinf(end[1]))
        self.assertEqual(self.graph.shortest_path_avoiding([nodes[1]], matrices, start, end),
                         {'indices': [0], 'cost': 4.})
        self.assertEqual(set(records), original_keys)
        self.assertEqual(wrapped.first_known_round, {0: 5})
        self.assertIsNone(self.graph.shortest_path_avoiding([nodes[1]], matrices, start, end, [(0,)]))

    def test_each_stage_has_independent_prebuild_and_cutoff(self):
        seen = []
        def build(layers, nodes, records, orders, starts, goals, stage):
            seen.append(dict(records))
            return [1], [], np.zeros(1), np.zeros(1), {(stage, 0, 'start', 'a')}
        wrapped = self.helper.known_graph_builder(build, 2, 1)
        args = ({}, {}, {}, [], [], [])
        for _ in range(3):
            wrapped(*args, 0)
        self.assertEqual(wrapped.first_known_round, {0: 2})
        wrapped(*args, 1)
        self.assertEqual(seen[-1], {})
        self.assertEqual(wrapped.calls, {0: 3, 1: 1})

    def test_requires_at_least_one_exploration_and_one_known_round(self):
        for iterations, known in ((1, 1), (6, 0), (6, 6), (6, -1), (6., 2)):
            with self.subTest(iterations=iterations, known=known), self.assertRaises(ValueError):
                self.helper.known_graph_builder(lambda *a: None, iterations, known)


if __name__ == '__main__':
    unittest.main()
