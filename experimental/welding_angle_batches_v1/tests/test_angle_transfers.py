"""Pure order tests; load only the function, avoiding native geometry imports."""
import ast
from pathlib import Path
import unittest
import numpy as np


SOURCE = Path(__file__).absolute().parents[1] / 'transfers.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
FUNCTION = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'order_nodes')
NAMESPACE = {'np': np}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
order_nodes = NAMESPACE['order_nodes']


def state(value, axis=0):
    result = np.zeros(15)
    result[axis] = value
    return result


class TransferOrderTests(unittest.TestCase):
    def test_empty_order(self):
        self.assertEqual(order_nodes([], state(0), state(1)), [])

    def test_single_node_once(self):
        self.assertEqual(order_nodes([state(4)], state(-3), state(12)), [0])

    def test_fixed_endpoints_change_optimal_direction(self):
        states = [state(0), state(10), state(20)]
        self.assertEqual(order_nodes(states, state(30), state(-10)), [2, 1, 0])
        self.assertEqual(order_nodes(states, state(-10), state(30)), [0, 1, 2])

    def test_raw_axes_do_not_wrap_at_pi(self):
        states = [state(np.deg2rad(179)), state(np.deg2rad(-179))]
        self.assertEqual(order_nodes(states, state(np.deg2rad(170)), state(np.deg2rad(-170))), [0, 1])

    def test_positioner_raw_axis_does_not_wrap(self):
        states = [state(np.deg2rad(179), 12), state(np.deg2rad(-179), 12)]
        self.assertEqual(order_nodes(states, state(np.deg2rad(170), 12), state(np.deg2rad(-170), 12)), [0, 1])

    def test_all_21_nodes_exactly_once_and_deterministic(self):
        states = np.random.RandomState(144).normal(size=(21, 15))
        route = order_nodes(states, np.zeros(15), np.ones(15))
        self.assertEqual(sorted(route), list(range(21)))
        self.assertEqual(route, order_nodes(states, np.zeros(15), np.ones(15)))


if __name__ == '__main__':
    unittest.main()
