import importlib
import unittest
import numpy as np


class BridgeTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('world_bridge')
        except ModuleNotFoundError:
            self.fail('world_bridge module has not been implemented')

    def test_node_preserves_other_arm_and_both_drivers(self):
        m = self.module()
        node = dict(robot_id='b', other_robot_id='a', other_q_rad=list(range(6)),
                    q_end_rad=list(range(10, 16)), driver_mm=-60., positioner_deg=40.)
        state = m.node_state(node, ['a', 'b'])
        np.testing.assert_allclose(state[:6], range(6))
        np.testing.assert_allclose(state[6:12], range(10, 16))
        self.assertAlmostEqual(state[12], np.deg2rad(40.))
        np.testing.assert_allclose(state[13:], [0., -60.])

    def test_wrong_robot_identity_rejected(self):
        m = self.module()
        node = dict(robot_id='b', other_robot_id='b', other_q_rad=[0.]*6,
                    q_end_rad=[0.]*6, driver_mm=-60., positioner_deg=0.)
        with self.assertRaises(ValueError):
            m.node_state(node, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
