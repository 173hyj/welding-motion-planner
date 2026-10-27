import unittest
import numpy as np
from shared_gateway import connect_shared


class ObstacleBridge:
    """Synthetic cell: a closed gun collides when arm 0 is near x=1."""
    def __init__(self):
        self.scale = np.ones(15)
        self.low = np.full(15, -4.)
        self.high = np.full(15, 4.)
        self.accepted = []

    def edge_scaled(self, a, b):
        for t in np.linspace(0., 1., 31):
            s = a+(b-a)*t
            if abs(s[0]-1.) < .1 and abs(s[13]) < .2:
                return False
        self.accepted.append((a.copy(), b.copy()))
        return True


class SharedGatewayTests(unittest.TestCase):
    def test_open_gun_moves_to_gateway_without_forced_closure(self):
        bridge = ObstacleBridge()
        a = np.zeros(15); a[0] = 1.; a[13] = -.3
        b = np.zeros(15); b[0] = -1.; b[6] = 2.; b[12] = .4; b[14] = -.5
        gateway = np.zeros(15); gateway[0] = 2.; gateway[6] = -2.; gateway[13:] = -.8
        path, trials = connect_shared(bridge, a, b, gateway, (0, 1), 1, 10)
        self.assertIsNotNone(path)
        np.testing.assert_array_equal(path[0], a)
        np.testing.assert_array_equal(path[-1], b)
        self.assertTrue(any(np.array_equal(p, gateway) for p in path))
        for x, y in zip(path, path[1:]):
            changed = set(np.flatnonzero(x != y))
            self.assertTrue(changed <= set(range(6)) | {13} or
                            changed <= set(range(6, 12)) | {14} or changed <= {12})

    def test_reverse_arm_order_remains_continuous(self):
        bridge = ObstacleBridge()
        a = np.zeros(15); a[13:] = -.5
        b = np.full(15, .2); b[13:] = -.7
        gateway = np.full(15, -.2); gateway[12] = a[12]
        path, _ = connect_shared(bridge, a, b, gateway, (1, 0), 2, 10)
        self.assertIsNotNone(path)
        np.testing.assert_array_equal(path[0], a)
        np.testing.assert_array_equal(path[-1], b)
        for x, y in zip(path, path[1:]):
            self.assertTrue(any(np.array_equal(x, u) and np.array_equal(y, v)
                                for u, v in bridge.accepted))


if __name__ == '__main__':
    unittest.main()
