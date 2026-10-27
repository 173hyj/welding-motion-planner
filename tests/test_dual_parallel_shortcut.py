import unittest
import numpy as np

try:
    from experimental.dual_arm_parallel_v1.shortcut import greedy_shortcut
except ImportError:
    greedy_shortcut = None


def corners(n=5):
    p = np.zeros((n, 15), dtype=np.float64)
    p[:, 0] = np.arange(n)
    p[1:-1, 1] = 2
    return p


class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(greedy_shortcut, 'shortcut helper is missing')

    def test_blocked_preserves_original(self):
        p = corners()
        q, s = greedy_shortcut(p, lambda a, b: False, lambda q: len(q))
        self.assertEqual(q.tobytes(), p.tobytes())
        self.assertEqual(s['accepted'], 0)
        self.assertEqual(s['attempts'], 6)

    def test_success_uses_whole_path_cost_and_certifies(self):
        p = corners()
        seen = []
        q, s = greedy_shortcut(p, lambda a, b: seen.append((a.copy(), b.copy())) or True,
                              lambda q: len(q))
        self.assertEqual(len(q), 2)
        self.assertEqual(s['original_cost'], 5)
        self.assertEqual(s['new_cost'], 2)
        self.assertEqual(s['accepted'], 1)
        np.testing.assert_array_equal(seen[0][0], p[0])
        np.testing.assert_array_equal(seen[0][1], p[-1])

    def test_endpoint_bytes_and_no_mutation(self):
        p = corners().astype(np.float32)
        p[0, 3] = -0.0
        before = p.tobytes()
        q, _ = greedy_shortcut(p, lambda a, b: True, lambda q: len(q))
        self.assertEqual(q[0].tobytes(), p[0].tobytes())
        self.assertEqual(q[-1].tobytes(), p[-1].tobytes())
        self.assertEqual(p.tobytes(), before)

    def test_budget_and_determinism(self):
        p = corners()
        q, s = greedy_shortcut(p, lambda a, b: False, lambda q: len(q), max_trials=2)
        self.assertEqual(s['attempts'], 2)
        self.assertTrue(s['limit_reached'])
        q2, s2 = greedy_shortcut(p, lambda a, b: False, lambda q: len(q), max_trials=2)
        self.assertEqual(q.tobytes(), q2.tobytes())
        self.assertEqual(s, s2)

    def test_shorter_euclidean_but_longer_display_rejected(self):
        p = corners()
        q, s = greedy_shortcut(p, lambda a, b: True, lambda q: 100 - len(q))
        self.assertEqual(q.tobytes(), p.tobytes())
        self.assertEqual(s['accepted'], 0)

    def test_callback_failures_are_skipped(self):
        p = corners()
        def fail(a, b):
            raise RuntimeError('blocked')
        q, s = greedy_shortcut(p, fail, lambda q: len(q))
        self.assertEqual(q.tobytes(), p.tobytes())
        self.assertEqual(s['attempts'], 6)
        q, s = greedy_shortcut(p, lambda a, b: True,
                              lambda q: len(q) if len(q) == 5 else float('nan'))
        self.assertEqual(s['accepted'], 0)

    def test_invalid_inputs(self):
        bad = corners(); bad[2, 4] = np.nan
        for p in (bad, np.zeros((2, 14)), np.zeros((1, 15))):
            with self.assertRaises(ValueError):
                greedy_shortcut(p, lambda a, b: True, lambda q: len(q))
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                greedy_shortcut(corners(), lambda a, b: True, lambda q: len(q), budget)


if __name__ == '__main__':
    unittest.main()
