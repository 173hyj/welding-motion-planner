import importlib
import unittest
import numpy as np

try:
    search = importlib.import_module('conservative_search_v30')
except ModuleNotFoundError:
    search = None


class ConservativeSearchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(search, 'Candidate search with full source acceptance is missing')
        self.a = np.zeros(15)
        self.b = self.a.copy(); self.b[0] = .03

    def bridge(self, valid):
        bridge = search.SearchBridge.__new__(search.SearchBridge)
        bridge.scale = np.r_[np.ones(13), 100., 100.]
        bridge.resolutions = np.r_[np.full(12, np.deg2rad(.05)), np.deg2rad(.25), .5, .5]
        bridge.rejected_search_edges = set()
        bridge.valid_scaled = valid
        return bridge

    def test_thin_obstacle_missed_by_coarse_check_is_rejected_before_acceptance(self):
        bridge = self.bridge(lambda q: abs(q[0] - .01) > .0003)
        self.assertTrue(bridge.edge_scaled(self.a, self.b))
        self.assertFalse(bridge.certify_path([self.a, self.b]))

    def test_failed_edge_cannot_be_reproposed_in_either_direction(self):
        bridge = self.bridge(lambda q: abs(q[0] - .01) > .0003)
        self.assertFalse(bridge.certify_path([self.a, self.b]))
        self.assertFalse(bridge.edge_scaled(self.a, self.b))
        self.assertFalse(bridge.edge_scaled(self.b, self.a))

    def test_successful_path_is_checked_at_original_fine_resolution(self):
        checked = []
        bridge = self.bridge(lambda q: checked.append(q.copy()) is None)
        self.assertTrue(bridge.certify_path([self.a, self.b]))
        unique = np.unique([p[0] for p in checked])
        self.assertGreater(len(unique), 30)
        self.assertLessEqual(np.max(np.diff(unique)), np.deg2rad(.05))
        self.assertEqual(unique[0], self.a[0])
        self.assertEqual(unique[-1], self.b[0])

    def test_empty_candidate_is_not_certified(self):
        bridge = self.bridge(lambda q: True)
        self.assertFalse(bridge.certify_path([]))

    def test_rejected_proposal_is_retried_then_only_certified_detour_is_returned(self):
        self.assertTrue(hasattr(search, 'certified_search'), 'Certified candidate retry is missing')
        bridge = self.bridge(lambda q: abs(q[0] - .01) > .0003 or abs(q[1]) > .01)
        offset = np.zeros(15); offset[1] = .03
        proposals = [[self.a, self.b], [self.a, self.a + offset, self.b + offset, self.b]]
        path, trials = search.certified_search(bridge, lambda k: (proposals[k], dict(attempt=k)), attempts=2)
        self.assertEqual(len(trials), 2)
        self.assertFalse(trials[0]['fine_certified'])
        self.assertTrue(trials[1]['fine_certified'])
        self.assertEqual(len(path), 4)
        np.testing.assert_array_equal(path[0], self.a)
        np.testing.assert_array_equal(path[-1], self.b)

    def test_exhausted_fine_rejections_return_no_path(self):
        self.assertTrue(hasattr(search, 'certified_search'), 'Certified candidate retry is missing')
        bridge = self.bridge(lambda q: abs(q[0] - .01) > .0003)
        path, trials = search.certified_search(bridge, lambda k: ([self.a, self.b], {}), attempts=2)
        self.assertIsNone(path)
        self.assertEqual(len(trials), 2)
        self.assertFalse(any(row['fine_certified'] for row in trials))


if __name__ == '__main__':
    unittest.main()
