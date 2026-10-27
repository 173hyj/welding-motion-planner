import unittest
import numpy as np
import probe_near_retreat_v72 as probe


class NearRetreatTests(unittest.TestCase):
    def test_only_failed_near_endpoint_is_a_seed(self):
        archive = {'weld_id': 'w', 'alternative_pose_trials': [
            {'candidate': {'robot_id': 'r', 'roll_deg': 355, 'axis_sign': 1},
             'trials': [{'q_rad': [0.] * 6, 'directions': [
                 {'reason': 'passed', 'last_accepted_travel_mm': 2000.},
                 {'reason': 'full_retreat_blocked', 'last_accepted_travel_mm': 400.},
                 {'reason': 'full_retreat_blocked', 'last_accepted_travel_mm': 1917.,
                  'direction_world': [1., 0., 0.], 'amplitude_mm': 60.}]}]}]}
        seed = probe.near_seed(archive, 1900.)
        self.assertEqual(seed['ray']['last_accepted_travel_mm'], 1917.)
        self.assertEqual(seed['q_start_rad'], [0.] * 6)
        self.assertEqual(seed['candidate']['roll_deg'], 355)
        with self.assertRaises(ValueError):
            probe.near_seed(archive, 1950.)

    def test_release_filter_precedes_budget_and_grid_is_five_degrees(self):
        start = np.eye(4)
        rays = probe.release_neighbors(start, np.array([1., 0., 0.]),
            [{'outward_normal': [0., 1., 0.]}, {'outward_normal': [0., 0., 1.]}], 64)
        self.assertEqual(len(rays), 64)
        self.assertTrue(np.all(rays[:, 1] > 1e-6))
        self.assertTrue(np.all(rays[:, 2] > 1e-6))
        self.assertTrue(np.allclose(np.linalg.norm(rays, axis=1), 1.))
        elevations = np.degrees(np.arcsin(rays[:, 2]))
        azimuths = np.degrees(np.arctan2(rays[:, 1], rays[:, 0]))
        self.assertTrue(np.allclose(elevations / 5., np.round(elevations / 5.)))
        self.assertTrue(np.allclose(azimuths / 5., np.round(azimuths / 5.)))
        self.assertTrue(np.all(np.diff(rays[:, 0]) <= 1e-12))


if __name__ == '__main__':
    unittest.main()
