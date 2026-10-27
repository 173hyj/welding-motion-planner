import unittest
import seed_selector_v79 as select


def pose(rid, q, rays):
    return {'candidate': {'robot_id': rid, 'roll_deg': 10, 'axis_sign': 1},
            'trials': [{'q_rad': q, 'directions': rays}]}


class DeferredSeedTests(unittest.TestCase):
    def test_deduplicates_and_preserves_far_trace_and_robot_diversity(self):
        a = {'weld_id': 'w', 'initial': {'positioner_deg': 70.},
             'alternative_pose_trials': [
                pose('R1', [1.] * 6, [{'reason': 'full_retreat_blocked',
                    'last_accepted_travel_mm': 900., 'endpoint_ik_candidates': 1,
                    'direction_world': [1., 0., 0.]}]),
                pose('R1', [1.] * 6, [{'reason': 'full_retreat_blocked',
                    'last_accepted_travel_mm': 910., 'endpoint_ik_candidates': 1,
                    'direction_world': [0., 1., 0.]}]),
                pose('R1', [2.] * 6, [{'endpoint_ik_candidates': 2,
                    'direction_world': [1., 0., 0.]}]),
                pose('R2', [3.] * 6, [{'endpoint_ik_candidates': 1,
                    'direction_world': [1., 0., 0.]}])]}
        seeds = select.select_seeds([('a.json', a)], 'w', 2)
        self.assertEqual(len(seeds), 2)
        self.assertEqual(seeds[0]['q_rad'], [1.] * 6)
        self.assertEqual(seeds[0]['reference'], [0., 1., 0.])
        self.assertEqual(seeds[1]['candidate']['robot_id'], 'R2')
        self.assertEqual(seeds[0]['source_report'], 'a.json')

    def test_old_endpoint_failure_can_still_seed_new_direction_domain(self):
        r = {'weld_id': 'w', 'initial': {'positioner_deg': 25.},
             'alternative_pose_trials': [pose('R1', [0.] * 6,
                [{'reason': 'finite_endpoint_ik_prefilter_rejected',
                  'endpoint_ik_candidates': 0, 'direction_world': [1., 0., 0.]}])]}
        seeds = select.select_seeds([('b.json', r)], 'w', 2)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]['positioner_deg'], 25.)
        self.assertEqual(select.select_seeds([('b.json', r)], 'different', 2), [])


if __name__ == '__main__':
    unittest.main()
