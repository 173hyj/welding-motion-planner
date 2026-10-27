import math
import unittest

from experimental.welding_candidate_catalog_v1 import core


class CandidateCatalogCoreTests(unittest.TestCase):
    def test_pose_grid_covers_both_robots_signs_and_every_roll(self):
        poses = core.pose_grid(['R1', 'R2'])
        self.assertEqual(len(poses), 288)
        self.assertEqual({(p['robot_id'], p['roll_deg'], p['axis_sign']) for p in poses},
                         {(r, roll, sign) for r in ['R1', 'R2']
                          for roll in range(0, 360, 5) for sign in (1, -1)})

    def test_angle_grid_has_72_unique_physical_angles(self):
        angles = core.angle_grid()
        self.assertEqual(angles, [0] + [v for n in range(5, 180, 5) for v in (n, -n)] + [180])
        self.assertEqual(len(set(a % 360 for a in angles)), 72)

    def test_key_is_stable_and_distinguishes_all_dimensions(self):
        poses = core.pose_grid(['R1', 'R2'])
        keys = [core.candidate_key('W/1', 5, p) for p in poses]
        self.assertEqual(len(set(keys)), 288)
        self.assertEqual(keys[0], core.candidate_key('W/1', 5.0, dict(poses[0])))
        self.assertNotEqual(keys[0], core.candidate_key('W/2', 5, poses[0]))
        self.assertNotEqual(keys[0], core.candidate_key('W/1', -5, poses[0]))

    def test_dedup_compares_raw_actuator_states_preserving_windings(self):
        states = [[0, 0], [0.00001, 0], [2 * math.pi, 0], [0, 0.01]]
        self.assertEqual(core.deduplicate_joint_states(states), [states[0], states[2], states[3]])

    def test_geometry_precedes_ik_and_records_every_pose_after_success(self):
        poses = core.pose_grid(['R1'])[:4]
        events = []

        def geometry(pose):
            events.append(('geometry', pose['roll_deg'], pose['axis_sign']))
            return {'passed': pose != poses[0], 'reason': 'fixture'}

        def solve(pose):
            events.append(('ik', pose['roll_deg'], pose['axis_sign']))
            return ([{'q_rad': [0]}, {'q_rad': [2 * math.pi]}, {'q_rad': [1]}]
                    if pose == poses[1] else []), {'attempts': 3}

        def accept(pose, solution):
            return {'passed': solution['q_rad'][0] != 1, 'reason': 'joint_limit'}

        records = core.enumerate_angle('W1', 10, poses, geometry, solve, accept)
        self.assertEqual(len(records), 4)
        self.assertEqual([r['status'] for r in records],
                         ['geometry_rejected', 'candidates_found', 'ik_not_found', 'ik_not_found'])
        self.assertEqual(len(records[1]['solutions']), 3)
        self.assertEqual(len(records[1]['accepted']), 2)
        self.assertEqual(records[1]['ik_stats'], {'attempts': 3})
        self.assertFalse(records[1]['numerically_complete'])
        self.assertEqual(events[0], ('geometry', 0, 1))
        self.assertNotIn(('ik', 0, 1), events)
        for index, event in enumerate(events):
            if event[0] == 'ik':
                self.assertEqual(events[index - 1], ('geometry', *event[1:]))

    def test_rejected_solutions_have_explicit_status_and_outcomes(self):
        pose = core.pose_grid(['R1'])[:1]
        records = core.enumerate_angle('W1', 0, pose,
            lambda p: {'passed': True},
            lambda p: ([{'q_rad': [1]}], {}),
            lambda p, s: {'passed': False, 'reason': 'collision'})
        self.assertEqual(records[0]['status'], 'no_accepted_ik')
        self.assertEqual(records[0]['solutions'][0]['acceptance']['reason'], 'collision')
        self.assertEqual(records[0]['accepted'], [])


if __name__ == '__main__':
    unittest.main()
