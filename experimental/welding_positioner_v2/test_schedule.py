import unittest
from schedule import search_angles, remaining_rows


class ScheduleTests(unittest.TestCase):
    def test_stationary_stage_never_changes_positioner(self):
        self.assertEqual(search_angles('stationary'), [0])

    def test_external_stage_has_only_5deg_angles_and_never_retries_zero(self):
        angles = search_angles('positioner')
        self.assertEqual(angles[:4], [5, -5, 10, -10])
        self.assertEqual(len(angles), 71)
        self.assertNotIn(0, angles)
        self.assertTrue(all(a % 5 == 0 for a in angles))

    def test_satisfied_stationary_points_never_enter_external_stage(self):
        rows = [dict(passed=True, relative_gun_candidates=1),
                dict(passed=False, relative_gun_candidates=1),
                dict(passed=False, relative_gun_candidates=0)]
        self.assertEqual(remaining_rows(rows), [1])


if __name__ == '__main__':
    unittest.main()
