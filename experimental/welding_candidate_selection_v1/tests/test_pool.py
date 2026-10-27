import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('candidate_pool', Path(__file__).parents[1] / 'pool.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PoolTests(unittest.TestCase):
    def test_ranking_preserves_pose_and_deduplicates_raw_coordinates(self):
        base = dict(robot_id='a', roll_deg=0, axis_sign=1, q_rad=[0.] * 6)
        rows = [dict(base, q_rad=[6.] * 6), base, dict(base), dict(base, roll_deg=90, q_rad=[2.] * 6)]
        ranked = p.rank_existing(rows, base, [1.] * 6, [1.] * 6)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]['q_rad'], [6.] * 6)
        self.assertEqual(ranked[1]['roll_deg'], 90)

    def test_numeric_refit_is_skipped(self):
        base = dict(robot_id='a', roll_deg=0, axis_sign=1, q_rad=[0.] * 6)
        self.assertEqual(p.rank_existing([dict(base, q_rad=[.00001]*6)], base), [])

    def test_baseline_initial_keeps_other_park(self):
        path = dict(candidate=dict(roll_deg=30, axis_sign=-1), robot_id='a',
                    q_start_rad=[1.] * 6, other_robot_id='b', other_q_rad=[2.] * 6,
                    positioner_deg=120)
        self.assertEqual(p.path_initial(path)['other_q_rad'], [2.] * 6)

    def test_projection_checks_interpolated_points(self):
        import numpy as np
        p.assert_polyline(np.array([[0., 0.], [.25, .5], [1., 2.]]), np.array([[0., 0.], [1., 2.]]))
        with self.assertRaises(ValueError):
            p.assert_polyline(np.array([[.25, .6]]), np.array([[0., 0.], [1., 2.]]))


if __name__ == '__main__':
    unittest.main()
