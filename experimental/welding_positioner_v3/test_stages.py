import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fast_search import job_indices, summary_counts


class StageTests(unittest.TestCase):
    def rows(self):
        return [dict(relative_gun_candidates=0, passed=False, stationary_passed=False,
                     stationary_done=False, positioner_done=False),
                dict(relative_gun_candidates=5, passed=True, stationary_passed=True,
                     stationary_done=True, positioner_done=False),
                dict(relative_gun_candidates=5, passed=False, stationary_passed=False,
                     stationary_done=True, positioner_done=False)]

    def test_geometry_discarded_points_have_no_ik_jobs_at_all(self):
        rows = self.rows()
        self.assertEqual(job_indices(rows, 'stationary'), [])
        self.assertEqual(job_indices(rows, 'positioner'), [2])

    def test_stationary_barrier_prevents_positioner_jobs(self):
        rows = self.rows(); rows[2]['stationary_done'] = False
        with self.assertRaises(ValueError): job_indices(rows, 'positioner')

    def test_counts_do_not_label_already_passed_points_pending(self):
        rows = self.rows(); rows[0].update(stationary_done=True, positioner_done=True)
        counts = summary_counts(rows)
        self.assertEqual(counts['geometry_discarded'], 1)
        self.assertEqual(counts['geometry_candidates'], 2)
        self.assertEqual(counts['positioner_resolved'], 2)
        self.assertEqual(counts['pending_candidates'], 1)


if __name__ == '__main__': unittest.main()
