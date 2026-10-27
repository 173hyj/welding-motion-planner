import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT), str(ROOT/'experimental/welding_endpoints_v1')]
from welding_kinematics import RobotKinematics
from refinement import solve_seed_bank
from fast_ik import iter_solutions, legal_sobol, nearest_seeds


class FastIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        cls.model = RobotKinematics(scene, next(a for a in scene['assets'] if a['kind'] == 'robot'))

    def test_first_valid_seed_yields_before_remaining_initializations(self):
        q = self.model.q_initial
        stats = {}
        generator = iter_solutions(self.model, self.model.tcp(q), [q], stats=stats)
        solution = next(generator)
        self.assertEqual(stats['starts'], 1)
        self.assertTrue(self.model.limits_valid(np.asarray(solution['q_rad'])))
        self.assertLess(solution['position_error_mm'], .05)
        generator.close()
        self.assertEqual(stats['starts'], 1)

    def test_exhausted_stream_preserves_baseline_branch_search(self):
        q = self.model.q_initial + [.1, .02, -.02, .03, .04, .05]
        target = self.model.tcp(q)
        baseline, old_stats = solve_seed_bank(self.model, target, [q], random_count=4)
        stats = {}
        fast = list(iter_solutions(self.model, target, [q], random_count=4, stats=stats))
        self.assertEqual(stats['starts'], old_stats['starts'])
        self.assertEqual(len(fast), len(baseline))
        for state in fast:
            self.assertTrue(any(np.linalg.norm(np.arctan2(np.sin(np.asarray(state['q_rad'])-np.asarray(old['q_rad'])),
                np.cos(np.asarray(state['q_rad'])-np.asarray(old['q_rad'])))) < 1e-4 for old in baseline))

    def test_unreachable_target_never_yields_unverified_state(self):
        target = self.model.tcp(self.model.q_initial)
        target[:3, 3] += 100000
        stats = {}
        self.assertEqual(list(iter_solutions(self.model, target, [], random_count=0, stats=stats)), [])
        self.assertEqual(stats['starts'], 2)

    def test_nearest_external_angle_prioritized_instead_of_last_evaluated(self):
        states = [(20, np.ones(6)), (-20, np.ones(6)*2), (10, np.ones(6)*3)]
        self.assertEqual(nearest_seeds(states, 25, limit=1)[0].tolist(), [1.]*6)
        self.assertEqual(nearest_seeds(states, -25, limit=1)[0].tolist(), [2.]*6)

    def test_sobol_bank_is_legal_deterministic_and_keeps_all_16(self):
        first = legal_sobol(self.model, 16)
        self.assertEqual(len(first), 16)
        np.testing.assert_array_equal(first, legal_sobol(self.model, 16))
        self.assertTrue(all(self.model.limits_valid(q) for q in first))


if __name__ == '__main__': unittest.main()
