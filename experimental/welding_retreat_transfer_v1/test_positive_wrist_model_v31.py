import importlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from welding_kinematics import RobotKinematics
from analytic_ik_v2 import solver_for

try:
    domain = importlib.import_module('positive_wrist_model_v31')
except ModuleNotFoundError:
    domain = None


class PositiveWristModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(domain, 'Continuity-oriented source search domain missing')
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        asset = next(a for a in scene['assets'] if a['kind']=='robot')
        self.source = RobotKinematics(scene, asset)
        self.filtered = domain.PositiveWristModel(self.source)
        self.q = np.array([-3., -.8, .5, 0., .4, np.pi/2])

    def test_original_source_bounds_are_preserved(self):
        self.assertLess(self.source.bounds[0, 5], 0.)
        self.assertGreater(self.filtered.bounds[0, 5], 0.)
        self.assertFalse(np.shares_memory(self.filtered.bounds, self.source.bounds))

    def test_search_rejects_negative_q6_that_source_would_allow(self):
        q = self.q.copy(); q[5] = -.5
        self.assertTrue(self.source.limits_valid(q))
        self.assertFalse(self.filtered.limits_valid(q))

    def test_original_source_coupling_remains_required(self):
        q = self.q.copy(); q[1] = -1.; q[2] = -2.
        self.assertFalse(self.source.limits_valid(q))
        self.assertFalse(self.filtered.limits_valid(q))

    def test_source_fk_and_analytic_ik_are_unchanged_within_search_domain(self):
        self.assertTrue(self.filtered.limits_valid(self.q))
        target = self.source.tcp(self.q)
        np.testing.assert_array_equal(self.filtered.tcp(self.q), target)
        solutions = solver_for(self.filtered).solve(target, self.q)
        self.assertTrue(solutions)
        for q in solutions:
            self.assertGreater(q[5], 0.)
            self.assertTrue(self.source.limits_valid(q))
            np.testing.assert_allclose(self.source.tcp(q), target, atol=1e-6, rtol=0.)


if __name__ == '__main__':
    unittest.main()
