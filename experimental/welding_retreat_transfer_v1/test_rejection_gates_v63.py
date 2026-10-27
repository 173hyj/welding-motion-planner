import importlib
import unittest


class EarlyRejectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('rejection_gates_v63')
        except ModuleNotFoundError:
            self.fail('Reject-only ordered gates are missing')

    def test_unreachable_candidate_never_calls_geometry(self):
        def expensive():
            self.fail('Unreachable ray should not require a geometry certificate')
        result = self.module.filter_ray({}, 'unreachable', lambda: {'passed': False}, expensive)
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'], 'finite_continuation_prefilter_rejected')

    def test_kinematic_positive_cannot_bypass_failed_geometry(self):
        result = self.module.filter_ray({}, 'ray', lambda: {'passed': True}, lambda: {'passed': False})
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'], 'geometry_certificate_failed')

    def test_cache_never_reuses_geometry_between_opening_amplitudes(self):
        cache = {}
        first = self.module.filter_ray(cache, 'same-q-and-ray', lambda: {'passed': True}, lambda: {'passed': False})
        def repeated():
            self.fail('Identical source kinematics should be reused')
        second = self.module.filter_ray(cache, 'same-q-and-ray', repeated, lambda: {'passed': True})
        self.assertFalse(first['passed'])
        self.assertTrue(second['passed'])
        self.assertFalse(first['geometry']['passed'])
        self.assertTrue(second['geometry']['passed'])

    def test_new_order_has_identical_two_gate_acceptance_set(self):
        for kin in (False, True):
            for geom in (False, True):
                with self.subTest(kin=kin, geometry=geom):
                    result = self.module.filter_ray({}, (kin, geom), lambda: {'passed': kin}, lambda: {'passed': geom})
                    self.assertEqual(result['passed'], kin and geom)
                    self.assertNotIn('path_file', result)


if __name__ == '__main__':
    unittest.main()
