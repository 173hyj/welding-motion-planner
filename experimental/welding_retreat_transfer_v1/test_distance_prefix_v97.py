import copy
import unittest
from distance_prefix_v97 import crop_exact_prefix


class DistancePrefixTests(unittest.TestCase):
    def example(self):
        return dict(passed=True, length_mm=2000., fixed_distance_mm=2000.,
                    other_q_rad=[1., 2., 3.], q_start_rad=[0.], q_end_rad=[2.],
                    endpoint_clearance_mm=200., geometry_certificate={'old': True},
                    maximum_sampled_line_error_mm=0.04,
                    frames=[dict(travel_mm=s, q_rad=[s / 1000.], driver_mm=-10.)
                            for s in [0., 500., 1000., 1500., 2000.]])

    def test_exact_endpoint_and_real_park_preserved_without_old_metrics(self):
        old = self.example(); unchanged = copy.deepcopy(old)
        new = crop_exact_prefix(old, 1000.)
        self.assertEqual(old, unchanged)
        self.assertEqual(new['length_mm'], 1000.)
        self.assertEqual(new['fixed_distance_mm'], 1000.)
        self.assertEqual(new['q_end_rad'], [1.])
        self.assertEqual(new['other_q_rad'], old['other_q_rad'])
        self.assertEqual(len(new['frames']), 3)
        for key in ['endpoint_clearance_mm', 'geometry_certificate',
                    'maximum_sampled_line_error_mm']:
            self.assertNotIn(key, new)
        self.assertFalse(new['passed'])

    def test_no_interpolated_or_truncated_short_endpoint(self):
        old = self.example(); old['frames'].pop(2)
        with self.assertRaises(ValueError): crop_exact_prefix(old, 1000.)

    def test_nonmonotonic_path_rejected(self):
        old = self.example(); old['frames'][1]['travel_mm'] = 1200.
        with self.assertRaises(ValueError): crop_exact_prefix(old, 1000.)

    def test_invalid_distance_rejected(self):
        for distance in [0., -1., 2500., float('nan')]:
            with self.assertRaises(ValueError): crop_exact_prefix(self.example(), distance)


if __name__ == '__main__': unittest.main()
