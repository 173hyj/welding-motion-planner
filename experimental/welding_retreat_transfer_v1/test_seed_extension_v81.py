import unittest
from seed_selector_v79 import select_seeds
import seed_extension_v81 as extend


class SeedExtensionTests(unittest.TestCase):
    def test_exact_old_state_excluded_but_new_angle_and_nearby_q_retained(self):
        old = dict(candidate={'robot_id': 'r'}, positioner_deg=70., q_rad=[0.] * 6, rank=[900, 0, 1])
        other_angle = dict(old, positioner_deg=75., rank=[800, 0, 1])
        nearby = dict(old, q_rad=[1e-12] + [0.] * 5, rank=[700, 0, 1])
        chosen = extend.unsearched_seeds([old, other_angle, nearby], [old], 4)
        self.assertEqual(chosen, [other_angle, nearby])

    def test_augmented_opening_domain_preserves_originals_and_adds_finite_gaps(self):
        values = extend.opening_amplitudes()
        self.assertEqual(set(values), {5., 10., 20., 30., 40., 50., 60., 70., 80.})
        self.assertEqual(len(values), len(set(values)))


if __name__ == '__main__':
    unittest.main()
