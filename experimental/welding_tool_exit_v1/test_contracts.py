import unittest
from types import SimpleNamespace
import numpy as np
try:
    import contracts as c
except ModuleNotFoundError:
    c = None


class ContractsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(c, 'tool-extraction contracts not implemented')

    def test_hidden_thin_wall_between_clear_endpoints_rejected(self):
        def distance(s):
            return max(0., abs(s - .413) - .004)
        result = c.certify_translation(distance, 0., 1., resolution_mm=.0001)
        self.assertFalse(result['passed'])

    def test_clearance_lipschitz_certifies_clear_interval(self):
        result = c.certify_translation(lambda s: 3., 0., 2.)
        self.assertTrue(result['passed'])

    def test_unresolved_tangent_contact_is_rejected(self):
        result = c.certify_translation(lambda s: .0000001, 0., .1,
                                       resolution_mm=.01)
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'], 'unresolved_clearance')

    def test_separated_electrode_cannot_recontact(self):
        state = c.ContactTracker({'fixed'})
        self.assertTrue(state.advance({'fixed'}, .1))
        self.assertTrue(state.advance(set(), .2))
        self.assertFalse(state.advance({'fixed'}, .3))

    def test_other_electrode_cannot_receive_contact_permission(self):
        state = c.ContactTracker({'fixed'})
        self.assertFalse(state.advance({'moving'}, .1))

    def test_contact_cannot_persist_beyond_release_window(self):
        state = c.ContactTracker({'fixed'}, max_release_mm=5.)
        self.assertFalse(state.advance({'fixed'}, 5.1))

    def test_equivalent_angle_respects_physical_joint_limits(self):
        model = SimpleNamespace(bounds=np.array([[-7.]*6,[7.]*6]),
            limits_valid=lambda q: bool(np.all(q >= -7.) and np.all(q <= 7.)))
        q = np.zeros(6); q[5] = -3.13
        prev = np.zeros(6); prev[5] = 3.13
        equivalent = c.nearest_equivalent(model, q, prev)
        self.assertLess(abs(equivalent[5]-prev[5]), .03)
        restricted = SimpleNamespace(bounds=np.array([[-3.14]*6,[3.14]*6]),
            limits_valid=lambda q: bool(np.all(abs(q) <= 3.14)))
        self.assertAlmostEqual(c.nearest_equivalent(restricted, q, prev)[5], -3.13)

    def test_direction_grid_has_sideways_and_normal_rays(self):
        directions = c.direction_bank()
        self.assertTrue(all(abs(np.linalg.norm(d)-1.)<1e-12 for d in directions))
        self.assertTrue(any(np.linalg.norm(d-[1,0,0])<1e-12 for d in directions))
        self.assertTrue(any(np.linalg.norm(d-[0,0,-1])<1e-12 for d in directions))


if __name__ == '__main__': unittest.main()
