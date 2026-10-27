import unittest
from acceptance import posture_passed


class EndpointAcceptanceTests(unittest.TestCase):
    def state(self):
        return dict(both_source_joint_limits_valid=True, remaining_pairs=[],
                    certificates={'fixed': {'passed': True}, 'moving': {'passed': True}})

    def test_remaining_fixture_collision_never_accepted(self):
        state = self.state()
        state['remaining_pairs'] = [dict(asset_name='fixture')]
        self.assertFalse(posture_passed(state, .01, ['fixed', 'moving'], {}))

    def test_one_electrode_without_contact_rejects_welding(self):
        state = self.state()
        del state['certificates']['moving']
        self.assertFalse(posture_passed(state, .01, ['fixed', 'moving'], {'moving': 1.}))

    def test_wrong_tcp_and_invalid_limits_rejected(self):
        self.assertFalse(posture_passed(self.state(), 1., ['fixed', 'moving'], {}))
        state = self.state()
        state['both_source_joint_limits_valid'] = False
        self.assertFalse(posture_passed(state, .01, ['fixed', 'moving'], {}))

    def test_two_certified_contacts_and_no_unintended_collision_pass(self):
        self.assertTrue(posture_passed(self.state(), .01, ['fixed', 'moving'], {}))

    def test_truncated_certificate_does_not_use_distance_as_escape(self):
        state = self.state()
        state['certificates']['moving'] = {'passed': False}
        self.assertFalse(posture_passed(state, .01, ['fixed', 'moving'], {'moving': 0.}))


if __name__ == '__main__':
    unittest.main()
