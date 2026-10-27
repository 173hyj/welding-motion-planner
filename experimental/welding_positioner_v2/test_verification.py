import unittest
from types import SimpleNamespace
import fcl
from verify import audit_layout_state


class VerificationTests(unittest.TestCase):
    def object(self):
        return fcl.CollisionObject(fcl.Box(10, 10, 10))

    def test_product_against_hanging_interface_is_not_exempt(self):
        world = SimpleNamespace(static=[('product', 'product', self.object()),
                                        ('interface', 'interface', self.object())])
        state = audit_layout_state(world, {'product'}, 'moving', {'fixed', 'interface'})
        self.assertEqual(len(state['blocking_pairs']), 1)

    def test_moving_fixture_against_external_wall_is_not_exempt(self):
        world = SimpleNamespace(static=[('moving', 'moving', self.object()),
                                        ('wall', 'wall', self.object())])
        state = audit_layout_state(world, {'moving'}, 'moving', {'fixed', 'interface'})
        self.assertEqual(len(state['blocking_pairs']), 1)

    def test_declared_assembly_pair_is_still_recorded_with_qualification(self):
        world = SimpleNamespace(static=[('moving', 'moving', self.object()),
                                        ('interface', 'interface', self.object())])
        state = audit_layout_state(world, {'moving'}, 'moving', {'fixed', 'interface'})
        self.assertFalse(state['blocking_pairs'])
        self.assertEqual(len(state['uncertified_assembly_interface_pairs']), 1)


if __name__ == '__main__': unittest.main()
