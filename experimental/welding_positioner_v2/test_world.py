import unittest
import numpy as np
from world import PositionerWorld


class FakeObject:
    def setTransform(self, transform):
        self.transform = transform


class FakePositioner:
    base = np.eye(4)
    base[:3, 3] = [15000, 0, -10000]
    attached_product_ids = ['product']
    def delta(self, angle):
        delta = np.eye(4)
        delta[:3, 3] = [0, angle, 0]
        return delta


class WorldTests(unittest.TestCase):
    def test_world_baked_fixture_receives_only_rotary_delta(self):
        world = PositionerWorld.__new__(PositionerWorld)
        world.positioner = FakePositioner()
        world.moving_id = 'moving'
        world.objects = {'moving': FakeObject(), 'product': FakeObject()}
        delta = world.set_angle(0)
        np.testing.assert_allclose(world.objects['moving'].transform.getTranslation(), delta[:3, 3])
        np.testing.assert_allclose(world.objects['product'].transform.getTranslation(), delta[:3, 3])


if __name__ == '__main__': unittest.main()
