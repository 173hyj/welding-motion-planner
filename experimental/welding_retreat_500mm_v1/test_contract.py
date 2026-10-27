import unittest
from experimental.welding_retreat_500mm_v1.run import frozen_ray

class ContractTest(unittest.TestCase):
    def test_new_distance_preserves_source_direction_and_driver(self):
        ray, driver = frozen_ray({'direction_world':[0.,1.,0.], 'driver_mm':-123., 'length_mm':1000.},500.)
        self.assertEqual(ray,dict(direction_world=[0.,1.,0.],direction_frame='current_world',length_mm=500.))
        self.assertEqual(driver,-123.)
    def test_rejects_forged_or_unapproved_distance(self):
        with self.assertRaises(ValueError): frozen_ray({'direction_world':[0.,2.,0.],'driver_mm':-1.},500.)
        with self.assertRaises(ValueError): frozen_ray({'direction_world':[0.,1.,0.],'driver_mm':-1.},499.)

if __name__=='__main__':unittest.main()
