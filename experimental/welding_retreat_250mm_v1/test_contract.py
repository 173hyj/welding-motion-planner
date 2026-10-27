import unittest
from . import run,per_length_fast
class Contract(unittest.TestCase):
 def test_distance(self):
  s=dict(direction_world=[1.,0.,0.],driver_mm=12.)
  self.assertEqual(run.frozen_ray(s,250.)[0]['length_mm'],250.)
  for d in (200.,300.,500.):
   with self.assertRaises(ValueError):run.frozen_ray(s,d)
 def test_receipt(self):
  r=dict(complete=True,accepted=True,source_assets_frozen_before_and_after=True,rows=[dict(length_mm=250.,task=t,passed=True) for t in range(1,22)])
  self.assertEqual(len(per_length_fast.require_generation(r)),21)
  r['rows'][-1]=r['rows'][0]
  with self.assertRaises(ValueError):per_length_fast.require_generation(r)
if __name__=='__main__':unittest.main()
