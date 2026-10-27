import unittest
from .run import build_registry, exact_identity, ray_spec

class LibraryTests(unittest.TestCase):
 def test_all_rows_kept_including_unresolved(self):
  rows=[dict(id=str(i),status='search_unresolved' if i%2 else 'certified') for i in range(784)]
  actual=build_registry(rows)
  self.assertEqual(len(actual),784)
  self.assertTrue(all(r['status']=='pending' for r in actual))
 def test_length_propagation(self):
  self.assertEqual(ray_spec([0,0,1])['length_mm'],300.)
 def test_reuse_requires_full_identity(self):
  a=dict(robot_id='a',roll_deg=5,axis_sign=1,q_rad=[0]*6,positioner_deg=30)
  self.assertTrue(exact_identity(a,dict(a)))
  self.assertFalse(exact_identity(a,dict(a,q_rad=[1]*6)))
  self.assertFalse(exact_identity(a,dict(a,positioner_deg=60)))
 def test_fallback_never_sends_one_metre(self):
  from . import search
  import inspect
  code=inspect.getsource(search.try_configuration)
  self.assertNotIn('1000',code)
  self.assertIn('LENGTH_MM',code)

if __name__=='__main__':unittest.main()
