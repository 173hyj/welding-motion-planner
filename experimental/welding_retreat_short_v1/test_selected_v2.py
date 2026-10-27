import unittest
from .selected_length_v2 import require_selected_generation
class Selected(unittest.TestCase):
 def test_valid_length_survives_other_rejection(self):
  r=dict(complete=True,source_assets_frozen_before_and_after=True,accepted=False,rows=[dict(length_mm=d,task=t,passed=d==300.) for d in (200.,300.) for t in range(1,22)])
  self.assertEqual(len(require_selected_generation(r,300)),42)
  with self.assertRaises(ValueError):require_selected_generation(r,200)
  r["rows"][-1]=r["rows"][0]
  with self.assertRaises(ValueError):require_selected_generation(r,300)
if __name__=="__main__":unittest.main()
