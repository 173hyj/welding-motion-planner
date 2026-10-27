import unittest
from . import run, per_length_fast
class ShortContract(unittest.TestCase):
 def test_distance_contract(self):
  source=dict(direction_world=[1.,0.,0.],driver_mm=12.)
  for d in (200.,300.): self.assertEqual(run.frozen_ray(source,d)[0]["length_mm"],d)
  for d in (199.,500.,1000.):
   with self.assertRaises(ValueError): run.frozen_ray(source,d)
 def test_report_contract(self):
  rows=[dict(length_mm=d,task=t,passed=True) for d in (200.,300.) for t in range(1,22)]
  report=dict(rows=rows,accepted=True,complete=True,source_assets_frozen_before_and_after=True)
  self.assertEqual(len(per_length_fast.require_generation(report)),42)
  report["rows"]=rows[:-1]+[rows[0]]
  with self.assertRaises(ValueError):per_length_fast.require_generation(report)
if __name__=="__main__":unittest.main()
