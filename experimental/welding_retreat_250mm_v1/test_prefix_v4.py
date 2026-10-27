import unittest
import numpy as np
from .exact_prefix_v4 import select_subsection
class Prefix(unittest.TestCase):
 def test_ties_choose_short_free_prefix(self):
  old=np.zeros((9,15));old[:,0]=[0,.2,1,1,.8,1,.2,0,0]
  phases=np.full((9,2),'transfer',dtype='<U10');phases[3:6,0]='weld'
  got=select_subsection(old,phases,0,0.,np.zeros(7),np.r_[1.,np.zeros(6)],np.zeros(7))
  self.assertTrue(np.array_equal(got,old[:3]))
 def test_rejects_other_park(self):
  old=np.zeros((2,15));old[1,0]=1.;old[:,6]=.5
  self.assertIsNone(select_subsection(old,np.full((2,2),'transfer'),0,0.,np.zeros(7),np.r_[1.,np.zeros(6)],np.zeros(7)))
if __name__=='__main__':unittest.main()
