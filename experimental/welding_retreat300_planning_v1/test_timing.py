import unittest
import numpy as np
from .timing import audit_timing
class TimingTests(unittest.TestCase):
 def data(self):
  return dict(states=np.zeros((3,15)),time=np.array([0.,.6,1.]),tcp=np.zeros((3,2,3)),phases=np.array([['weld','idle'],['transfer','idle'],['idle','idle']]),tasks=np.array([[1,0],[1,0],[0,0]]))
 def test_start_phase_dwell(self):
  self.assertTrue(audit_timing(self.data(),{1:0})['passed'])
 def test_tcp_speed_and_short_dwell_rejected(self):
  data=self.data();data['tcp'][2,0,0]=81.
  self.assertFalse(audit_timing(data,{1:0})['passed'])
  data=self.data();data['time'][1]=.5
  self.assertFalse(audit_timing(data,{1:0})['passed'])
if __name__=='__main__':unittest.main()
