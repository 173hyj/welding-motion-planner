import importlib
import unittest
import numpy as np
from mixed_bridge_v47 import ROBOT_IDS,branch_state_valid


class ExpandedGatewayTests(unittest.TestCase):
    def test_opposite_first_axis_parks_are_generated_as_full_states(self):
        try:m=importlib.import_module('mixed_bridge_v52')
        except ModuleNotFoundError:self.fail('Diagnostic-informed gateway expansion missing')
        a=np.zeros(15);a[5]=-1.;a[11]=1.;a[12]=1.;a[13:]=[-80.,-10.]
        b=a.copy();b[12]=0.
        parks={ROBOT_IDS[0]:[-3.,-.8,.5,0.,0.,-np.pi/2],ROBOT_IDS[1]:[-3.,-.8,.5,0.,0.,np.pi/2]}
        bank=m.gateway_states(a,b,ROBOT_IDS,parks)
        for r1_q1 in (0.,1.,2.):
            self.assertTrue(any(q[0]==-3. and q[6]==r1_q1 and np.array_equal(q[13:],np.zeros(2)) for q in bank))
        self.assertEqual(len(bank),len({tuple(q) for q in bank}))
        for q in bank:
            self.assertTrue(branch_state_valid(q,ROBOT_IDS));self.assertEqual(q[12],a[12])


if __name__=='__main__':unittest.main()
