import unittest
import numpy as np
try:
    import endpoint_folded_gateway_v94 as gateways
except ImportError:
    gateways=None
from mixed_bridge_v52 import ROBOT_IDS,branch_state_valid


class EndpointGatewayTests(unittest.TestCase):
    def test_both_actual_folded_endpoint_parks_are_retained_as_gateway_proposals(self):
        self.assertIsNotNone(gateways,'endpoint folded gateway feature missing')
        parks={ROBOT_IDS[0]:[-3.,-.8,.5,0,0,-np.pi/2],ROBOT_IDS[1]:[-3.,-.8,.5,0,0,np.pi/2]}
        a=np.r_[parks[ROBOT_IDS[0]],[-3.,-np.pi/12,0.,0.,0.,np.pi/2],1.,-80.,0.]
        b=np.r_[[-2.,-.3,.7,.2,.4,-1.],parks[ROBOT_IDS[1]],2.,0.,-90.]
        bank=gateways.endpoint_folded_gateways(a,b,ROBOT_IDS,parks)
        self.assertTrue(any(np.array_equal(q[6:12],a[6:12]) and np.array_equal(q[:6],parks[ROBOT_IDS[0]]) for q in bank))
        self.assertTrue(any(np.array_equal(q[:6],b[:6]) and np.array_equal(q[6:12],a[6:12]) for q in bank))
        self.assertEqual(len(bank),len({tuple(q) for q in bank}))
        self.assertTrue(all(branch_state_valid(q,ROBOT_IDS) and q[12]==a[12] for q in bank))
        np.testing.assert_array_equal(a[6:12],[-3.,-np.pi/12,0.,0.,0.,np.pi/2])


if __name__=='__main__': unittest.main()
