import importlib
import unittest
import numpy as np


class BranchTransferTests(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('mixed_bridge_v47')
        except ModuleNotFoundError:self.fail('Source-independent mixed transfer domain missing')
        self.ids=['F46171CC-24BD-45DD-82E8-21711C110EE9','F7690943-8A45-40E4-9AD3-F92856F982FA']

    def test_sampling_bounds_shrink_only_two_wrist_axes(self):
        low=np.full(15,-4.);high=np.full(15,4.)
        a,b=self.module.branch_bounds(low,high,self.ids)
        np.testing.assert_array_equal(low,np.full(15,-4.))
        np.testing.assert_array_equal(high,np.full(15,4.))
        self.assertLess(b[5],0.);self.assertGreater(a[11],0.)
        keep=[i for i in range(15) if i not in (5,11)]
        np.testing.assert_array_equal(a[keep],low[keep]);np.testing.assert_array_equal(b[keep],high[keep])

    def test_branch_guard_rejects_invalid_state_and_wrong_identity(self):
        q=np.zeros(15);q[5]=-1.;q[11]=1.
        self.assertTrue(self.module.branch_state_valid(q,self.ids))
        q[5]=1.;self.assertFalse(self.module.branch_state_valid(q,self.ids))
        q[5]=-1.;q[7]=np.nan;self.assertFalse(self.module.branch_state_valid(q,self.ids))
        with self.assertRaises(ValueError):self.module.branch_bounds(np.zeros(15),np.ones(15),['other','r1'])

    def test_gateways_keep_full_state_and_live_external_angle(self):
        a=np.zeros(15);a[5]=-1.;a[11]=1.;a[12]=.4;a[13:]=[-10.,-20.]
        b=a.copy();b[12]=2.;b[13:]=[-60.,0.]
        parks={self.ids[0]:[-3.,-.8,.5,0.,0.,-np.pi/2],self.ids[1]:[-3.,-.8,.5,0.,0.,np.pi/2]}
        bank=self.module.gateway_states(a,b,self.ids,parks)
        self.assertGreater(len(bank),0)
        for q in bank:
            self.assertEqual(q.shape,(15,));self.assertEqual(q[12],a[12])
            self.assertTrue(self.module.branch_state_valid(q,self.ids))
        self.assertEqual(len(bank),len({tuple(q) for q in bank}))
        self.assertTrue(any(np.array_equal(q[13:],a[13:]) for q in bank))
        self.assertTrue(any(np.array_equal(q[13:],np.zeros(2)) for q in bank))


if __name__=='__main__':unittest.main()
