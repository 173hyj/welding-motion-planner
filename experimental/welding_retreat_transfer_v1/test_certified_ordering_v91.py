import unittest
import numpy as np
try:
    import certified_ordering_v91 as certified
except ImportError:
    certified=None
from ordering import visit_order


class CertifiedOrderingTests(unittest.TestCase):
    def test_classic_order_prefers_existing_connections_without_hiding_unconnected_node(self):
        self.assertIsNotNone(certified, 'certified ordering feature missing')
        costs=np.array([[np.inf,3.,1.,1.],[3.,np.inf,1.,1.],[1.,1.,np.inf,3.],[1.,1.,3.,np.inf]])
        known={(0,1):{'passed':True,'metric_length':3.},(2,3):{'passed':True,'metric_length':3.}}
        weighted=certified.proposal_costs(costs,known,np.zeros((4,4)))
        order=visit_order(weighted)
        self.assertEqual(sorted(order),list(range(4)))
        edges=list(zip(order,order[1:]))
        self.assertIn((0,1),edges)
        self.assertIn((2,3),edges)
        np.testing.assert_array_equal(costs,[[np.inf,3,1,1],[3,np.inf,1,1],[1,1,np.inf,3],[1,1,3,np.inf]])
        known[(1,2)]={'passed':False}
        weighted=certified.proposal_costs(costs,known,np.zeros((4,4)))
        self.assertTrue(np.isinf(weighted[1,2]))
        self.assertTrue(np.isfinite(weighted[2,1]))


if __name__=='__main__': unittest.main()
