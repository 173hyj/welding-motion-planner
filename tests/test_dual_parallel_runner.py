import unittest
import numpy as np


class RunnerTests(unittest.TestCase):
    def test_delayed_chain_holds_actual_initial_state_and_idle_context(self):
        from experimental.dual_arm_parallel_v1.run_parallel import sample_chain
        chain={'times':[0.,1.], 'states':[np.ones(7),np.ones(7)*2],
               'pieces':[{'start':0.,'end':1.,'phase':'transfer','task':None}]}
        s,p=sample_chain(chain,-.5)
        np.testing.assert_array_equal(s,np.ones(7));self.assertIsNone(p)
        s,p=sample_chain(chain,.5)
        np.testing.assert_array_equal(s,np.ones(7)*1.5);self.assertEqual(p['phase'],'transfer')

    def test_overlap_counts_actual_weld_intersection(self):
        from experimental.dual_arm_parallel_v1.run_parallel import overlap_seconds
        a=[{'start':1.,'end':2.,'phase':'weld','task':1}]
        b=[{'start':0.,'end':1.5,'phase':'weld','task':2}]
        self.assertAlmostEqual(overlap_seconds(a,b,(0.,.5)),1.)
        self.assertEqual(overlap_seconds(a,b,(0.,2.)),0.)

    def test_combination_keeps_drivers_in_mm(self):
        from experimental.dual_arm_parallel_v1.run_parallel import combine_state
        s=combine_state(np.r_[np.zeros(6),-80.],np.r_[np.ones(6),-140.],30.)
        self.assertEqual(s.shape,(15,));np.testing.assert_array_equal(s[13:],[-80.,-140.])
        self.assertAlmostEqual(s[12],np.pi/6)

    def test_release_contact_cannot_reappear(self):
        from experimental.dual_arm_parallel_v1.run_parallel import release_history_passed
        self.assertFalse(release_history_passed([(0.,['cap']),(.1,[]),(.2,['cap'])],['cap']))
        self.assertTrue(release_history_passed([(0.,['cap']),(.1,['cap']),(.2,[]),(10.,[])],['cap']))

    def test_reverse_ingress_uses_forward_release_order(self):
        from experimental.dual_arm_parallel_v1.run_parallel import release_history_passed
        ingress=[(10.,[]),(.2,[]),(.1,['cap']),(0.,['cap'])]
        self.assertTrue(release_history_passed(ingress,['cap'],reverse=True))

    def test_baseline_time_uses_identical_display_limits(self):
        from experimental.dual_arm_parallel_v1.run_parallel import check_baseline_timing
        states=np.zeros((3,15));states[1:,0]=np.deg2rad(30.)
        data={'states':states,'times':np.array([0.,1.,1.6]),'tcp':np.zeros((3,2,3)),
              'phases':np.array(['transfer','transfer','weld'])}
        self.assertAlmostEqual(check_baseline_timing(data),1.6)
        data['times']=np.array([0.,2.,2.6])
        with self.assertRaises(ValueError):check_baseline_timing(data)
