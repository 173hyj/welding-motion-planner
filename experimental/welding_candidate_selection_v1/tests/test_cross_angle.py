import unittest
import numpy as np
from experimental.welding_candidate_selection_v1.cross_angle import warp_seed


class CrossAngleTests(unittest.TestCase):
    def test_both_wrist_endpoints_change_without_wrapping(self):
        old=np.zeros((5,15));old[:,3]=[2,1,0,1,2]
        old[:,12]=[0,0,.5,1,1];old[:,6]=3;old[:,14]=-10
        a,b=old[[0,-1]].copy();a[3]-=2*np.pi;b[3]-=2*np.pi
        result=warp_seed(old,a,b)
        np.testing.assert_array_equal(result[[0,-1]],[a,b])
        np.testing.assert_array_equal(result[:,6:13],old[:,6:13])
        np.testing.assert_array_equal(result[:,14],old[:,14])
        np.testing.assert_allclose(result[:,3],old[:,3]-2*np.pi)

    def test_progress_keeps_original_stationary_stages(self):
        old=np.zeros((5,15));old[:,0]=[0,0,1,2,2];old[:,12]=[0,0,.5,1,1]
        a,b=old[[0,-1]].copy();a[0]=1;b[0]=4
        result=warp_seed(old,a,b)
        np.testing.assert_array_equal(result[:,0],[1,1,2.5,4,4])

    def test_rejects_passive_change(self):
        old=np.zeros((2,15));b=old[-1].copy();b[6]=1
        with self.assertRaises(ValueError):warp_seed(old,old[0],b)


if __name__=='__main__':unittest.main()
