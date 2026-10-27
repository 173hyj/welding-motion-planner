import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('handoff',Path(__file__).absolute().parents[1]/'handoff.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class HandoffTests(unittest.TestCase):
    def test_winding_proposal_preserves_original_other_arm_and_rotation(self):
        original=np.zeros((5,15));original[:,3]=[0,0,1,2,3]
        original[:,6]=[2,1,0,-1,-3];original[:,12]=[2,2,2,1,1]
        target=original[-1].copy();target[3]-=2*np.pi
        proposed=h.warp_seed(original,original[0],target)
        np.testing.assert_array_equal(proposed[:,6:],original[:,6:])
        np.testing.assert_array_equal(proposed[0],original[0])
        np.testing.assert_array_equal(proposed[-1],target)
        self.assertEqual(proposed[1,3],0)
        self.assertLess(proposed[-1,3],-3.)

    def test_invalid_seed_shape_rejected(self):
        with self.assertRaises(ValueError):h.warp_seed(np.zeros((2,6)),np.zeros(15),np.zeros(15))


if __name__=='__main__':unittest.main()
