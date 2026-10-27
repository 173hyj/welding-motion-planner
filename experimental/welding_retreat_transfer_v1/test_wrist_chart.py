import unittest
import numpy as np
from wrist_chart import chart_options


class WristChartTests(unittest.TestCase):
    def test_chart_uses_one_constant_mapping_for_the_whole_path(self):
        path=np.deg2rad([[0,0,0,-40,-115,-20],[0,0,0,-39,-114,-18]])
        choices=chart_options(path,np.deg2rad([-185,-60,-132,-210,-120,-205]),
                                   np.deg2rad([185,76,90,210,120,205]))
        self.assertTrue(choices)
        mapped=choices[0]['q']
        self.assertTrue(np.all(mapped[:,5]>0.))
        np.testing.assert_allclose(mapped[:,3]-path[:,3],np.pi)
        np.testing.assert_allclose(mapped[:,4],-path[:,4])
        np.testing.assert_allclose(mapped[:,5]-path[:,5],np.pi)

    def test_equivalent_rotation_only_if_all_frames_within_limits(self):
        path=np.deg2rad([[0,0,0,-55,-65,-190],[0,0,0,-54,-66,-187]])
        lo=np.deg2rad([-185,-60,-132,-210,-120,-205]);hi=np.deg2rad([185,76,90,210,120,205])
        choices=chart_options(path,lo,hi)
        self.assertTrue(any(np.allclose(c['q'][:,5],path[:,5]+2*np.pi) for c in choices))
        bad=path.copy();bad[0,5]=np.deg2rad(-150)
        self.assertFalse(any(c['label']=='q6_plus_360' for c in chart_options(bad,lo,hi)))


if __name__=='__main__':unittest.main()
