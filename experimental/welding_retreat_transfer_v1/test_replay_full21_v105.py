import unittest
import numpy as np
from replay_full21_core_v105 import append_states, local_states, interpolate_state


class ReplayTests(unittest.TestCase):
    def test_local_state_has_real_inactive_park_and_positioner(self):
        path=dict(robot_id='b', other_robot_id='a', other_q_rad=[1.]*6,
                  positioner_deg=45., frames=[dict(q_rad=[2.]*6,driver_mm=-120.)])
        state=local_states(path,['a','b'])[0]
        np.testing.assert_array_equal(state[:6],[1.]*6)
        np.testing.assert_array_equal(state[6:12],[2.]*6)
        self.assertAlmostEqual(state[12],np.pi/4)
        np.testing.assert_array_equal(state[13:],[0.,-120.])

    def test_bad_seam_rejected_and_matching_seam_deduplicated(self):
        data=[np.zeros(15)]
        with self.assertRaises(ValueError):append_states(data,[np.ones(15)])
        append_states(data,[np.zeros(15),np.ones(15)])
        self.assertEqual(len(data),2)

    def test_interpolation_does_not_wrap_or_extrapolate(self):
        states=np.array([[0.,3.],[1.,-3.]])
        np.testing.assert_allclose(interpolate_state([0.,2.],states,1.),[.5,0.])
        np.testing.assert_array_equal(interpolate_state([0.,2.],states,10.),states[-1])


if __name__=='__main__':unittest.main()
