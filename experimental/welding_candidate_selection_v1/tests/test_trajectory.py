import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).absolute().parents[3]))
from experimental.welding_candidate_selection_v1.trajectory import combine, motion_metrics, acceptance


class TrajectoryTests(unittest.TestCase):
    def test_splice_preserves_prefix_and_accumulates_piece_clocks(self):
        states=np.zeros((2,15));states[1,0]=1
        prefix=dict(states=states,time=np.array([0.,2.]),times=np.array([0.,2.]),
                    tcp=np.zeros((2,2,3)),phases=np.full((2,2),'idle'),
                    tasks=np.zeros((2,2),int),stage=np.zeros(2,int))
        tail=states.copy();tail[:,0]=[1,2]
        piece=dict(states=tail,time=np.array([0.,3.]),tcp=np.zeros((2,2,3)),
                   phases=np.full((2,2),'transfer'),tasks=np.zeros((2,2),int),stage=np.ones(2,int))
        result=combine(prefix,[piece])
        np.testing.assert_array_equal(result['states'][:2],states)
        np.testing.assert_array_equal(result['time'],[0,2,5])
        np.testing.assert_array_equal(result['times'],result['time'])

    def test_joint_seam_never_wraps(self):
        prefix=dict(states=np.zeros((1,15)),time=np.array([0.]),tcp=np.zeros((1,2,3)),
                    phases=np.full((1,2),'idle'),tasks=np.zeros((1,2),int),stage=np.zeros(1,int))
        states=np.zeros((2,15));states[:,5]=2*np.pi
        piece=dict(prefix,states=states,time=np.array([0,1.]))
        with self.assertRaises(ValueError):combine(prefix,[piece])

    def test_motion_counts_actual_detour_and_wrist_without_wrap(self):
        states=np.zeros((3,15));states[:,5]=np.deg2rad([179,-179,179])
        result=motion_metrics(states,np.array([0.,20.,40.]))
        self.assertAlmostEqual(result['joint_travel_deg'],716.)
        self.assertAlmostEqual(result['wrist_travel_deg'],716.)

    def test_adoption_requires_time_and_motion_improvement(self):
        before=dict(duration_s=10.,joint_travel_deg=100.,wrist_travel_deg=50.,rotation_episodes=5,total_positioner_travel_deg=360.)
        after=dict(before,joint_travel_deg=90.,wrist_travel_deg=45.)
        self.assertTrue(acceptance(before,after)['accepted'])
        self.assertFalse(acceptance(before,dict(after,duration_s=11.))['accepted'])
        self.assertFalse(acceptance(before,dict(after,wrist_travel_deg=55.))['accepted'])


if __name__=='__main__':unittest.main()
