import unittest
try:
    import selected_pose_v86 as pose
except ImportError:
    pose=None


class SelectedPoseTests(unittest.TestCase):
    def test_source_robot_id_in_candidate_and_explicit_fields_does_not_collide(self):
        self.assertIsNotNone(pose, 'selected pose feature missing')
        original=dict(robot_id='old',q_rad=[0],other_q_rad=[1],untouched=True)
        candidate=dict(robot_id='new',roll_deg=55,axis_sign=1)
        result=pose.selected_pose(original,candidate,robot_id='new',q_rad=[2],other_q_rad=[3])
        self.assertEqual(result,dict(robot_id='new',roll_deg=55,axis_sign=1,q_rad=[2],other_q_rad=[3],untouched=True))
        self.assertEqual(original['robot_id'],'old')
        self.assertEqual(candidate['robot_id'],'new')
        with self.assertRaises(ValueError):
            pose.selected_pose(original,candidate,robot_id='wrong')


if __name__=='__main__': unittest.main()
