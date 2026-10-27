import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
try:
    import trace_motion as motion
except ImportError:
    motion=None


class TraceTests(unittest.TestCase):
    def test_collision_failure_keeps_actual_failure_q_and_last_accepted_q(self):
        self.assertIsNotNone(motion,'failure-q trace missing')
        class Model:
            def tcp(self,q):
                t=np.eye(4);t[0,3]=q[0]*1000.;return t
        e=SimpleNamespace(world=SimpleNamespace(models={'r':Model()},static=[]),delta=np.eye(4))
        e.target=lambda *args: ({},np.eye(4))
        e.state=lambda *args:dict(touching=[],blocking_pairs=[])
        e.release_context=lambda *args,**kw:dict(initial_touching=[],supports=[])
        e.minimum_distance=lambda *args:10.
        initial=dict(q_rad=[0.],other_robot_id='other',other_q_rad=[0.],positioner_deg=0.)
        frames=[dict(q_rad=[0.],driver_mm=d,phase='opening',tcp_xyz_mm=[0.,0.,0.],travel_mm=0.) for d in (0.,-30.)]
        with patch.object(motion,'full_probe',lambda e,r,q,*args:dict(passed=q[0]<.003,touching=[])), \
             patch.object(motion,'continuous_ik',lambda m,t,q:dict(passed=True,q_rad=[t[0,3]/1000.])):
            result=motion.lift_path(e,{},dict(robot_id='r'),initial,
                dict(direction_world=[1.,0.,0.],direction_frame='current_world',length_mm=100.),driver_mm=-30.,opening_frames=frames)
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'],'full_retreat_blocked')
        self.assertGreaterEqual(result['failure_q_rad'][0],.003)
        self.assertLess(result['last_accepted_q_rad'][0],.003)
        self.assertGreater(result['failure_travel_mm'],result['last_accepted_travel_mm'])


if __name__=='__main__':unittest.main()
