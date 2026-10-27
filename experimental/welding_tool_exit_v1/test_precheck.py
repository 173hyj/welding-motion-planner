import unittest
import numpy as np
from tool_world import ToolWorld


class PrecheckTests(unittest.TestCase):
    def test_obvious_retreat_body_collision_checked_before_fine_release(self):
        engine = ToolWorld.__new__(ToolWorld)
        phases = []
        def blocked(rid,target,driver,weld,environment,phase='opening',release=None):
            phases.append(phase)
            return dict(blocking_pairs=[{'part':'body'}],touching=[],certificates={})
        engine.state = blocked
        context = dict(initial_touching=['cap'],supports=[{'outward_normal':[0,0,1]}])
        result = engine.straight_exit('r',np.eye(4),{},context,[0,0,1],environment=[])
        self.assertEqual(result['reason'],'retreat_blocked')
        self.assertEqual(phases,['separated'])


if __name__ == '__main__': unittest.main()
