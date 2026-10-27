import unittest
import numpy as np
from experimental.welding_retreat300_planning_v1 import run

class IntegrationTests(unittest.TestCase):
    def test_runner_reorders_tasks_and_selects_last_candidate(self):
        def node(name,q,duration=0):
            return dict(id=name,endpoint=np.array([q,0,0,0,0,0,0.]),duration_s=duration,joint_travel_deg=0.,wrist_travel_deg=0.)
        nodes={1:[node('a0',3),node('a1',2)],2:[node('b',1)]}
        # No top-k: only last task1 candidate has cheap certified inbound.
        records={}
        def edge(left,right,d):records[0,0,left,right]=dict(passed=True,duration_s=d,joint_travel_deg=0.,wrist_travel_deg=0.)
        for left,right,d in [('start','a0',100),('start','a1',100),('start','b',0),('b','a0',100),('b','a1',0),('a0','b',100),('a1','b',100),('a0','end',0),('a1','end',0),('b','end',100)]:edge(left,right,d)
        orders=[[1,2],[]];parks=[np.zeros(7),np.zeros(7)]
        ordered,_,_,_,keys=run.build_graph({},nodes,records,orders,parks,parks,0)
        self.assertEqual(ordered,[2,1])
        self.assertEqual(orders,[[2,1],[]])
        self.assertEqual(run.CURRENT_SOLUTION['indices'],[0,1])
        self.assertIn((0,0,'start','a0'),keys)
        transfers=run.selected_transfers(ordered,[0,1],nodes,orders,parks,parks,0)
        self.assertIn((0,1,'start','end'),[x[0] for x in transfers])

    def test_complete_original_task_set_and_four_rotations(self):
        self.assertEqual(sorted(t for a,orders in run.STAGES for order in orders for t in order),list(range(1,22)))
        angles=[a for a,_ in run.STAGES]
        self.assertEqual(angles,[30,0,60,120,90])
        self.assertEqual(sum(a!=b for a,b in zip([30]+angles,angles)),4)

    def test_adoption_never_accepts_slower_four_rotation_route(self):
        before=dict(duration_s=459.,rotation_episodes=5)
        self.assertFalse(run.acceptance(before,dict(duration_s=460.,rotation_episodes=4))['accepted'])
        self.assertTrue(run.acceptance(before,dict(duration_s=458.,rotation_episodes=4))['accepted'])

    def test_inherited_hash_conflicts_rejected(self):
        with self.assertRaises(ValueError):run.merge_bindings({'sealed':'a'},{'sealed':'b'})
        self.assertEqual(run.merge_bindings({'sealed':'a'},{'sealed':'a'}),{'sealed':'a'})

    def test_rejects_nonfinite_secondary_and_endpoint_extra_rows(self):
        from .graph import grouped_open_path
        with self.assertRaises(ValueError):grouped_open_path([[[0.,np.inf]]],{})
        with self.assertRaises(ValueError):grouped_open_path([[[0.]]],{},start_costs=[[[0.]],[[0.]]])

if __name__=='__main__':unittest.main()
