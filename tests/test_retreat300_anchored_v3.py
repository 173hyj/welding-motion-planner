import unittest
import numpy as np
from experimental.welding_retreat300_anchored_v3 import run

class Anchored300Tests(unittest.TestCase):
    def test_original_six_stages_and_no_alias(self):
        self.assertEqual([x[0] for x in run.STAGES],[30,120,60,0,120,90])
        self.assertEqual(sorted(t for _,o in run.STAGES for arm in o for t in arm),list(range(1,22)))
    def test_literal_segments_preserve_actual_interstage(self):
        states=np.zeros((12,15));states[:,0]=np.arange(12);states[:,12]=np.arange(12)*.01
        old=dict(states=states,time=np.arange(12)*.7,phases=np.full((12,2),'transfer'),tasks=np.zeros((12,2),int),stage=np.zeros(12,int))
        pieces=[run.literal_slice(old,2,4),run.literal_slice(old,7,9)]
        pieces[0]['time']*=.5;pieces[1]['time']*=.5
        result=run.assemble_segments(old,[(2,4),(7,9)],pieces)
        np.testing.assert_array_equal(result['states'],old['states'])
        np.testing.assert_allclose(np.diff(result['time'])[4:7],np.diff(old['time'])[4:7])
    def test_joint_dp_can_reorder_original_tasks(self):
        node=lambda ident,q:dict(id=ident,endpoint=np.full(7,q),duration_s=1.,joint_travel_deg=1.,wrist_travel_deg=1.)
        nodes={1:[node('a',10.)],2:[node('b',1.)]};orders=[[1,2],[]]
        result=run.build_graph({},nodes,{},orders,[np.zeros(7),np.zeros(7)],[np.full(7,11.),np.zeros(7)],0)
        self.assertEqual(orders[0],[2,1])
        self.assertEqual(result[0],[2,1])
    def test_old_1m_actions_rejected_without_fallback(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d);pool=dict(accepted=True,complete=True,layers=[dict(actions=[dict(length_mm=1000.)]) for _ in range(21)],coverage_all_rows=dict(registered=784,attempted=784,pending=0),source_assets_frozen_before_and_after=True)
            (p/'report.json').write_text(json.dumps(pool))
            with self.assertRaisesRegex(ValueError,'Only freshly generated 300mm'):
                run.optimize(p,p/'out',p/'missing',workers=1,iterations=3)
    def test_hard_duration_gate(self):
        a=dict(duration_s=460.,rotation_episodes=5)
        self.assertFalse(run.acceptance(a,a)['accepted'])

if __name__=='__main__':unittest.main()

