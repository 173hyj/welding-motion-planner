import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experimental.welding_retreat300_planning_v1 import run as base
from experimental.welding_retreat300_known_v2.run import KnownEdges, load_sealed, reusable, GraphRounds

class Tests(unittest.TestCase):
    def test_far_candidate_preserved_and_unknown_blocked(self):
        nodes={1:[dict(id=str(i),endpoint=np.ones(7)*i,duration_s=1.,joint_travel_deg=0.,wrist_travel_deg=0.) for i in range(100)]}
        row=dict(passed=True,duration_s=2.,joint_travel_deg=0.,wrist_travel_deg=0.)
        records={(0,0,'start','99'):row,(0,0,'99','end'):row}
        orders=[[1],[]]
        base.build_graph({},nodes,KnownEdges(records),orders,[np.zeros(7)]*2,[np.zeros(7)]*2,0)
        self.assertEqual(base.CURRENT_SOLUTION['indices'],[99])
        self.assertEqual(len(nodes[1]),100)
        del records[0,0,'99','end']
        base.build_graph({},nodes,KnownEdges(records),orders,[np.zeros(7)]*2,[np.zeros(7)]*2,0)
        self.assertIsNone(base.CURRENT_SOLUTION)
        self.assertEqual(len(nodes[1]),100)

    def test_final_rounds_exclude_initial_participation(self):
        seen=[]
        def graph(*args):seen.append(isinstance(args[2],KnownEdges));return None
        wrapper=GraphRounds(graph,4,2,'.')
        for stage in range(len(base.STAGES)):wrapper({}, {}, {}, [[],[]], [], [], stage)
        for i in range(4):wrapper({}, {}, {}, [[],[]], [], [], 0)
        self.assertEqual(seen,[False]*7+[True,True])
        self.assertEqual(wrapper.counts,{0:4})

    def test_reuse_hash_endpoints_and_certification(self):
        from experimental.welding_candidate_selection_v1.pool import digest
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            folder=Path(folder);(folder/'edges').mkdir();p=folder/'edges/e00000.npz'
            states=np.array([np.zeros(7),np.ones(7)])
            np.savez(p,states=states,times=[0.,1.],tcp=np.zeros((2,3)))
            row=dict(key=[0,0,'start','x'],passed=True,file=str(p),sha256=digest(p),stats=dict(fine_certified=True),duration_s=1.,joint_travel_deg=0.,wrist_travel_deg=0.)
            row.update(base.local_metrics(states,np.array([0.,1.])))
            receipt=p.with_suffix('.json');receipt.write_text(json.dumps(row))
            index,bindings=load_sealed(folder)
            self.assertIsNotNone(reusable(index,tuple(row['key']),states[0],states[-1]))
            self.assertIsNone(reusable(index,tuple(row['key']),states[0],states[-1]+1))
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):reusable(index,tuple(row['key']),states[0],states[-1])
            row['stats']['fine_certified']=False;receipt.write_text(json.dumps(row))
            self.assertEqual(load_sealed(folder)[0],{})

if __name__=='__main__':unittest.main()

