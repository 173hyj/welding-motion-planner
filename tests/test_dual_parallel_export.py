import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    from experimental.dual_arm_parallel_v1.export_parallel import export_parallel, verify_replay
except ImportError:
    export_parallel = verify_replay = None


class ParallelExportTests(unittest.TestCase):
    def fixture(self, root):
        run=root/'run';run.mkdir();source=root/'source.json';source.write_text('{}')
        s=np.zeros((22,15));s[:,5]=-1;s[:,11]=1
        phases=np.full((22,2),'idle',dtype='U16');tasks=np.zeros((22,2),int)
        for task in range(1,22): phases[task,task%2]='weld';tasks[task,task%2]=task
        np.savez_compressed(run/'trajectory.npz',states=s,time=np.arange(22.),phases=phases,tasks=tasks,stage=np.zeros(22,int))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        report=dict(complete=True,display_only=True,source_sha256={str(source):sha(source)},
            trajectory_sha256=sha(run/'trajectory.npz'),order_weld_ids=list(range(101,122)),robot_ids=['r0','r1'],
            original_assignment=[task%2 for task in range(1,22)],
            candidate_policy='existing_selected_tasks_only',validation={'passed':True,'continuous_collision_certified':False},
            timing={'parallel_duration_s':21.},stages=[{'pieces':[[dict(task=t,phase='weld') for t in range(2,22,2)],
                                                            [dict(task=t,phase='weld') for t in range(1,22,2)]]}])
        (run/'report.json').write_text(json.dumps(report));return run,source,report

    def test_export_preserves_trajectory_and_verifies_seal(self):
        self.assertIsNotNone(export_parallel,'parallel exporter missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,source,report=self.fixture(root)
            audit=export_parallel(run,root/'out',tcp_provider=lambda s,r:np.zeros((len(s),2,3)))
            with np.load(run/'trajectory.npz') as original,np.load(root/'out/route.npz') as replay:
                for key in ('states','time','phases','tasks','stage'):np.testing.assert_array_equal(original[key],replay[key])
            self.assertTrue(verify_replay(root/'out')['complete'])
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed bound input'):verify_replay(root/'out')

    def test_duplicate_weld_and_reordered_arm_rejected(self):
        self.assertIsNotNone(export_parallel,'parallel exporter missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,_,report=self.fixture(root)
            report['stages'][0]['pieces'][0].append(dict(task=2,phase='weld'))
            (run/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'21 weld actions'):export_parallel(run,root/'out',tcp_provider=lambda s,r:None)
            report['stages'][0]['pieces'][0]=list(reversed(report['stages'][0]['pieces'][0][:-1]))
            (run/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'arm order'):export_parallel(run,root/'out',tcp_provider=lambda s,r:None)

    def test_modified_array_hash_rejected(self):
        self.assertIsNotNone(export_parallel,'parallel exporter missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,_,_=self.fixture(root)
            (run/'trajectory.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'trajectory'):export_parallel(run,root/'out',tcp_provider=lambda s,r:None)

    def test_missing_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,_,report=self.fixture(root)
            del report['original_assignment'];(run/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'assignment'):export_parallel(run,root/'out',tcp_provider=lambda s,r:None)

if __name__=='__main__':unittest.main()
