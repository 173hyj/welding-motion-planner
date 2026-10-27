import unittest
import numpy as np
try:
    from experimental.dual_arm_parallel_v1 import splice
except ImportError:
    splice=None
from experimental.dual_arm_parallel_v1.motion import piece_at


def fixture():
    times=[0.];states=[np.zeros(15)];phases=['ingress'];tasks=[1]
    assignment=[i%2 for i in range(21)]
    for task in range(1,22):
        for phase in ('ingress','close','weld','open','retreat','transfer'):
            if task==21 and phase=='transfer':continue
            for repeat in range(2):
                state=states[-1].copy()
                if phase!='weld' and repeat==1:state[assignment[task-1]*6]+=0.01
                states.append(state);times.append(times[-1]+(.6 if phase=='weld' else .125));phases.append(phase);tasks.append(task)
    return dict(times=np.array(times),states=np.array(states),phases=np.array(phases),tasks=np.array(tasks)),assignment


class SpliceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(splice,'baseline splice helpers must exist')
    def test_span_retains_predecessor_hold_and_weld_dwell(self):
        data,_=fixture();s=splice.baseline_span(data,7,'weld')
        idx=np.flatnonzero((data['tasks']==7)&(data['phases']=='weld'))
        np.testing.assert_array_equal(s['states'],data['states'][idx[0]-1:idx[-1]+1])
        np.testing.assert_allclose(s['times'],[0,.6,1.2])
        s=splice.baseline_span(data,21)
        self.assertEqual(s['phases'][-1],'retreat')
        self.assertEqual(s['tasks'][-1],21)

    def test_chain_keeps_exact_times_projection_and_piece_boundaries(self):
        data,assignment=fixture();chain=splice.arm_chain_from_baseline(data,0,[0]*21,1,2)
        end=np.flatnonzero((data['tasks']==2)&(data['phases']=='retreat'))[-1]
        np.testing.assert_array_equal(chain['times'],data['times'][:end+1])
        np.testing.assert_array_equal(chain['states'],data['states'][:end+1][:,[0,1,2,3,4,5,13]])
        self.assertEqual([p['phase'] for p in chain['pieces']],['ingress','close','weld','open','retreat','transfer','ingress','close','weld','open','retreat'])
        self.assertIsNone(next(p for p in chain['pieces'] if p['phase']=='transfer')['task'])
        for p in chain['pieces']:self.assertEqual(piece_at(chain,p['start'])['phase'],p['phase'])
        with self.assertRaises(ValueError):splice.arm_chain_from_baseline(data,0,assignment,1,2)

    def test_stitch_serial_preserves_every_timestamp_and_task21(self):
        data,assignment=fixture();cut=19
        a=splice.serial_part(data,assignment,end_index=cut)
        b=splice.serial_part(data,assignment,start_index=cut)
        merged=splice.stitch_parts([a,b])
        np.testing.assert_array_equal(merged['time'],data['times'])
        np.testing.assert_array_equal(merged['states'],data['states'])
        self.assertEqual(merged['tasks'][-1,0],21)
        self.assertEqual(merged['phases'][-1,1],'idle')
        transfer=data['phases']=='transfer'
        self.assertTrue(np.all(merged['phases'][transfer]=='transfer'))
        self.assertTrue(np.all(merged['tasks'][transfer]==0))
        b['states'][0,0]+=1e-9
        with self.assertRaisesRegex(ValueError,'seam'):splice.stitch_parts([a,b])

    def test_invalid_arrays_and_noncontiguous_spans_rejected(self):
        data,assignment=fixture();data['times'][3]=data['times'][2]
        with self.assertRaises(ValueError):splice.baseline_span(data,1)
        data,assignment=fixture();data['states'][2,0]=np.nan
        with self.assertRaises(ValueError):splice.serial_part(data,assignment)
        data,assignment=fixture();data['tasks'][3]=9
        with self.assertRaises(ValueError):splice.baseline_span(data,1)

if __name__=='__main__':unittest.main()


