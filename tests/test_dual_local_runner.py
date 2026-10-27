import unittest
import numpy as np
from experimental.dual_arm_parallel_v1.run_local import choose_improvement,validate_action_coverage

class AcceptanceTests(unittest.TestCase):
    def test_slower_or_unchecked_parallel_never_replaces_baseline(self):
        rows=[dict(duration=510.,passed=True),dict(duration=490.,passed=False),dict(duration=500.,passed=True)]
        self.assertIsNone(choose_improvement(rows,500.))
    def test_fastest_source_checked_candidate_selected(self):
        rows=[dict(duration=490.,passed=True),dict(duration=470.,passed=True),dict(duration=450.,passed=False)]
        self.assertIs(choose_improvement(rows,500.),rows[1])
    def test_nonfinite_duration_rejected(self):
        self.assertIsNone(choose_improvement([dict(duration=float('nan'),passed=True)],500.))
    def test_missing_or_duplicate_task_cannot_be_accepted(self):
        data=dict(states=np.zeros((2,15)),time=np.array([0.,1.]),tasks=np.zeros((2,2),int),phases=np.full((2,2),'transfer'))
        with self.assertRaises(ValueError):validate_action_coverage(data,[1]*6+[0]*15,np.zeros(15),np.zeros(15))
    def test_changed_global_endpoint_cannot_be_accepted(self):
        states=np.zeros((2,15));states[-1,0]=1.
        data=dict(states=states,time=np.array([0.,1.]),tasks=np.zeros((2,2),int),phases=np.full((2,2),'transfer'))
        with self.assertRaisesRegex(ValueError,'endpoint'):validate_action_coverage(data,[1]*6+[0]*15,np.zeros(15),np.zeros(15))
    def test_all_fixed_actions_once_are_accepted_and_duplicates_rejected(self):
        assignment=[1]*6+[0]*15;labels=[];owners=[]
        for task in range(1,22):
            for phase in ['ingress','close','weld','open','retreat']:
                p=['idle','idle'];t=[0,0];p[assignment[task-1]]=phase;t[assignment[task-1]]=task
                labels.append(p);owners.append(t)
        data=dict(states=np.zeros((105,15)),time=np.arange(105,dtype=float),tasks=np.asarray(owners),phases=np.asarray(labels))
        validate_action_coverage(data,assignment,np.zeros(15),np.zeros(15))
        # Existing sealed action knots can be coarser than new transfer checks.
        # Their exact baseline identity is audited separately; new knots cannot.
        data['states'][1,0]=.002;data['stage']=np.ones(105,int)
        validate_action_coverage(data,assignment,np.zeros(15),np.zeros(15))
        data['stage'][1]=0
        with self.assertRaisesRegex(ValueError,'Fine'):validate_action_coverage(data,assignment,np.zeros(15),np.zeros(15))
        data['states'][:]=0.;data['stage'][:]=0
        data['tasks'][-1,0]=7;data['phases'][-1,0]='weld'
        with self.assertRaises(ValueError):validate_action_coverage(data,assignment,np.zeros(15),np.zeros(15))

if __name__=='__main__':unittest.main()
