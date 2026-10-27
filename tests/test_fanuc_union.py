import copy
import math
import unittest
from experimental.welding_candidate_catalog_analytic_v1.union import merge_trials


def solution(q=0):
    return dict(q_rad=[q,0,0,0,0,0], source_joint_limits_valid=True,
                acceptance=dict(passed=True))


def row(states, robot='a', passed=True):
    return dict(robot_id=robot,roll_deg=0,axis_sign=1,geometry=dict(passed=passed),
                status='ik_not_found',solutions=[dict(raw=True)],accepted=states)


class UnionTests(unittest.TestCase):
    def test_union_preserves_raw_windings_and_inputs(self):
        analytic=[row([solution(0),solution(1)])]
        numeric=[row([solution(0.00001),solution(2*math.pi)])]
        before=copy.deepcopy((analytic,numeric))
        merged,stats=merge_trials(analytic,numeric)
        self.assertEqual([s['origin'] for s in merged[0]['accepted']],
                         ['analytic_and_numeric','analytic','numeric_retained'])
        self.assertEqual(stats,dict(numeric_candidates=2,analytic_candidates=2,
            matched_numeric=1,numeric_retained=1,new_analytic=1,union_candidates=3))
        self.assertEqual(merged[0]['solutions'],analytic[0]['solutions'])
        self.assertEqual(merged[0]['numeric_retained_count'],1)
        self.assertEqual(merged[0]['numeric_retained'],[merged[0]['accepted'][-1]])
        self.assertEqual(merged[0]['status'],'candidates_found')
        self.assertEqual((analytic,numeric),before)
        merged[0]['accepted'][0]['acceptance']['passed']=False
        self.assertEqual((analytic,numeric),before)

    def test_raw_solution_aliases_are_not_annotated(self):
        state=solution()
        a=row([state]); a['solutions']=[state]
        merged,_=merge_trials([a],[row([])])
        self.assertEqual(merged[0]['solutions'],[state])
        self.assertNotIn('origin',merged[0]['solutions'][0])

    def test_analytic_only_and_empty_status(self):
        merged,stats=merge_trials([row([solution()]),row([],robot='b')],
                                  [row([]),row([],robot='b')])
        self.assertEqual(stats['new_analytic'],1)
        self.assertEqual(merged[0]['accepted'][0]['origin'],'analytic')
        self.assertEqual(merged[1]['status'],'ik_not_found')

    def test_threshold_is_strict_and_euclidean(self):
        a=solution(); n=solution(); n['q_rad'][:2]=[0.00008,0.00008]
        merged,stats=merge_trials([row([a])],[row([n,solution(.0001)])])
        self.assertEqual(stats['numeric_retained'],2)
        self.assertEqual(len(merged[0]['accepted']),3)

    def test_matching_counts_each_numeric_without_adding_duplicates(self):
        _,stats=merge_trials([row([solution()])],[row([solution(-.00009),solution(.00009)])])
        self.assertEqual(stats['matched_numeric'],2)
        self.assertEqual(stats['union_candidates'],1)
        self.assertEqual(stats['new_analytic'],0)

    def test_nested_pose_and_reordered_rows(self):
        a=row([solution()]); b=row([],robot='b')
        for item in (a,b):
            item['pose']={k:item.pop(k) for k in ('robot_id','roll_deg','axis_sign')}
        merged,_=merge_trials([a,b],[b,a])
        self.assertEqual(merged[0]['pose']['robot_id'],'a')

    def test_invalid_inputs_raise(self):
        bad_solutions=[dict(solution(),q_rad=[0]*5),dict(solution(),q_rad=[float('nan')]*6),
                       dict(solution(),source_joint_limits_valid=False),
                       dict(solution(),acceptance=dict(passed=False))]
        for bad in bad_solutions:
            for side in (0,1):
                args=[[row([])],[row([])]]; args[side]=[row([bad])]
                with self.subTest(bad=bad,side=side), self.assertRaises(ValueError):
                    merge_trials(*args)
        for a,n in [([row([])],[]),([row([]),row([])],[row([])]),
                    ([row([])],[row([],passed=False)]),
                    ([row([solution()],passed=False)],[row([],passed=False)])]:
            with self.assertRaises(ValueError): merge_trials(a,n)
        for tolerance in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): merge_trials([],[],tolerance)


if __name__=='__main__': unittest.main()

