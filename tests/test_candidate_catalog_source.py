import math
import unittest
from types import SimpleNamespace
import numpy as np
from experimental.welding_candidate_catalog_v1 import source

class SourceTests(unittest.TestCase):
    def test_only_own_self_and_environment_pairs_remain(self):
        pairs = [dict(checked_robot_id='a',asset_name='self',link='240'),
                 dict(checked_robot_id='a',asset_name='fixture'),
                 dict(checked_robot_id='a',asset_name='other_robot'),
                 dict(checked_robot_id='b',asset_name='fixture')]
        self.assertEqual(source.filter_single_arm_pairs(pairs,'a'), pairs[:2])

    def test_unannotated_own_pairs_are_not_lost(self):
        pair = dict(asset_name='self',link='240',asset_id='gun:body')
        self.assertEqual(source.filter_single_arm_pairs([pair],'a'), [pair])

    def test_certificate_filter_result_uses_remaining_pairs(self):
        own = dict(checked_robot_id='a',asset_name='fixture')
        filtered = dict(remaining_pairs=[own,dict(checked_robot_id='b',asset_name='self')],passed=False)
        self.assertEqual(source.qualified_single_arm_pairs(filtered,'a'),[own])

    def test_solve_retains_converged_coupled_limit_failure_then_accept_rejects(self):
        # A linear six-axis pose problem reaches a known exact branch, while
        # the source coupled-domain predicate excludes that branch.
        class LinearPose:
            def __init__(self, model, target): pass
            def evaluate(self, q): return np.asarray(q)-.25, np.eye(6)
        catalog = source.SourceCatalog.__new__(source.SourceCatalog)
        catalog.np = np
        catalog.PoseProblem = LinearPose
        catalog.seed_count = 0
        catalog.nearby = {}
        model = SimpleNamespace(q_initial=np.zeros(6), bounds=np.array([[-1.]*6,[1.]*6]),
                                limits_valid=lambda q: bool(q[0] < .1))
        catalog.world = SimpleNamespace(models={'a':model})
        catalog.inventory = {'w':{'candidates':[]}}
        catalog._target = lambda weld, pose: ({}, np.eye(4))
        solutions, stats = catalog.solve({'id':'w'}, {'robot_id':'a'})
        self.assertEqual(len(solutions), 1)
        self.assertFalse(solutions[0]['source_joint_limits_valid'])
        self.assertGreater(stats['illegal_solutions'], 0)
        self.assertTrue(np.isfinite(stats['best_residual']))
        result = catalog.accept({'id':'w'}, {'robot_id':'a'}, solutions[0])
        self.assertFalse(result['passed'])
        self.assertEqual(result['reason'],'source_joint_limits_invalid')

    def test_seed_inventory_includes_failed_checks_and_selected_state(self):
        a = dict(robot_id='a',q_rad=[0.]*6,passed=False)
        b = dict(robot_id='b',q_rad=[1.]*6,passed=True)
        selected = dict(robot_id='a',q_rad=[2.]*6)
        inventory = source.inventory_from_static_report({'rows':[dict(weld_id='w',full_checks=[a,b],selected=selected)]})
        candidates = inventory['w']['candidates']
        self.assertEqual([c['q_rad'] for c in candidates], [a['q_rad'],b['q_rad'],selected['q_rad']])
        self.assertTrue(all(c['fk_candidate_passed'] for c in candidates))

    def test_distinct_windings_survive_raw_deduplication(self):
        candidates = [dict(q_rad=[0]*6),dict(q_rad=[2*math.pi]+[0]*5),dict(q_rad=[1e-7]+[0]*5)]
        self.assertEqual(source.unique_raw_solutions(candidates),candidates[:2])

if __name__ == '__main__': unittest.main()
