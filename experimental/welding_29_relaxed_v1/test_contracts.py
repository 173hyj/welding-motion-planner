import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from contracts import legal_charts, workflow_phases, validate_workflow, quintic
from staged import select_local_block, freeze_local_blocks
from welding_kinematics import RobotKinematics
import json
import tempfile
from types import SimpleNamespace
from staged import StagedPlanner


class WorkflowTests(unittest.TestCase):
    def test_local_block_selection_uses_whole_line_not_endpoint(self):
        blocks=[dict(weld_id='A',q_rad=[0]*6,local_max_depth_mm=12.,local_mean_soft_cost=1.,
                     stroke=dict(distance_mm=100.,qs=[[0]*6,[.1]*6])),
                dict(weld_id='A',q_rad=[1]*6,local_max_depth_mm=1.,local_mean_soft_cost=2.,
                     stroke=dict(distance_mm=50.,qs=[[1]*6,[1.1]*6]))]
        chosen=select_local_block(blocks,2.)
        self.assertEqual(chosen['q_rad'],[1]*6)
        frozen=freeze_local_blocks([chosen])
        blocks[1]['q_rad'][0]=20
        self.assertEqual(frozen[0]['q_rad'][0],1)

    def test_complete_order_has_ingress_close_weld_open_retreat_transfer(self):
        self.assertEqual(workflow_phases(True), ['ingress', 'close', 'weld', 'open', 'retreat', 'transfer'])
        self.assertEqual(workflow_phases(False), ['ingress', 'close', 'weld', 'open', 'retreat'])
        records = [{'id': str(i), 'phases': workflow_phases(i < 28),
                    'entry_q_rad': [0]*6, 'exit_q_rad': [0]*6} for i in range(29)]
        self.assertTrue(validate_workflow([str(i) for i in range(29)], records))
        with self.assertRaisesRegex(ValueError, '29|missing|source'):
            validate_workflow([str(i) for i in range(29)], records[:-1])

    def test_source_chart_does_not_wrap_samples_out_of_limits(self):
        scene = json.loads((ROOT/'build/scene.json').read_text())
        asset = next(a for a in scene['assets'] if a['kind']=='robot')
        model = RobotKinematics(scene, asset)
        q = np.array(json.loads((ROOT/'build/ik_source_normal_roll_optimized.json').read_text())['welds'][5]['robots'][0]['solutions'][0]['q_rad'])
        reference=q.copy();reference[5]-=2*np.pi
        charts=legal_charts(model,q,reference)
        self.assertTrue(all(model.limits_valid(x) for x in charts))
        self.assertLess(abs(charts[0][5]-reference[5]),1e-6)

    def test_quintic_endpoints(self):
        np.testing.assert_allclose(quintic(np.array([0,.5,1.])),[0,.5,1.])

    def test_rrt_transfer_keeps_endpoints_and_gun_open(self):
        planner=object.__new__(StagedPlanner)
        planner.model=SimpleNamespace(bounds=np.array([[-1.]*6,[1.]*6]),limits_valid=lambda q:np.all(np.abs(q)<=1.))
        drivers=[]
        def check(q,driver):
            drivers.append(driver)
            return dict(observed_max_depth_mm=0.,soft_cost=0.)
        planner.cost=SimpleNamespace(reference=2.,check=check)
        planner.open=-185.;planner.stage3=[];planner.stage2_sha256='frozen'
        planner.force_rrt=True;planner.adaptive_rrt=True;planner.edge_step=.02
        planner.rrt_iterations=30;planner.rrt_seconds=2.
        qa=np.zeros(6);qb=np.full(6,.1)
        with tempfile.TemporaryDirectory() as directory:
            planner.out=Path(directory)
            route=planner.transfer_route(qa,qb,0)
        np.testing.assert_allclose(route[0],qa)
        np.testing.assert_allclose(route[-1],qb)
        self.assertTrue(planner.stage3[0]['rrt_solved'])
        self.assertEqual(planner.stage3[0]['planner'],'RRT-Connect')
        self.assertTrue(all(d==-185. for d in drivers))

    def test_rrt_failure_is_explicit_diagnostic_fallback(self):
        planner=object.__new__(StagedPlanner)
        planner.model=SimpleNamespace(bounds=np.array([[-1.]*6,[1.]*6]),limits_valid=lambda q:np.all(np.abs(q)<=1.))
        qa=np.zeros(6);qb=np.full(6,.1)
        def check(q,driver):
            depth=0. if np.allclose(q,qa) or np.allclose(q,qb) else 10.
            return dict(observed_max_depth_mm=depth,soft_cost=depth)
        planner.cost=SimpleNamespace(reference=2.,check=check)
        planner.open=-185.;planner.stage3=[];planner.stage2_sha256='frozen'
        planner.force_rrt=True;planner.adaptive_rrt=True;planner.edge_step=.02
        planner.rrt_iterations=3;planner.rrt_seconds=2.
        with tempfile.TemporaryDirectory() as directory:
            planner.out=Path(directory)
            route=planner.transfer_route(qa,qb,0)
        self.assertFalse(planner.stage3[0]['rrt_solved'])
        self.assertTrue(planner.stage3[0]['fallback_retained'])
        self.assertEqual(planner.stage3[0]['planner'],'diagnostic_fallback')


if __name__ == '__main__': unittest.main()
