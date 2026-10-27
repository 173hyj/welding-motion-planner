import json
import unittest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WeldTaskPlannerTests(unittest.TestCase):
    def test_planner_preserves_all_source_weld_ids_and_declares_unavailable(self):
        from weld_task_planner import plan_source_tasks
        scene = json.loads((ROOT / 'build/scene.json').read_text())
        result = plan_source_tasks(scene, ROOT / 'build/eight_hour_v17')
        self.assertEqual([x['id'] for x in result['welds']], [x['id'] for x in scene['welds']])
        self.assertEqual(len(result['welds']), 29)
        self.assertTrue(all(x['status'] in {'planned', 'endpoint_unavailable', 'transit_rejected'} for x in result['welds']))
        self.assertTrue(result['status']['collision_verified'] is False)
        self.assertTrue(result['status']['source_to_park_path_verified'] is False)

    def test_planned_segment_time_uses_source_limits(self):
        from weld_task_planner import segment_duration
        self.assertGreater(segment_duration([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1.5707964] * 6, [1.5707964] * 6), 0.6)
        self.assertNotEqual(segment_duration([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1.5707964] * 6, [1.5707964] * 6), 0.1)

    def test_invalid_transit_is_never_marked_planned(self):
        from weld_task_planner import classify_route
        self.assertEqual(classify_route([{'collision_verified': False}]), 'transit_rejected')

    def test_unknown_inventory_cannot_silently_disappear(self):
        from weld_task_planner import plan_source_tasks
        scene = json.loads((ROOT / 'build/scene.json').read_text())
        with self.assertRaisesRegex(ValueError, 'Unknown source weld'):
            plan_source_tasks(scene, ROOT / 'build/eight_hour_v17', {'invented': []})

    def test_time_law_identifies_source_limits_and_coupling(self):
        from gun_kinematics import GunKinematics
        from weld_task_planner import SourceTiming
        scene = json.loads((ROOT / 'build/scene.json').read_text())
        gun = GunKinematics(json.loads((ROOT / 'build/gunR1_detail.json').read_text()))
        timing = SourceTiming(scene, gun)
        metadata = timing.metadata()
        self.assertEqual(metadata['polynomial'], '10*tau^3-15*tau^4+6*tau^5')
        self.assertEqual(metadata['boundary_velocity'], [0., 0.])
        self.assertEqual(len(metadata['robot_axes']['maxSpeed_rad_s']), 6)
        np.testing.assert_allclose(metadata['robot_axes']['coupling_matrix'], timing.coupling)
        self.assertEqual(len(metadata['gun_axes']['maxSpeed_source_units_s']), 4)

    def test_failed_first_state_is_retained_in_edge_audit(self):
        from weld_task_planner import WeldTaskPlanner
        planner = WeldTaskPlanner.__new__(WeldTaskPlanner)
        class Timing:
            def duration(self, *args): return 1.
            def sample_motion(self, qa, qb, da, db, *args):
                yield 0., qa, da
                yield 1., qb, db
            def metadata(self): return {'name': 'test'}
        planner.timing, planner.joint_step, planner.gun_step, planner.audit = Timing(), .01, 5., []
        planner.check = lambda *args: {'collision_verified': False, 'joint_limits_valid': True, 'pairs': [{'link': 'test'}]}
        result = planner.edge(np.zeros(6), np.ones(6), 0., 0., 'transfer')
        self.assertEqual(result['status'], 'transit_rejected')
        self.assertEqual(result['samples'][0]['time_s'], 0.)
        self.assertEqual(planner.audit[0]['failure']['time_s'], 0.)
        self.assertEqual(result['time_law']['name'], 'test')

    def test_phase_seam_rejects_teleport_even_when_samples_pass(self):
        from weld_task_planner import validate_segment_chain
        def segment(a, b):
            return {'status': 'planned', 'duration_s': 1., 'samples': [
                {'q_rad': [a]*6, 'driver_mm': -185., 'time_s': 0., 'collision_verified': True},
                {'q_rad': [b]*6, 'driver_mm': -185., 'time_s': 1., 'collision_verified': True}]}
        validate_segment_chain([segment(0., 1.), segment(1., 2.)])
        with self.assertRaisesRegex(ValueError, 'Discontinuous'):
            validate_segment_chain([segment(0., 1.), segment(1.1, 2.)])

    def test_source_timing_bounds_independent_coupled_and_nonlinear_axes(self):
        from gun_kinematics import GunKinematics
        from weld_task_planner import SourceTiming
        scene = json.loads((ROOT / 'build/scene.json').read_text())
        gun = GunKinematics(json.loads((ROOT / 'build/gunR1_detail.json').read_text()))
        timing = SourceTiming(scene, gun)
        qa, qb = np.zeros(6), np.ones(6)
        T = timing.duration(qa, qb, 0., -185.)
        times = np.linspace(0., T, 10001)
        tau = times/T
        u = 10*tau**3-15*tau**4+6*tau**5
        independent = qa + u[:,None]*(qb-qa)
        coupled = independent @ timing.coupling.T
        nonlinear = np.array([gun.joint_values(-185.*value) for value in u])
        for positions, speed, acceleration in [(independent,timing.speeds,timing.accelerations),
                (coupled,timing.speeds,timing.accelerations), (nonlinear,timing.gun_speeds,timing.gun_accelerations)]:
            velocity = np.gradient(positions,times,axis=0,edge_order=2)
            actual_acceleration = np.gradient(velocity,times,axis=0,edge_order=2)
            self.assertLess(float(np.max(np.abs(velocity)/speed)),1.)
            self.assertLess(float(np.max(np.abs(actual_acceleration)/acceleration)),1.)

    def test_contact_boundary_samples_cannot_be_skipped_by_coarse_edges(self):
        from weld_task_planner import contact_boundary_fractions
        class LinearTCP:
            def tcp(self, q):
                p=np.eye(4);p[0,3]=q[0]*100.;return p
        qa,qb=np.zeros(6),np.array([1.,0.,0.,0.,0.,0.])
        values=contact_boundary_fractions(LinearTCP(),qa,qb,{'world_xyz':[0.,0.,0.]})
        self.assertTrue(any(.1 < value*100. < .12 for value in values))
        self.assertTrue(any(.19 < value*100. < .21 for value in values))
        reversed_values=contact_boundary_fractions(LinearTCP(),qb,qa,{'world_xyz':[0.,0.,0.]})
        self.assertTrue(any(.1 < (1-value)*100. < .12 for value in reversed_values))

    def test_source_passive_values_reconstruct_closure_transforms(self):
        from weld_task_planner import passive_joint_values
        from welding_kinematics import RobotKinematics, axis_matrix
        scene=json.loads((ROOT/'build/scene.json').read_text())
        model=RobotKinematics(scene,next(a for a in scene['assets'] if a['kind']=='robot'))
        for q in [np.zeros(6),np.array([1.2,.5,-.4,.7,.2,-1.])]:
            links=model.links(q);values=passive_joint_values(scene,q)
            a=np.array(scene['joints'][6]['axis1']);b=np.array(scene['joints'][8]['axis1'])
            reconstructed246=links['241']@axis_matrix(a,[0.,1.,0.],values[0])
            reconstructed247=links['240']@axis_matrix(b,[0.,1.,0.],values[2])
            np.testing.assert_allclose(reconstructed246,links['246'],atol=1e-9)
            np.testing.assert_allclose(reconstructed247,links['247'],atol=1e-9)
            source_axis=np.array(scene['joints'][7]['axis2'])-np.array(scene['joints'][7]['axis1']);source_axis/=np.linalg.norm(source_axis)
            prism=np.eye(4);prism[:3,3]=source_axis*values[1]
            # Source prism axis is rounded 8.32e-7 rad from the exact pin span.
            np.testing.assert_allclose(reconstructed246@prism,reconstructed247,atol=2e-4)


if __name__ == '__main__':
    unittest.main()
