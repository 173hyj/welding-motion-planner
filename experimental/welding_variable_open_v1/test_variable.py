import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    import variable_core as core
    from variable_world import VariableToolWorld
    import variable_motion as motion
except ImportError:
    core = None
    VariableToolWorld = None
    motion = None


class VariableTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, 'variable opening implementation missing')

    def test_opening_prefix_cannot_cross_a_collision(self):
        visited = []
        def probe(driver):
            visited.append(driver)
            return dict(passed=driver > -39., certificates={'e1': {'passed': True}, 'e2': {'passed': True}})
        profile = core.opening_profile(probe, ['e1', 'e2'], 185.)
        self.assertEqual(visited[-1], -39.)
        self.assertEqual(profile['last_passed_amplitude_mm'], 38.)
        self.assertTrue(core.prefix_available(profile, 35.))
        self.assertFalse(core.prefix_available(profile, 40.))

    def test_both_closed_electrodes_must_be_certified(self):
        profile = core.opening_profile(lambda d: dict(passed=True, certificates={'e1': {'passed': True}}), ['e1', 'e2'], 20.)
        self.assertFalse(core.prefix_available(profile, 10.))
        self.assertEqual(profile['checked_samples'], 1)

    def test_driver_samples_include_partial_endpoint(self):
        np.testing.assert_allclose(core.driver_samples(-2.5), [0., -.8333333333333333, -1.6666666666666667, -2.5])
        for invalid in (1., -186., float('nan')):
            with self.assertRaises(ValueError): core.driver_samples(invalid)

    def test_completed_resume_is_read_only(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'report.json'
            path.write_text(json.dumps(dict(manifest={'x':1},summary={'complete':True,'wall_s':123.})))
            before = path.stat().st_mtime_ns
            result = core.completed_report(path, {'x':1}, resume=True)
            self.assertEqual(result['summary']['wall_s'], 123.)
            self.assertEqual(path.stat().st_mtime_ns, before)
            with self.assertRaises(RuntimeError): core.completed_report(path, {'x':2}, resume=True)
            with self.assertRaises(FileExistsError): core.completed_report(path, {'x':1}, resume=False)

    def test_geometry_uses_requested_opening_everywhere(self):
        engine = VariableToolWorld.__new__(VariableToolWorld)
        calls = []
        engine.environment = SimpleNamespace(relative_env=[])
        engine.state = lambda rid, t, d, *args: (calls.append(d) or dict(blocking_pairs=[], touching=[]))
        engine.minimum_distance = lambda rid, t, d, env: (calls.append(d) or 10.)
        result = engine.straight_exit('r', np.eye(4), {}, dict(supports=[], initial_touching=[]), [1., 0., 0.], driver_mm=-35.)
        self.assertTrue(result['passed'])
        self.assertEqual(result['length_mm'], 100.)
        self.assertEqual(set(calls), {-35.})

    def test_motion_uses_partial_driver_and_current_world_direction(self):
        class Model:
            def tcp(self, q):
                t = np.eye(4); t[0, 3] = q[0]*1000.; return t
        engine = SimpleNamespace(world=SimpleNamespace(models={'r': Model()}, static=[]), delta=np.eye(4))
        engine.delta[:3,:3] = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        calls = []
        engine.target = lambda *args: ({}, np.eye(4))
        engine.state = lambda rid,t,d,*args: (calls.append(d) or dict(touching=[],blocking_pairs=[]))
        engine.release_context = lambda *args, **kw: (calls.append(kw['driver_mm']) or dict(initial_touching=[],supports=[]))
        engine.minimum_distance = lambda rid,t,d,env: (calls.append(d) or 10.)
        selected = dict(q_rad=[0.],other_robot_id='other',other_q_rad=[0.],positioner_deg=65.)
        path = dict(direction_world=[1.,0.,0.],direction_frame='current_world',length_mm=100.)
        prefix = [dict(q_rad=[0.],driver_mm=d,phase='closed' if d==0 else 'opening',tcp_xyz_mm=[0.,0.,0.],travel_mm=0.) for d in core.driver_samples(-35.)]
        def probe(e,r,q,o,p,d,*args):
            calls.append(d); return dict(passed=True,touching=[])
        with patch.object(motion, 'full_probe', probe), patch.object(motion, 'continuous_ik', lambda m,t,q: dict(passed=True,q_rad=[t[0,3]/1000.])):
            result = motion.lift_path(engine, {}, {'robot_id':'r'}, selected, path, driver_mm=-35., opening_frames=prefix)
        self.assertTrue(result['passed'])
        self.assertEqual(set(calls), {-35.})
        np.testing.assert_allclose(result['direction_world'], [1.,0.,0.])
        self.assertTrue(all(f['driver_mm']==-35. for f in result['frames'] if f['travel_mm']>0))
        self.assertEqual(result['q_start_rad'], [0.])

    def test_report_rejects_cold_evidence_for_other_report(self):
        try:
            from delivery_report import validate_delivery
        except ImportError:
            validate_delivery = None
        self.assertIsNotNone(validate_delivery, 'variable report validator missing')
        with self.assertRaises(RuntimeError):
            validate_delivery({'summary':{'complete':True,'full_opening_baseline_matches':True,'any_amplitude_passed':2}},
                              {'report_sha256':'other','summary':{'all_passed':True,'reverified_retreats':2}},'expected')

    def test_independent_fk_mismatch_cannot_pass_cold_verification(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('variable_verify', Path(__file__).parent/'verify.py')
        verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify)
        self.assertTrue(hasattr(verify,'source_fk_agrees'), 'source FK agreement gate missing')
        self.assertTrue(verify.source_fk_agrees(1e-10))
        self.assertFalse(verify.source_fk_agrees(.001))
        self.assertFalse(verify.source_fk_agrees(float('nan')))


if __name__ == '__main__': unittest.main()
