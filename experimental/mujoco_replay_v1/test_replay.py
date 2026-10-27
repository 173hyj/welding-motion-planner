import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from timeline import audited_plan, slerp_many
from replay import Playback
from replay import draw_motion,configure_camera
from focus import MotionFocus
import mujoco
from types import SimpleNamespace


class AuditTests(unittest.TestCase):
    def test_relaxed_plan_uses_separate_path_bound_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'build/eight_hour_v17/audit').mkdir(parents=True)
            strict = root / 'build/eight_hour_v17/audit/plan_overrides.json'
            strict.write_text(json.dumps({'plans': {}}))
            experiment = root / 'build/welding_29_relaxed_v1'
            experiment.mkdir()
            raw = json.dumps(dict(frames=[dict(time_s=0), dict(time_s=1)])).encode()
            path = experiment / 'test.json.gz'
            path.write_bytes(gzip.compress(raw))
            (experiment / 'audit.json').write_text(json.dumps({'plans': {path.name: dict(
                sha256=hashlib.sha256(raw).hexdigest(), relative_path=str(path.relative_to(root)),
                status={'relaxed_interference_mode': True})}}))
            audited_plan(root,path)
            wrong=root/'build/test.json.gz';wrong.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError,'path|registered'):
                audited_plan(root,wrong)
            self.assertEqual(json.loads(strict.read_text()),{'plans':{}})

    def test_decoded_gzip_hash_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'build/eight_hour_v17/audit').mkdir(parents=True)
            raw = json.dumps(dict(frames=[dict(time_s=0), dict(time_s=1)])).encode()
            path = root / 'build/plan.json.gz'
            path.write_bytes(gzip.compress(raw))
            audit = {'plans': {path.name: {'sha256': hashlib.sha256(raw).hexdigest()}}}
            (root / 'build/eight_hour_v17/audit/plan_overrides.json').write_text(json.dumps(audit))
            _, metadata = audited_plan(root, path)
            self.assertEqual(metadata['frame_count'], 2)
            path.write_bytes(gzip.compress(raw + b' '))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                audited_plan(root, path)

    def test_slerp_avoids_quaternion_sign_jump(self):
        a = np.array([[1., 0, 0, 0]])
        np.testing.assert_allclose(slerp_many(a, -a, .5), a)
        b = np.roll(Rotation.from_euler('z', 90, degrees=True).as_quat(), 1)[None]
        q = slerp_many(a, b, .5)
        angle = Rotation.from_quat(np.roll(q[0], -1)).as_euler('xyz', degrees=True)[2]
        self.assertAlmostEqual(angle, 45)


class PlaybackTests(unittest.TestCase):
    def test_phase_navigation_focus_and_repeated_cycle(self):
        player = Playback(np.array([0., .5, 1.]), phases=['open', 'transfer', 'close'])
        player.command(78)
        self.assertEqual(player.time, .5)
        self.assertFalse(player.playing)
        player.command(78)
        self.assertEqual(player.time, 1.)
        player.command(66)
        self.assertEqual(player.time, .5)
        player.command(70)
        self.assertEqual(player.focus, 1)
        player.command(82)
        player.advance(2.25)
        self.assertEqual(player.cycles, 2)
        self.assertEqual(player.time, .25)

    def test_pause_step_reset_and_loop(self):
        player = Playback(np.array([0., .5, .5, 1.]))
        player.advance(.3)
        self.assertAlmostEqual(player.time, .3)
        player.command(32)
        player.advance(.2)
        self.assertAlmostEqual(player.time, .3)
        player.command(262)
        self.assertAlmostEqual(player.time, .5)
        player.command(263)
        self.assertAlmostEqual(player.time, 0.)
        player.command(82)
        self.assertTrue(player.playing)
        player.advance(1.2)
        self.assertAlmostEqual(player.time, .2)


class MotionFocusTests(unittest.TestCase):
    def test_default_camera_observes_workpiece_from_robot_side(self):
        def asset(kind,point):
            transform=np.eye(4);transform[:3,3]=point
            return dict(kind=kind,transform=transform.ravel().tolist())
        # Source Y is up. Robots sit on the +Z side of the workpiece.
        builder=SimpleNamespace(camera_center=np.array([0.,1.,1.]),camera_extent=3.,
                                scene=dict(assets=[asset('robot',[0,1000,0])],
                                           welds=[dict(world_xyz=[0,1000,-1000])]))
        camera=mujoco.MjvCamera()
        configure_camera(camera,builder)
        self.assertAlmostEqual(camera.azimuth,90.)
        model=mujoco.MjModel.from_xml_string('<mujoco/>')
        data=mujoco.MjData(model);scene=mujoco.MjvScene(model,100)
        mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),mujoco.MjvPerturb(),camera,
                              mujoco.mjtCatBit.mjCAT_ALL,scene)
        offset=scene.camera[0].pos-camera.lookat
        self.assertLess(offset[1],0) # robot source +Z maps to MuJoCo -Y
        configure_camera(camera,builder,view=2)
        self.assertAlmostEqual(camera.azimuth,270.)

    def test_drawn_trail_stops_at_current_frame_without_future_chords(self):
        frames=[]
        for i in range(6):
            pose=np.eye(4);pose[:3,3]=[i*10,i*i*10,0]
            frames.append(dict(time_s=float(i),phase='transfer',robots={
                'R1':dict(q_rad=[i/10]*6,tcp_matrix=pose.ravel().tolist())}))
        motion=MotionFocus(frames)
        model=mujoco.MjModel.from_xml_string('<mujoco/>')
        scene=mujoco.MjvScene(model,maxgeom=20)
        draw_motion(scene,motion,motion.tcp[3],frame_index=3)
        self.assertEqual(scene.ngeom,4) # three adjacent saved segments + TCP marker
        draw_motion(scene,motion,motion.tcp[-1])
        self.assertEqual(scene.ngeom,6)
        with self.assertRaisesRegex(ValueError,'refusing sparse chord'):
            draw_motion(mujoco.MjvScene(model,maxgeom=2),motion,motion.tcp[-1])

    def test_trajectory_keeps_all_saved_tcp_samples_and_bends(self):
        frames=[]
        for i in range(701):
            pose=np.eye(4)
            pose[:3,3]=[i,100*np.sin(i/20),i/10]
            frames.append(dict(time_s=i/30,phase='transfer',task_index=1,
                               robots={'R1':dict(q_rad=[i/1000]*6,tcp_matrix=pose.ravel().tolist())}))
        motion=MotionFocus(frames)
        self.assertEqual(len(motion.path),len(frames))
        np.testing.assert_array_equal(motion.path,motion.tcp)
        # The sharp maximum between sparse samples must be retained.
        self.assertAlmostEqual(motion.path[31,2],.1*np.sin(31/20))

    def test_identifies_active_robot_and_true_mm_path(self):
        frames = []
        for i in range(2):
            moving, fixed = np.eye(4), np.eye(4)
            moving[:3, 3] = [2000, 3000 + 100 * i, 4000]
            frames.append(dict(time_s=float(i), phase=['open', 'transfer'][i], robots={
                'R1': dict(q_rad=[0, .1 * i, 0, 0, 0, 0], tcp_matrix=moving.ravel().tolist()),
                'R2': dict(q_rad=[0] * 6, tcp_matrix=fixed.ravel().tolist())}))
        focus = MotionFocus(frames)
        self.assertEqual(focus.robot_id, 'R1')
        np.testing.assert_allclose(focus.tcp[1], [2, -4, 3.1])
        np.testing.assert_allclose(focus.summary['tcp_range_mm'], [0, 100, 0])
        self.assertAlmostEqual(focus.current(1, 1, 0)['tcp_displacement_mm'], 100)
        self.assertEqual(focus.summary['fixed_robot_ids'], ['R2'])


if __name__ == '__main__':
    unittest.main()
