"""Native MuJoCo replay of hash-audited robot and welding-gun matrices."""
import argparse
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from scene import ROOT, SceneBuilder
from timeline import Timeline, audited_plan, pair_at
from focus import MotionFocus

DEFAULT_PLAN = 'build/welding_plan_point7_to_point9_negative_sequence_v3.json'
STAGED_PLAN = 'build/welding_29_staged_v2/welding_plan_29_staged_v2.json.gz'
CORRECTED_SCENE='build/scene_placement_corrected_v1.json'


class Playback:
    def __init__(self, times, paused=False, speed=1., loop=True, phases=None):
        self.times = np.asarray(times, dtype=float)
        self.duration = float(self.times[-1])
        self.time = 0.
        self.playing = not paused
        self.speed = speed
        self.loop = loop
        self.show_welds = False
        self.view = 0
        self.closed = False
        self.focus = 0
        self.cycles = 0
        self.show_path = True
        self.phase_times = np.array([self.times[i] for i in range(len(self.times))
                                    if phases is not None and (i == 0 or phases[i] != phases[i-1])])

    def advance(self, elapsed):
        if self.playing:
            self.time += max(0., elapsed) * self.speed
            if self.time >= self.duration:
                if self.loop:
                    self.cycles += int(self.time / self.duration)
                    self.time %= self.duration
                else:
                    self.time = self.duration
                    self.playing = False

    def command(self, key):
        if key == 32:  # Space
            self.playing = not self.playing
        elif key == 82:  # R
            self.time, self.playing = 0., True
            self.cycles = 0
        elif key in (78, 66) and len(self.phase_times):  # N/B: next/previous phase
            i = (np.searchsorted(self.phase_times, self.time, side='right') if key == 78 else
                 np.searchsorted(self.phase_times, self.time, side='left') - 1)
            self.time = float(self.phase_times[np.clip(i, 0, len(self.phase_times) - 1)])
            self.playing = False
        elif key == 70:  # F: tool / arm / whole workcell
            self.focus = (self.focus + 1) % 3
        elif key == 84:
            self.show_path = not self.show_path
        elif key in (262, 263):  # Right / left: distinct source timestamps.
            unique = np.unique(self.times)
            i = (np.searchsorted(unique, self.time, side='right') if key == 262 else
                 np.searchsorted(unique, self.time, side='left') - 1)
            self.time = float(unique[np.clip(i, 0, len(unique) - 1)])
            self.playing = False
        elif key == 87:  # W
            self.show_welds = not self.show_welds
        elif key == 86:  # V
            self.view = (self.view + 1) % 3
        elif key in (45, 333):  # minus
            self.speed = max(.125, self.speed / 2.)
        elif key in (61, 334):  # plus
            self.speed = min(8., self.speed * 2.)
        elif key == 256:
            self.closed = True


def prepare(plan_path=DEFAULT_PLAN, include_layout=False, scene_path=None):
    plan, metadata = audited_plan(ROOT, plan_path)
    builder = SceneBuilder(include_layout=include_layout,scene_path=scene_path)
    model = builder.compile(plan['frames'][0], plan.get('selected_weld_ids', []))
    data = mujoco.MjData(model)
    timeline = Timeline(plan['frames'], builder.bindings)
    timeline.apply(model, data, 0.)
    return model, data, timeline, builder, metadata


def configure_camera(camera, builder, view=0, motion=None, focus=2):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = builder.camera_center
    camera.distance = max(1., builder.camera_extent * 1.04)
    # MuJoCo azimuth describes the viewing direction; camera position is on
    # the opposite horizontal side. Derive a robot-side view from source poses.
    azimuth=135.
    scene=getattr(builder,'scene',{})
    bases=[np.array(a['transform']).reshape(4,4)[:3,3] for a in scene.get('assets',[]) if a['kind']=='robot']
    welds=[w['world_xyz'] for w in scene.get('welds',[])]
    if bases and welds:
        from timeline import SOURCE_TO_MUJOCO
        robot_side=SOURCE_TO_MUJOCO@(np.mean(bases,axis=0)-np.mean(welds,axis=0))
        if np.linalg.norm(robot_side[:2])>1e-8:
            azimuth=float(np.rad2deg(np.arctan2(-robot_side[1],-robot_side[0])))
    camera.azimuth = [azimuth, azimuth, azimuth+180.][view]
    camera.elevation = [-24., -89., -15.][view]
    if motion is not None and focus < 2:
        if focus == 0:
            camera.lookat[:] = motion.center
            camera.distance = motion.distance
        else:
            selected = [b for b in builder.bodies if motion.robot_id in
                        next((x['path'] for x in builder.bindings if x['name']==b['name']), ())]
            points = np.array([np.array(b['source_matrix']).reshape(4,4)[:3,3] for b in selected])
            if len(points):
                from timeline import SOURCE_TO_MUJOCO
                points=points@SOURCE_TO_MUJOCO.T/1000.
                camera.lookat[:]=(points.min(axis=0)+points.max(axis=0))/2.
                camera.distance=max(2.5,float(np.linalg.norm(np.ptp(points,axis=0)))*.9)


def draw_motion(scene, motion, position, tasks=(), frame_index=None):
    scene.ngeom = 0
    identity = np.eye(3).reshape(-1)
    pairs=motion.path_pairs
    if frame_index is not None:pairs=pairs[pairs[:,1]<=frame_index]
    required=len(pairs)+2*len(tasks)+2
    if required>scene.maxgeom:
        raise ValueError(f'actual trajectory needs {required} geoms; capacity {scene.maxgeom}; refusing sparse chord replacement')
    for pair_index,(a_index,b_index) in enumerate(pairs):
        a,b=motion.path[a_index],motion.path[b_index]
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),identity,motion.path_colors[pair_index])
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.5,a,b)
        scene.ngeom+=1
    if frame_index is not None and np.linalg.norm(position-motion.path[frame_index])>=1e-8:
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),identity,np.array([1.,.6,.05,.8],dtype=np.float32))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.5,motion.path[frame_index],position)
        scene.ngeom+=1
    from timeline import SOURCE_TO_MUJOCO
    for task in tasks:
        for key,color in [('entry_tcp_matrix',[.1,.8,1.,1.]),('exit_tcp_matrix',[.1,1.,.4,1.])]:
            point=np.array(task[key]).reshape(4,4)[:3,3]@SOURCE_TO_MUJOCO.T/1000.
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.005]*3),point,identity,np.array(color,dtype=np.float32))
            scene.ngeom+=1
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.009]*3),position,identity,np.array([1.,.25,.1,1.],dtype=np.float32))
    scene.ngeom+=1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=STAGED_PLAN if (ROOT / STAGED_PLAN).exists() else DEFAULT_PLAN)
    parser.add_argument('--layout', action='store_true', help='also include full original 7.5M-triangle factory layout')
    parser.add_argument('--scene',default=CORRECTED_SCENE if (ROOT/CORRECTED_SCENE).exists() else 'build/scene.json',
                        help='scene placement file; corrected source world placements are preferred when available')
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--no-loop', action='store_true')
    parser.add_argument('--gui-test-seconds', type=float, default=0., help='bounded native viewer verification')
    parser.add_argument('--report', type=Path, help='optional GUI playback evidence JSON')
    parser.add_argument('--focus', choices=['tool','arm','workcell'], default='arm')
    parser.add_argument('--start-time', type=float, default=0.)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    model, data, timeline, builder, metadata = prepare(args.plan, args.layout,args.scene)
    import mujoco.viewer
    phases=[f.get('phase','') for f in timeline.frames]
    player = Playback(timeline.times, args.paused, args.speed, not args.no_loop, phases)
    player.focus=['tool','arm','workcell'].index(args.focus)
    if not np.isfinite(args.start_time) or not 0. <= args.start_time <= timeline.duration:
        parser.error('--start-time must be within the plan duration')
    player.time=args.start_time
    motion=MotionFocus(timeline.frames)
    plan,_=audited_plan(ROOT,args.plan)
    tasks=plan.get('tasks',[])
    summary=plan.get('summary',{})
    relaxed=summary.get('relaxed_interference_mode',False)
    commands = queue.SimpleQueue()
    started = previous = time.monotonic()
    loops, displayed_view, displayed_focus = 0, 0, player.focus
    phases_seen=set();tasks_seen=set();completed=0
    test_start = None
    first_positions = data.xpos[[b['body_id'] for b in builder.bindings]].copy()
    max_displacement, max_playback_time = 0., 0.
    print('Opening native MuJoCo. Space pause | R reset | arrows step | V view | W welds | +/- speed', flush=True)
    existing_threads=set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer_threads=[t for t in threading.enumerate() if t not in existing_threads
                        and '_launch_internal' in t.name]
        with viewer.lock():
            configure_camera(viewer.cam, builder, motion=motion, focus=player.focus)
            viewer.opt.sitegroup[:] = 0
        while viewer.is_running() and not player.closed:
            now = time.monotonic()
            if test_start is None:
                test_start = now
                previous = now
            while not commands.empty():
                player.command(commands.get())
            player.advance(now - previous)
            previous = now
            with viewer.lock():
                i, j, alpha = timeline.apply(model, data, player.time)
                info=motion.current(i,j,alpha)
                frame=timeline.frames[i]
                phases_seen.add(frame.get('phase',''));tasks_seen.add(frame.get('task_index',0))
                completed=max(completed,int(frame.get('completed_welds',0)))
                viewer.opt.sitegroup[3] = int(player.show_welds)
                if displayed_view != player.view or displayed_focus != player.focus:
                    configure_camera(viewer.cam, builder, player.view, motion, player.focus)
                    displayed_view = player.view
                    displayed_focus = player.focus
                if player.show_path:
                    draw_motion(viewer.user_scn,motion,info['position'],tasks,frame_index=i)
                else:viewer.user_scn.ngeom=0
                current = data.xpos[[b['body_id'] for b in builder.bindings]]
                max_displacement = max(max_displacement, float(np.max(np.linalg.norm(current - first_positions, axis=1))))
            max_playback_time = max(max_playback_time, player.time)
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'Shanghangfa | FANUC x2 | original GLB\nTime / phase\nPlayback\nControls\nTask status\nMotion\nJoint change deg\nInterference\nTransfer plan\nScene placement',
                               f'{len(timeline.frames)} source frames\n{player.time:.3f} / {timeline.duration:.3f}s | {timeline.frames[i].get("phase", "")}\n'
                               f'{"PLAY" if player.playing else "PAUSE"} | {player.speed:g}x | loop {player.cycles}\nSpace R arrows | N/B phase F focus T path V W +/-\n'
                               f'{"RELAXED29" if relaxed else "Local replay"} | task {frame.get("task_index",0)}/29 | welded {frame.get("completed_welds",0)}/29 | strict 0/29\n'
                               f'TCP delta {info["tcp_displacement_mm"]:.1f} mm | R2 fixed | {frame.get("weld_id", "")}\n'
                               + ' '.join(f'{x:+.1f}' for x in info['joint_delta_deg']) + '\n'
                               + (f'RELAXED | local max {tasks[frame["task_index"]-1].get("local_max_observed_depth_mm",tasks[frame["task_index"]-1]["maximum_observed_depth_mm"]):.2f} mm | system_safe=false'
                                  if relaxed and tasks and frame.get('task_index') else 'Archived local replay | system_safe=false')
                               + '\n' + (tasks[frame['task_index']-1].get('transfer_planner','archived')
                                         + (' | FAILED search: diagnostic fallback' if tasks[frame['task_index']-1].get('transfer_fallback_retained') else '')
                                         if tasks and frame.get('task_index') else 'archived')
                               + '\n' + ('Source world poses corrected | OLD COLLISION AUDIT STALE'
                                         if builder.manifest['placement_corrected'] else 'Archived relative-layout import'))])
            viewer.sync()
            loops += 1
            if args.gui_test_seconds and now - test_start >= args.gui_test_seconds:
                break
            time.sleep(max(0., 1. / 60. - (time.monotonic() - now)))
    # The passive viewer owns a daemon rendering thread. Keep model/data alive
    # until that thread has released its OpenGL resources before Python exits.
    for thread in viewer_threads:
        thread.join(timeout=10.)
        if thread.is_alive():
            raise RuntimeError('MuJoCo rendering thread did not finish shutdown')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = dict(native_viewer_opened=True, sync_count=loops, elapsed_s=time.monotonic() - started,
                      max_playback_time_s=max_playback_time, max_body_displacement_m=max_displacement,
                      moving_bodies=model.nmocap, plan=metadata, scene=builder.manifest,
                      phases_seen=sorted(phases_seen),tasks_seen=sorted(tasks_seen),maximum_completed_welds=completed,
                      motion_summary=motion.summary,loop_count=player.cycles,
                      source_plan_unchanged=hashlib.sha256(Path(metadata['path']).read_bytes()).hexdigest() == metadata['file_sha256'],
                      strict_planned_task_count=0, system_safe=False)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'GUI evidence written: {args.report}', flush=True)
    if args.gui_test_seconds and (loops < 2 or (not args.paused and max_displacement < 1e-5)):
        raise RuntimeError('native viewer did not demonstrate motion')


if __name__ == '__main__':
    main()
