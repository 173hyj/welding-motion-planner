"""Display-only simultaneous source-FK replay in the complete original factory."""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).absolute().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route',required=True,help='Directory containing route.npz and audit.json')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--start-time',type=float,default=0.)
    parser.add_argument('--paused',action='store_true')
    parser.add_argument('--gui-test-seconds',type=float,default=0.)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    sys.path[:0]=[str(ROOT),str(Path(__file__).absolute().parent)]
    from export_parallel import verify_replay,sha
    folder=Path(args.route).absolute();audit=verify_replay(folder)
    if not np.isfinite(args.speed) or args.speed<=0 or not np.isfinite(args.start_time):
        raise ValueError('Finite start time and positive speed required')
    # Reuse the original robot/gun FK, original full factory scene and binding
    # application, without altering the established viewer entry point.
    from experimental.trajopt_transfer_v1.replay_optimized import (
        LiveFK,FullFactoryScene,apply_frame,draw_paths,SOURCE_TO_MUJOCO,interpolate_state)
    import mujoco
    from mujoco import viewer as native_viewer
    factory_file=ROOT/'build/welding_full_factory_v109/report.json'
    factory=json.loads(factory_file.read_text())
    if not factory['complete'] or not factory['full_factory_not_cropped'] or sha(factory_file.parent/'factory.glb')!=factory['output_sha256']:
        raise ValueError('Complete unchanged original factory required')
    with np.load(folder/'route.npz',allow_pickle=False) as arrays:
        states=arrays['states'];times=arrays['time'];phases=arrays['phases'];tasks=arrays['tasks']
        stages=arrays['stage'];tcp=arrays['tcp']@SOURCE_TO_MUJOCO.T/1000.
    play_time=float(np.clip(args.start_time,times[0],times[-1]))
    at=max(0,int(np.searchsorted(times,play_time,side='right'))-1)
    source=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    fk=LiveFK(source,audit['robot_ids']);builder=FullFactoryScene(source,True)
    initial=fk.frame(interpolate_state(times,states,play_time))
    model=builder.compile(initial,audit['order']);data=mujoco.MjData(model)
    apply_frame(model,data,builder,initial)
    commands=queue.SimpleQueue();playing=not args.paused;show_path=True;speed=args.speed
    first=data.xpos[[b['body_id'] for b in builder.bindings]].copy()
    max_displacement=0.;syncs=0;seen=[set(),set()]
    started=previous=time.monotonic();existing_threads=set(threading.enumerate())
    print('OPENING_DUAL_PARALLEL_MUJOCO | Space pause | R replay | N/B task | T paths | +/- speed',flush=True)
    with native_viewer.launch_passive(model,data,key_callback=commands.put,show_left_ui=False,show_right_ui=False) as viewer:
        viewer_threads=[t for t in threading.enumerate() if t not in existing_threads and '_launch_internal' in t.name]
        with viewer.lock():
            viewer.cam.lookat[:]=builder.camera_center;viewer.cam.distance=max(3.,builder.camera_extent*.85)
            viewer.cam.azimuth=135.;viewer.cam.elevation=-22.;viewer.opt.sitegroup[3]=0
        while viewer.is_running():
            now=time.monotonic()
            while not commands.empty():
                key=commands.get()
                if key==32:playing=not playing
                elif key==82:play_time=float(times[0]);playing=True
                elif key==84:show_path=not show_path
                elif key in (61,334):speed=min(16.,speed*2)
                elif key in (45,333):speed=max(.125,speed/2)
                elif key in (78,66):
                    active=tasks[max(0,min(len(times)-1,int(np.searchsorted(times,play_time,side='right'))-1))]
                    current=max(active.tolist())
                    target=int(np.clip(current+(1 if key==78 else -1),1,21))
                    indexes=np.flatnonzero(np.any(tasks==target,axis=1))
                    if len(indexes):play_time=float(times[indexes[0]]);playing=False
            if playing:play_time=min(float(times[-1]),play_time+(now-previous)*speed)
            if play_time>=times[-1]:playing=False
            previous=now;index=max(0,min(len(times)-1,int(np.searchsorted(times,play_time,side='right'))-1))
            state=interpolate_state(times,states,play_time);frame=fk.frame(state)
            with viewer.lock():
                apply_frame(model,data,builder,frame)
                if show_path:draw_paths(viewer.user_scn,tcp,index)
                else:viewer.user_scn.ngeom=0
                current=data.xpos[[b['body_id'] for b in builder.bindings]]
                max_displacement=max(max_displacement,float(np.max(np.linalg.norm(current-first,axis=1))))
            labels=[]
            for arm in (0,1):
                task=int(tasks[index,arm]);phase=str(phases[index,arm]);seen[arm].add(phase)
                weld=audit['order'][task-1] if task else 'idle'
                labels.append(f'R{arm+1}: task {task}/21 | {phase} | {weld}')
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'Shanghangfa dual-arm parallel | display only\nR1\nR2\nTime / stage\nPositioner\nControls\nValidation',
                'Original source colors | full factory\n'+labels[0]+'\n'+labels[1]+'\n'+
                f'{play_time:.1f}/{times[-1]:.1f}s {speed:g}x {"PLAY" if playing else "PAUSE"} | {stages[index]}\n'+
                f'{np.rad2deg(state[12]):.2f} deg | two arms + actual gun drivers\n'+
                'Space R N/B T +/- | mouse camera\n21 preserved weld actions | fine sampled, display timing')])
            viewer.sync();syncs+=1
            if args.snapshot and syncs==20:
                from PIL import Image
                args.snapshot.parent.mkdir(parents=True,exist_ok=True)
                with viewer.lock():
                    renderer=mujoco.Renderer(model,height=900,width=1440)
                    renderer.update_scene(data,camera=viewer.cam)
                    Image.fromarray(renderer.render()).save(args.snapshot);renderer.close()
            if args.gui_test_seconds and now-started>=args.gui_test_seconds:break
            time.sleep(max(0.,1/60.-(time.monotonic()-now)))
    for thread in viewer_threads:
        thread.join(timeout=10.)
        if thread.is_alive():raise RuntimeError('Viewer OpenGL thread failed shutdown')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(dict(native_viewer_opened=True,syncs=syncs,
            max_body_displacement_m=max_displacement,phases_seen=[sorted(x) for x in seen],
            source_fk_live=True,display_only=True,full_factory=True,scene=builder.manifest,
            final_play_time_s=play_time),ensure_ascii=False,indent=2))
    if args.gui_test_seconds and (syncs<2 or (not args.paused and max_displacement<1e-5)):
        raise ValueError('Viewer failed to show motion')

if __name__=='__main__':main()
