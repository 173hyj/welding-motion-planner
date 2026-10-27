"""Render the accepted source-FK route into an MP4 without changing route data."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/home/ubuntu22/shanghangfa_migration_20261009/trajopt-worktree')
sys.path.insert(0,str(ROOT))
import importlib.util
spec=importlib.util.spec_from_file_location('final_replay', '/home/ubuntu22/shanghangfa_migration_20261009/replay-retreat300-final.py')
final_replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(final_replay)
verify_replay=final_replay.verify_final
from experimental.trajopt_transfer_v1.replay_optimized import LiveFK,FullFactoryScene,apply_frame,interpolate_state
import mujoco

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--route',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0);p.add_argument('--fps',type=int,default=25);p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args();started=time.monotonic()
    audit=verify_replay(args.route)
    array=args.route/'route.npz';original=hashlib.sha256(array.read_bytes()).hexdigest()
    with np.load(array,allow_pickle=False) as a:
        states=a['states'];times=a['time'];phases=a['phases'];tasks=a['tasks']
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    fk=LiveFK(scene,audit['robot_ids']);builder=FullFactoryScene(scene,True)
    first=fk.frame(states[0]);model=builder.compile(first,audit['order']);data=mujoco.MjData(model)
    model.vis.quality.offsamples=0
    dynamic_bodies={b['body_id'] for b in builder.bindings if b['path'][0] in ('robots','guns')}
    moving_bodies={b['body_id'] for b in builder.bindings if b['body_id'] not in dynamic_bodies}
    model.geom_group[:]=0
    model.geom_group[np.isin(model.geom_bodyid,list(moving_bodies))]=2
    model.geom_group[np.isin(model.geom_bodyid,list(dynamic_bodies))]=1
    static_mocaps=[b['mocap_id'] for b in builder.bindings if b['body_id'] not in dynamic_bodies]
    camera=mujoco.MjvCamera();camera.lookat[:]=builder.camera_center
    camera.distance=max(3.,builder.camera_extent*.85);camera.azimuth=135.;camera.elevation=-22.
    option=mujoco.MjvOption();option.sitegroup[3]=0
    static_option=mujoco.MjvOption();static_option.sitegroup[3]=0;static_option.geomgroup[:]=0;static_option.geomgroup[2]=1
    fixed_option=mujoco.MjvOption();fixed_option.sitegroup[:]=0;fixed_option.geomgroup[:]=0;fixed_option.geomgroup[0]=1
    dynamic_option=mujoco.MjvOption();dynamic_option.sitegroup[:]=0;dynamic_option.geomgroup[:]=0;dynamic_option.geomgroup[1]=1
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    total=math.ceil(float(times[-1])*args.fps)+1
    if args.max_frames:total=min(total,args.max_frames)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    log=args.out.with_suffix('.ffmpeg.log')
    command=['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r',str(args.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
        '-crf','20','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(args.out)]
    print('MODEL_READY','TRIANGLES',builder.manifest['compiled_unique_triangles'],'FRAMES',total,flush=True)
    render_started=time.monotonic()
    backgrounds=OrderedDict();comparisons=[];background_renders=0
    with log.open('wb') as error:
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=error)
        try:
            with mujoco.Renderer(model,height=720,width=1280) as renderer:
                def draw_scene(visibility,depth=False):
                    renderer.update_scene(data,camera=camera,scene_option=visibility)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                    if depth:renderer.enable_depth_rendering()
                    else:renderer.disable_depth_rendering()
                    return renderer.render().copy()
                fixed_background=None
                for i in range(args.start_frame,total):
                    t=min(float(times[-1]),i/args.fps)
                    apply_frame(model,data,builder,fk.frame(interpolate_state(times,states,t)))
                    key=(data.mocap_pos[static_mocaps].tobytes(),data.mocap_quat[static_mocaps].tobytes())
                    if fixed_background is None:
                        fixed_background=(draw_scene(fixed_option),draw_scene(fixed_option,True))
                    if key not in backgrounds:
                        moving_rgb=draw_scene(static_option);moving_depth=draw_scene(static_option,True)
                        fixed_rgb,fixed_depth=fixed_background
                        backgrounds[key]=(np.where((moving_depth<fixed_depth)[...,None],moving_rgb,fixed_rgb),np.minimum(moving_depth,fixed_depth))
                        background_renders+=1
                        if len(backgrounds)>16:backgrounds.popitem(last=False)
                    backgrounds.move_to_end(key)
                    background,background_depth=backgrounds[key]
                    foreground=draw_scene(dynamic_option);foreground_depth=draw_scene(dynamic_option,True)
                    pixels=np.where((foreground_depth<background_depth)[...,None],foreground,background)
                    if i in (args.start_frame,round(28.4*args.fps),round(200*args.fps),total-1):
                        full=draw_scene(option);delta=abs(full.astype(int)-pixels.astype(int))
                        comparison=dict(frame=i,mean_channel_error=float(delta.mean()),maximum_channel_error=int(delta.max()),
                                        differing_pixels=int(np.any(delta>0,axis=2).sum()))
                        comparisons.append(comparison);print('FULL_FRAME_COMPARISON',comparison,flush=True)
                        if delta.mean()>.2:raise RuntimeError('Cached rendering differs from full source rendering')
                    image=Image.fromarray(pixels)
                    overlay=Image.new('RGBA',image.size,(0,0,0,0));draw=ImageDraw.Draw(overlay)
                    draw.rectangle((0,0,1280,112),fill=(0,0,0,190))
                    draw.text((15,8),f'300 mm retreat | 5 rotations | {t:.2f} / {times[-1]:.2f} s | 1x | display timing',font=font,fill='white')
                    index=max(0,int(np.searchsorted(times,t,side='right'))-1)
                    for arm in (0,1):
                        task=int(tasks[index,arm]);phase=str(phases[index,arm])
                        label=audit['order'][task-1] if task else ('transfer' if phase=='transfer' else 'waiting')
                        draw.text((15,36+arm*26),f'R{arm+1} | {phase} | {label}',font=font,
                                  fill=(255,220,70,255) if phase=='weld' else (255,255,255,255))
                    draw.text((1030,87),f'Positioner {np.rad2deg(states[index,12]):.1f} deg',font=font,fill='white')
                    image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
                    encoder.stdin.write(image.tobytes())
                    if i in (args.start_frame,round(28.4*args.fps),round(200*args.fps),total-1):
                        image.save(args.out.with_name(args.out.stem+f'-frame-{i}.png'))
                    if i%100==0:print('VIDEO_FRAME',i,total,'ELAPSED',round(time.monotonic()-render_started,1),flush=True)
        finally:
            encoder.stdin.close()
        code=encoder.wait()
        if code:raise RuntimeError(f'Video encoder failed: {code}; {log}')
    if hashlib.sha256(array.read_bytes()).hexdigest()!=original:raise RuntimeError('Route data changed during render')
    meta=dict(complete=True,source_array_sha256=original,source_audit=str(args.route/'audit.json'),
        route_duration_s=float(times[-1]),start_frame=args.start_frame,fps=args.fps,frames=total-args.start_frame,real_time_speed=1.,width=1280,height=720,
        display_only=True,full_factory=True,source_fk_live=True,scene=builder.manifest,
        render_comparisons=comparisons,background_renders=background_renders,shadows=False,antialias_samples=0,
        elapsed_s=time.monotonic()-started,mp4_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('VIDEO_COMPLETE',args.out,'RENDER_FPS',total/(time.monotonic()-render_started),flush=True)

if __name__=='__main__':main()

