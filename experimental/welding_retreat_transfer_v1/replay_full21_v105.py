"""Native source-FK MuJoCo replay of the full21, nonzero-positioner 1000mm route."""
import argparse
import hashlib
import json
import queue
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/mujoco_replay_v1'),str(ROOT/'experimental/welding_positioner_v1')]
from scene import SceneBuilder,world_pose
from timeline import SOURCE_TO_MUJOCO
from positioner import SourcePositioner
from welding_kinematics import RobotKinematics
from gun_kinematics import GunKinematics
from replay_full21_core_v105 import interpolate_state

DEFAULT='build/welding_full21_replay_v105'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class LiveFK:
    def __init__(self,scene,rids):
        self.scene=scene;self.robot_ids=rids;self.positioner=SourcePositioner(scene)
        self.models={};self.guns={};self.gun_models={}
        for rid in rids:
            asset=next(a for a in scene['assets'] if a['id']==rid)
            model=RobotKinematics(scene,asset);self.models[rid]=model
            gun=next(a for a in scene['assets'] if a['kind']=='gun' and ('R1' if 'R1' in asset['name'] else 'R2') in a['name'])
            self.guns[rid]=gun;self.gun_models[rid]=GunKinematics(json.loads((ROOT/gun['gun']['source_detail']).read_text()))
            model.gun_home=np.linalg.inv(model.links(model.q_initial,world=True)['245'])@np.asarray(gun['transform']).reshape(4,4)
        self.products={a['id']:a for a in scene['assets'] if a['kind']=='product'}

    def frame(self,state):
        robots={};guns={}
        for i,rid in enumerate(self.robot_ids):
            model=self.models[rid];q=state[6*i:6*i+6];links=model.links(q,world=True)
            robots[rid]=dict(links={k:v.ravel() for k,v in links.items()},tcp_matrix=model.tcp(q).ravel())
            gun=self.guns[rid];deltas=self.gun_models[rid].transforms(float(state[13+i]))
            guns[gun['id']]=dict(parts={pid:(links['245']@model.gun_home@deltas[part['link_key']]).ravel()
                for pid,part in gun['gun']['link_models'].items()})
        delta=self.positioner.delta(float(np.rad2deg(state[12])))
        assets={pid:(delta@np.asarray(a['transform']).reshape(4,4)).ravel() for pid,a in self.products.items()}
        assets[self.positioner.asset['id']+':DONG']=(delta@self.positioner.base).ravel()
        return dict(robots=robots,guns=guns,assets=assets,delta=delta)


class MovingScene(SceneBuilder):
    def __init__(self,source,include_layout=True):
        super().__init__(include_layout=include_layout,scene_path=ROOT/'build/scene_placement_corrected_v1.json')
        self.scene=json.loads((ROOT/'build/welding_positioner_static_v2/geometry_audit.json').read_text())['scene']
        self.source_scene=source

    def body(self,label,url,matrix,kind,path=None):
        if path is None:
            asset=next((a for a in self.scene['assets'] if a['name']==label),None)
            if asset and (kind=='product' or asset['id'].endswith(':DONG')):
                path=('assets',asset['id']);matrix=self.first_frame['assets'][asset['id']]
        return super().body(label,url,matrix,kind,path)

    def compile(self,first_frame,selected_welds=()):
        self.first_frame=first_frame
        return super().compile(first_frame,selected_welds)


def apply_frame(model,data,builder,frame):
    for binding in builder.bindings:
        value=frame
        for key in binding['path']:value=value[key]
        pos,quat=world_pose(value)
        data.mocap_pos[binding['mocap_id']]=pos;data.mocap_quat[binding['mocap_id']]=quat
    for i,weld in enumerate(builder.source_scene['welds']):
        point=frame['delta']@np.r_[weld['world_xyz'],1.]
        site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'weld_{i}')
        model.site_pos[site]=SOURCE_TO_MUJOCO@point[:3]/1000.
    mujoco.mj_forward(model,data)


def draw_paths(scn,tcp,index,window=1200):
    scn.ngeom=0;identity=np.eye(3).ravel()
    for arm,color in [(0,[.1,.8,1.,.9]),(1,[1.,.6,.1,.9])]:
        for i in range(max(0,index-window),index):
            a,b=tcp[i,arm],tcp[i+1,arm]
            if np.linalg.norm(b-a)<1e-8:continue
            if scn.ngeom>=scn.maxgeom:raise ValueError('Actual trajectory exceeds scene capacity')
            g=scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),identity,np.asarray(color,dtype=np.float32))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.5,a,b);scn.ngeom+=1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',default=DEFAULT);p.add_argument('--speed',type=float,default=3.)
    p.add_argument('--start-task',type=int,default=1);p.add_argument('--paused',action='store_true')
    p.add_argument('--no-layout',action='store_true');p.add_argument('--gui-test-seconds',type=float,default=0.)
    p.add_argument('--report',type=Path);args=p.parse_args()
    folder=Path(args.route);folder=folder if folder.is_absolute() else ROOT/folder
    audit=json.loads((folder/'audit.json').read_text())
    if not audit['full21_cold_verified'] or sha(folder/'route.npz')!=audit['array_sha256']:
        raise ValueError('Full21 route seal invalid')
    for n,h in audit['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Bound source changed '+n)
    cold=json.loads((ROOT/audit['cold_path']).read_text())
    if not cold['summary']['full_route_verified']:raise ValueError('Unverified route')
    with np.load(folder/'route.npz') as arrays:
        states=arrays['states'];times=arrays['times'];phases=arrays['phases'];tasks=arrays['tasks']
        tcp=arrays['tcp']@SOURCE_TO_MUJOCO.T/1000.
    if not 1<=args.start_task<=21 or not np.isfinite(args.speed) or args.speed<=0.:
        raise ValueError('Invalid start task/speed')
    at=int(np.flatnonzero(tasks==args.start_task)[0]);play_time=float(times[at])
    source=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    fk=LiveFK(source,audit['robot_ids']);builder=MovingScene(source,not args.no_layout)
    initial=fk.frame(states[at]);model=builder.compile(initial,audit['order']);data=mujoco.MjData(model)
    apply_frame(model,data,builder,initial)
    import mujoco.viewer
    commands=queue.SimpleQueue();playing=not args.paused;show_path=True;speed=args.speed
    first=data.xpos[[b['body_id'] for b in builder.bindings]].copy()
    max_displacement=0.;start_angle=float(states[at,12]);max_angle=0.;syncs=0;seen=set()
    started=previous=time.monotonic()
    print('OPENING_FULL21_MUJOCO | Space pause | R replay | N/B next/previous weld | T actual path | +/- speed',flush=True)
    with mujoco.viewer.launch_passive(model,data,key_callback=commands.put,show_left_ui=False,show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:]=builder.camera_center;viewer.cam.distance=max(3.,builder.camera_extent*.85)
            viewer.cam.azimuth=135.;viewer.cam.elevation=-22.;viewer.opt.sitegroup[3]=0
        while viewer.is_running():
            now=time.monotonic()
            while not commands.empty():
                key=commands.get()
                if key==32:playing=not playing
                elif key==82:play_time=0.;playing=True
                elif key==84:show_path=not show_path
                elif key in (61,334):speed=min(16.,speed*2)
                elif key in (45,333):speed=max(.125,speed/2)
                elif key in (78,66):
                    task=int(tasks[min(len(tasks)-1,int(np.searchsorted(times,play_time)))])
                    target=int(np.clip(task+(1 if key==78 else -1),1,21))
                    play_time=float(times[np.flatnonzero(tasks==target)[0]]);playing=False
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
            max_angle=max(max_angle,abs(float(state[12])-start_angle));seen.add(str(phases[index]))
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'Shanghangfa full21 | actual RRT route\nWeld / phase\nTime\nPositioner\nControls\nValidation',
                f'Original source colors | retreat 1000 mm\n{tasks[index]}/21 {phases[index]} | {audit["order"][int(tasks[index])-1]}\n'
                f'{play_time:.1f}/{times[-1]:.1f}s {speed:g}x {"PLAY" if playing else "PAUSE"}\n'
                f'{np.rad2deg(state[12]):.2f} deg | two arms + two guns\nSpace R N/B T +/- | mouse camera\n21 retreats +20 transfers cold passed | sampled')])
            viewer.sync();syncs+=1
            if args.gui_test_seconds and now-started>=args.gui_test_seconds:break
            time.sleep(max(0.,1/60.-(time.monotonic()-now)))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(dict(native_viewer_opened=True,syncs=syncs,
            max_body_displacement_m=max_displacement,max_positioner_change_deg=float(np.rad2deg(max_angle)),
            phases_seen=sorted(seen),source_fk_live=True,trajectory='Both source TCP paths, exact consecutive samples in local window',
            scene=builder.manifest,final_play_time_s=play_time),ensure_ascii=False,indent=2))
    if args.gui_test_seconds and (syncs<2 or (not args.paused and max_displacement<1e-5)):
        raise ValueError('Viewer failed to show motion')


if __name__=='__main__':main()
