"""Full source 29-point process planning with explicit soft interference cost.

All source and archived strict artifacts are read-only.  This is an independent
offline process demonstration, not a collision-safety certificate.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import fcl
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from collision_audit import CollisionWorld
from gun_kinematics import GunKinematics
from welding_kinematics import RobotKinematics
from weld_task_planner import SourceTiming, passive_joint_values
from contracts import legal_charts, quintic, validate_workflow, workflow_phases

OUTPUT = ROOT / 'build/welding_29_relaxed_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_error(model, q, target):
    pose = model.tcp(q)
    return np.r_[(pose[:3, 3]-target[:3, 3])/1000.,
                 Rotation.from_matrix(target[:3, :3].T@pose[:3, :3]).as_rotvec()]


def local_ik(model, target, seed):
    result = least_squares(lambda q: pose_error(model, q, target),
                           np.clip(seed, model.bounds[0]+1e-8, model.bounds[1]-1e-8),
                           bounds=(model.bounds[0], model.bounds[1]), max_nfev=80,
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    charts = legal_charts(model, result.x, seed)
    if not charts:
        return None
    q = charts[0]
    error = pose_error(model, q, target)
    if (np.linalg.norm(error[:3])*1000. > .05 or np.linalg.norm(error[3:]) > .001 or
            np.max(np.abs(q-seed)) > .25):
        return None
    return q


def normal_path(model, q, distance, sign):
    pose = model.tcp(q)
    normal = pose[:3, 2]*sign
    qs = [q.copy()]
    for travel in np.linspace(0., distance, max(3, int(np.ceil(distance/5.))+1))[1:]:
        target = pose.copy()
        target[:3, 3] += normal*travel
        next_q = local_ik(model, target, qs[-1])
        if next_q is None:
            return None
        qs.append(next_q)
    qs = np.array(qs)
    spline = CubicSpline(np.linspace(0., 1., len(qs)), qs)
    max_line, max_rotation = 0., 0.
    for fraction in np.linspace(0, 1, max(51, len(qs)*3)):
        qi = spline(fraction)
        if not model.limits_valid(qi):
            return None
        actual = model.tcp(qi)
        expected = pose[:3, 3] + normal*distance*fraction
        max_line = max(max_line, float(np.linalg.norm(actual[:3, 3]-expected)))
        max_rotation = max(max_rotation, float(np.linalg.norm(Rotation.from_matrix(pose[:3, :3].T@actual[:3, :3]).as_rotvec())))
    if max_line > .1 or max_rotation > .001:
        return None
    return dict(qs=qs.tolist(), distance_mm=float(distance), sign=float(sign),
                normal_world=normal.tolist(), line_error_mm=max_line,
                rotation_error_rad=max_rotation, endpoint_tcp=model.tcp(qs[-1]).reshape(-1).tolist())


class Interference:
    """Exact source triangle BVH witnesses; depth is sampled FCL contact depth.

    Contact caps explicitly limit depth evidence.  No claim of full volumetric
    penetration or continuous clearance follows from these finite probes.
    """
    def __init__(self, world, rid, reference_mm=2.):
        self.world, self.rid, self.reference = world, rid, reference_mm
        self.cache = {}
        self.count = 0

    def check(self, q, driver, details=True):
        key = (tuple(np.round(q, 9)), round(float(driver), 6))
        if key in self.cache:
            return self.cache[key]
        world, rid = self.world, self.rid
        world.update(rid, q, float(driver))
        objects = world.robots[rid]
        other = [(rr+':'+k, 'other_robot', o) for rr, obs in world.robots.items() if rr != rid for k, o in obs.items()]
        pairs = []
        for link, obj in objects.items():
            if link == '239':
                continue
            for oid, name, environment in world.static + other:
                pairs.append((link, obj, oid, name, environment))
        numbers = [k for k in objects if k.isdigit()]
        mounts = {tuple(sorted(p)) for p in [('239','240'),('240','241'),('241','242'),
                  ('242','243'),('243','244'),('244','245'),('240','246'),('241','246'),
                  ('240','247'),('241','247'),('246','247')]}
        for a, b in itertools.combinations(numbers, 2):
            if tuple(sorted((a,b))) not in mounts:
                pairs.append((a,objects[a],b,'robot_self',objects[b]))
        for a in numbers:
            if a == '245':
                continue
            for b in (k for k in objects if k.startswith('gun:')):
                pairs.append((a,objects[a],b,'gun_robot_self',objects[b]))
        witnesses = []
        for link,obj,oid,name,environment in pairs:
            result = fcl.CollisionResult()
            fcl.collide(obj, environment, fcl.CollisionRequest(num_max_contacts=8, enable_contact=True), result)
            if result.is_collision:
                depths = [float(c.penetration_depth) for c in result.contacts]
                depth = max(depths, default=0.)
                witnesses.append(dict(link=link, asset_id=oid, asset_name=name,
                                      observed_depth_mm=depth, contacts_sampled=len(depths),
                                      contact_cap_reached=len(depths)>=8,
                                      contact_xyz=result.contacts[0].pos.tolist() if result.contacts else None))
        maximum = max((x['observed_depth_mm'] for x in witnesses), default=0.)
        # Soft objective, with a strong cost for intersections larger than the
        # reporting reference; no endpoint is silently declared collision-free.
        penalty = sum(.3 + min(x['observed_depth_mm'], 100.)*.05 +
                      2.*max(0., min(x['observed_depth_mm'], 100.)-self.reference)**2 for x in witnesses)
        record = dict(pairs=witnesses, pair_count=len(witnesses), observed_max_depth_mm=maximum,
                      soft_cost=penalty, above_reference=maximum>self.reference,
                      contact_caps_reached=sum(x['contact_cap_reached'] for x in witnesses))
        self.cache[key] = record
        self.count += 1
        return record


def continuous_joint_path(model, qa, qb):
    qa, qb = np.asarray(qa), np.asarray(qb)
    if all(model.limits_valid(qa+t*(qb-qa)) for t in np.linspace(0.,1.,81)):
        return [qa,qb]
    # Preserve all endpoints while routing the dependent J2/J3 pair inside the
    # exact source polygon; constant branch values for the other axes.
    for q2,q3 in [(0.,-.2),(.2,-.2),(.4,-.4),(-.3,0.)]:
        mid = (qa+qb)/2.;mid[1:3]=[q2,q3]
        if (all(model.limits_valid(qa+t*(mid-qa)) for t in np.linspace(0,1,81)) and
                all(model.limits_valid(mid+t*(qb-mid)) for t in np.linspace(0,1,81))):
            return [qa,mid,qb]
    return None


class Planner:
    def __init__(self, out, reference_mm=2., distance=100., dwell=.8, fps=25.):
        self.out = out;out.mkdir(parents=True,exist_ok=True)
        self.scene=json.loads((ROOT/'build/scene.json').read_text())
        self.inventory=json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
        self.source_files=[ROOT/'build/scene.json',ROOT/'build/robot.json',ROOT/'welding_kinematics.py',ROOT/'gun_kinematics.py',
                           ROOT/'build/eight_hour_v17/audit/plan_overrides.json',
                           ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json']
        self.hashes={str(p.relative_to(ROOT)):digest(p) for p in self.source_files}
        self.world=CollisionWorld(self.scene)
        self.rid=next(a['id'] for a in self.scene['assets'] if a['kind']=='robot' and 'R1' in a['name'])
        self.other=next(rid for rid in self.world.models if rid!=self.rid)
        self.model=self.world.models[self.rid]
        self.park=np.array(json.loads((ROOT/'build/r2_safe_park_search.json').read_text())['selected']['q_rad'])
        if not self.world.models[self.other].limits_valid(self.park):
            raise ValueError('archived alternative park violates limits')
        self.world.update(self.other,self.park,0.)
        self.cost=Interference(self.world,self.rid,reference_mm)
        self.timing=SourceTiming(self.scene,self.model.gun_model)
        self.open=float(self.model.gun_model.limits_mm[0])
        self.distance,self.dwell,self.fps=distance,dwell,fps
        self.frames,self.segments,self.tasks=[],[],[]
        self.at=0.

    def candidate_pool(self):
        pool=[]
        for w,row in zip(self.scene['welds'],self.inventory['rows']):
            valid=[]
            for c in row['candidates']:
                if c['robot_id']!=self.rid or not c['fk_candidate_passed']:
                    continue
                for q in legal_charts(self.model,np.array(c['q_rad']),self.model.q_initial):
                    if any(np.linalg.norm(q-np.array(x['q_rad']))<1e-4 for x in valid):continue
                    if np.linalg.norm(self.model.tcp(q)[:3,3]-np.array(w['world_xyz']))>.1:continue
                    closed=self.cost.check(q,0.);opened=self.cost.check(q,self.open)
                    valid.append(dict(q_rad=q.tolist(),candidate_index=c['candidate_index'],
                                      endpoint_closed=closed,endpoint_open=opened,
                                      score=closed['soft_cost']+opened['soft_cost']))
            valid.sort(key=lambda x:x['score'])
            usable=[]
            # Use all saved posture families if necessary to obtain real ingress
            # and egress states. Search both signs before shortening the stroke.
            for candidate in valid:
                if len(usable)>=6:break
                q=np.array(candidate['q_rad']);paths=[]
                for distance in dict.fromkeys([self.distance,75.,50.,30.]):
                    if distance>self.distance:continue
                    for sign in (-1.,1.):
                        path=normal_path(self.model,q,distance,sign)
                        if path is not None:
                            checks=[self.cost.check(np.array(path['qs'])[i],self.open) for i in
                                    np.unique(np.linspace(0,len(path['qs'])-1,5).astype(int))]
                            path['path_soft_cost']=float(np.mean([c['soft_cost'] for c in checks]))
                            path['path_max_observed_depth_mm']=max(c['observed_max_depth_mm'] for c in checks)
                            paths.append(path)
                    if paths:break
                if not paths:continue
                path=min(paths,key=lambda x:x['path_soft_cost'])
                candidate=dict(candidate,stroke=path,
                               score=candidate['score']+path['path_soft_cost'])
                usable.append(candidate)
            if not usable:raise RuntimeError(f'no complete legal ingress/egress IK at {w["id"]}')
            usable.sort(key=lambda x:x['score']);pool.append(usable)
            print(f'POOL {w["id"]}: {len(usable)} complete choices; depth closed/open '
                  f'{usable[0]["endpoint_closed"]["observed_max_depth_mm"]:.3f}/'
                  f'{usable[0]["endpoint_open"]["observed_max_depth_mm"]:.3f} mm; stroke {usable[0]["stroke"]["distance_mm"]}',flush=True)
            (self.out/'candidate_progress.json').write_text(json.dumps(dict(source_hashes=self.hashes,completed=len(pool),pool=pool),ensure_ascii=False))
        return pool

    def choose_sequence(self,pool):
        scores=np.array([c['score'] for c in pool[0]])
        parents=[];edges=[]
        for index in range(1,len(pool)):
            next_scores=np.full(len(pool[index]),np.inf);parent=np.full(len(pool[index]),-1,dtype=int);edge_map={}
            for j,current in enumerate(pool[index]):
                qb=np.array(current['stroke']['qs'][-1])
                for k,previous in enumerate(pool[index-1]):
                    qa=np.array(previous['stroke']['qs'][-1])
                    route=continuous_joint_path(self.model,qa,qb)
                    if route is None:continue
                    # Coarse true-mesh transfer probes guide posture selection.
                    edge_cost=0.
                    for a,b in zip(route,route[1:]):
                        edge_cost+=sum(self.cost.check(a+t*(b-a),self.open)['soft_cost'] for t in [.25,.5,.75])/3.
                    travel=sum(np.linalg.norm(b-a) for a,b in zip(route,route[1:]))
                    score=scores[k]+current['score']+edge_cost+3.*travel
                    edge_map[(k,j)]=[x.tolist() for x in route]
                    if score<next_scores[j]:next_scores[j]=score;parent[j]=k
            if not np.any(np.isfinite(next_scores)):raise RuntimeError(f'no legal joint connection into source weld {index+1}')
            print(f'CONNECT {index+1}/29 choices {np.sum(np.isfinite(next_scores))}',flush=True)
            parents.append(parent);edges.append(edge_map);scores=next_scores
        j=int(np.argmin(scores));indices=[j]
        for parent in reversed(parents):j=int(parent[j]);indices.append(j)
        indices=indices[::-1]
        choices=[pool[i][j] for i,j in enumerate(indices)]
        routes=[edges[i][(indices[i],indices[i+1])] for i in range(len(edges))]
        return choices,routes

    def frame(self,q,driver,phase,weld_index):
        robots,guns={},{}
        for rid,model in self.world.models.items():
            current=q if rid==self.rid else self.park
            value=float(driver if rid==self.rid else 0.)
            links=model.links(current,world=True)
            gun_pose=links['245']@model.gun_home
            deltas=model.gun_model.transforms(value)
            robots[rid]=dict(q_rad=current.tolist(),links={k:m.reshape(-1).tolist() for k,m in links.items()},
                             tcp_matrix=model.tcp(current).reshape(-1).tolist())
            guns[model.gun_id]=dict(transform=gun_pose.reshape(-1).tolist(),driver_mm=value,
                                    joint_values=model.gun_model.joint_values(value),
                                    parts={pid:(gun_pose@deltas[p['link_key']]).reshape(-1).tolist() for pid,p in model.gun_parts.items()})
        return dict(time_s=self.at,phase=phase,weld_id=self.scene['welds'][weld_index]['id'],
                    task_index=weld_index+1,completed_welds=weld_index+(phase in ('open','retreat','transfer','loop_return')),
                    robots=robots,guns=guns,relaxed_interference_mode=True)

    def motion(self,qs,da,db,phase,index,dwell=None):
        qs=np.asarray(qs,dtype=float)
        spline=CubicSpline(np.linspace(0.,1.,len(qs)),qs) if len(qs)>2 else None
        def path(tau):
            u=quintic(tau)
            return spline(u) if spline is not None else qs[0]+u*(qs[-1]-qs[0])
        dense=np.linspace(0,1,501)
        qdense=np.array([path(t) for t in dense])
        if not all(self.model.limits_valid(q) for q in qdense):
            raise RuntimeError(f'joint limits violated in {phase} at task {index+1}')
        coupled=qdense@self.timing.coupling.T
        passive=np.array([passive_joint_values(self.scene,q) for q in qdense])
        driver=np.array([da+quintic(t)*(db-da) for t in dense])
        gun=np.array([self.model.gun_model.joint_values(x) for x in driver])
        arrays=[(qdense,self.timing.speeds,self.timing.accelerations),
                (coupled,self.timing.speeds,self.timing.accelerations),
                (passive,self.timing.passive_speeds,self.timing.passive_accelerations),
                (gun,self.timing.gun_speeds,self.timing.gun_accelerations)]
        duration=.12
        for positions,v,a in arrays:
            velocity=np.gradient(positions,dense,axis=0,edge_order=2)
            acceleration=np.gradient(velocity,dense,axis=0,edge_order=2)
            duration=max(duration,float(np.max(np.abs(velocity)/v)),
                         float(np.sqrt(np.max(np.abs(acceleration)/a))))
        duration=float(dwell if dwell is not None else duration*1.10)
        probes=[]
        for t in np.linspace(0.,1.,11 if phase in ('transfer','loop_return') else 7):
            q=path(t);d=float(da+quintic(t)*(db-da));check=self.cost.check(q,d)
            probes.append(dict(fraction=float(t),q_rad=q.tolist(),driver_mm=d,**check))
        segment=dict(phase=phase,weld_id=self.scene['welds'][index]['id'],task_index=index+1,
                     start_time_s=self.at,duration_s=duration,knots_q_rad=qs.tolist(),driver_start_mm=da,driver_end_mm=db,
                     joint_limits_valid=True,observed_max_depth_mm=max(x['observed_max_depth_mm'] for x in probes),
                     interference_probes=probes)
        for t in np.linspace(0.,1.,max(2,int(np.ceil(duration*self.fps))+1)):
            frame=self.frame(path(t),da+quintic(t)*(db-da),phase,index)
            frame['time_s']=self.at+float(t)*duration
            self.frames.append(frame)
        self.at+=duration;self.segments.append(segment)
        return segment

    def build(self,choices,routes):
        for index,(w,candidate) in enumerate(zip(self.scene['welds'],choices)):
            q=np.array(candidate['q_rad']);stroke=np.array(candidate['stroke']['qs']);entry=stroke[-1]
            if self.frames:
                previous=np.array(self.frames[-1]['robots'][self.rid]['q_rad'])
                if np.max(np.abs(previous-entry))>1e-8:raise RuntimeError('transfer/entry discontinuity')
            self.motion(stroke[::-1],self.open,self.open,'ingress',index)
            self.motion([q,q],self.open,0.,'close',index)
            self.motion([q,q],0.,0.,'weld',index,dwell=self.dwell)
            self.motion([q,q],0.,self.open,'open',index)
            self.motion(stroke,self.open,self.open,'retreat',index)
            transfer=[]
            if index<len(routes):
                route=np.array(routes[index])
                for qa,qb in zip(route,route[1:]):transfer.append(self.motion([qa,qb],self.open,self.open,'transfer',index))
            task=dict(id=w['id'],name=w['name'],group=w['group'],source_sequence=w['sequence'],
                      assigned_robot=self.rid,phases=workflow_phases(index<len(routes)),weld_q_rad=q.tolist(),
                      entry_q_rad=entry.tolist(),exit_q_rad=entry.tolist(),
                      entry_tcp_matrix=self.model.tcp(entry).reshape(-1).tolist(),
                      exit_tcp_matrix=self.model.tcp(entry).reshape(-1).tolist(),
                      weld_tcp_matrix=self.model.tcp(q).reshape(-1).tolist(),
                      source_world_xyz=w['world_xyz'],stroke_distance_mm=candidate['stroke']['distance_mm'],
                      stroke_direction_world=candidate['stroke']['normal_world'],
                      entry_exit_same_clearance_pose=True,entry_line_error_mm=candidate['stroke']['line_error_mm'],
                      endpoint_position_error_mm=float(np.linalg.norm(self.model.tcp(q)[:3,3]-np.array(w['world_xyz']))),
                      endpoint_closed_interference=candidate['endpoint_closed'],
                      maximum_observed_depth_mm=max(s['observed_max_depth_mm'] for s in self.segments if s['task_index']==index+1))
            self.tasks.append(task)
            print(f'BUILT {index+1}/29 {w["id"]} at {self.at:.2f}s depth {task["maximum_observed_depth_mm"]:.3f} mm',flush=True)
        validate_workflow([w['id'] for w in self.scene['welds']],self.tasks)
        route=continuous_joint_path(self.model,np.array(choices[-1]['stroke']['qs'][-1]),np.array(choices[0]['stroke']['qs'][-1]))
        if route is None:raise RuntimeError('could not legally close loop')
        for qa,qb in zip(route,route[1:]):self.motion([qa,qb],self.open,self.open,'loop_return',len(choices)-1)
        self.export()

    def export(self):
        for a,b in zip(self.frames,self.frames[1:]):
            if b['time_s']<a['time_s']:raise RuntimeError('negative clock seam')
            if b['time_s']==a['time_s']:
                if (np.max(np.abs(np.array(a['robots'][self.rid]['q_rad'])-b['robots'][self.rid]['q_rad']))>1e-8 or
                        abs(a['guns'][self.model.gun_id]['driver_mm']-b['guns'][self.model.gun_id]['driver_mm'])>1e-8):
                    raise RuntimeError('pose jump at phase seam')
        hashes_after={str(p.relative_to(ROOT)):digest(p) for p in self.source_files}
        if hashes_after!=self.hashes:raise RuntimeError('source or archived strict artifacts changed')
        summary=dict(schema='user_authorized_relaxed29_offline_v1',relaxed_interference_mode=True,
                      relaxed_workflow_count=len(self.tasks),source_weld_count=29,strict_planned_task_count=0,
                      planned_welds_relaxed=len(self.tasks),weld_dwell_s=self.dwell,
                      reference_interference_mm=self.cost.reference,
                      tasks_within_observed_depth_reference=sum(t['maximum_observed_depth_mm']<=self.cost.reference for t in self.tasks),
                      maximum_observed_depth_mm=max(s['observed_max_depth_mm'] for s in self.segments),
                      collision_verified=False,system_safe=False,diagnostic_only=True,
                      position_verified=True,joint_limits_verified=True,frame_count=len(self.frames),duration_s=self.at,
                      selected_weld_ids=[t['id'] for t in self.tasks],
                      source_initial_to_start_verified=False,alternative_simulation_start=True,
                      interference_depth_scope='maximum of finite sampled FCL triangle contacts, capped at 8 contacts per pair; not global volumetric penetration',
                      orientation_source='existing alternate projected surface normal and optimized roll; original fixed Euler orientation is not claimed',
                      weld_scope='offline dwell with source gun closure, not weld current/thermal simulation')
        plan=dict(schema=summary['schema'],summary=summary,status=summary,units='mm/rad',
                  selected_weld_ids=summary['selected_weld_ids'],welds=self.scene['welds'],
                  tasks=self.tasks,frames=self.frames,loop_seam_verified=True)
        decoded=json.dumps(plan,ensure_ascii=False,separators=(',',':')).encode()
        name='welding_plan_29_relaxed_v1.json.gz';path=self.out/name
        path.write_bytes(gzip.compress(decoded,compresslevel=6,mtime=0))
        audit=dict(schema='separate_relaxed29_visual_audit_v1',plans={name:dict(sha256=hashlib.sha256(decoded).hexdigest(),
                   relative_path=str(path.relative_to(ROOT)),status=summary,note='User-authorized interference-tolerant offline demonstration; strict original29 is unchanged')})
        (self.out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        report=dict(summary=summary,tasks=self.tasks,segments=self.segments,source_hashes_before=self.hashes,
                    source_hashes_after=hashes_after,source_unchanged=True,FCL_unique_samples=self.cost.count,
                    source_timing=self.timing.metadata(),plan_path=str(path),
                    limitations=['Finite collision sampling and capped contact witnesses do not certify continuous safety.',
                                 'Interference above the reporting reference is explicitly retained, not silently accepted.',
                                 'R2 uses an archived alternate parked state; original start transition not verified.',
                                 'Internal gun assembly contacts and intended mounts are excluded; not certified as independent free pairs.'])
        (self.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        lines=['# 上航发原29焊点：允许干涉的完整工艺演示','',
               f'完成 {len(self.tasks)}/29 个工艺周期，{len(self.frames)} 帧，{self.at:.2f} s；每点进枪、合枪、焊接停留、打开、退枪、转移。',
               '',f'默认进/退枪偏移 {self.distance:g} mm，逐点实际偏移见下表；焊接停留 {self.dwell:g} s。',
               f'干涉报告参照值 {self.cost.reference:g} mm；{summary["tasks_within_observed_depth_reference"]}/29 个周期的有限FCL深度样本未超过参照值。',
               f'最大观察到的FCL三角接触深度 {summary["maximum_observed_depth_mm"]:.3f} mm。此值是有限接触见证，不是完整体积穿透量。',
               '', '严格验证仍0/29，system_safe=false；这是用户允许干涉的仿真工艺路线。',
               'R1执行29点，R2替代停放。原文件不变，姿态使用已有可达表面法向/滚转候选。',
               '', '|序号|焊点|进退枪距离 mm|TCP误差 mm|最大观察干涉 mm|', '|---|---|---:|---:|---:|']
        for i,t in enumerate(self.tasks):lines.append(f'|{i+1}|{t["id"]}|{t["stroke_distance_mm"]:.1f}|{t["endpoint_position_error_mm"]:.6f}|{t["maximum_observed_depth_mm"]:.3f}|')
        (self.out/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--reference-interference-mm',type=float,default=2.)
    parser.add_argument('--stroke-mm',type=float,default=100.)
    parser.add_argument('--weld-dwell-s',type=float,default=.8)
    parser.add_argument('--fps',type=float,default=25.)
    parser.add_argument('--reuse-pool',action='store_true')
    args=parser.parse_args()
    planner=Planner(args.output,args.reference_interference_mm,args.stroke_mm,args.weld_dwell_s,args.fps)
    if args.reuse_pool:
        progress=json.loads((args.output/'candidate_progress.json').read_text())
        if progress['source_hashes']!=planner.hashes or progress['completed']!=29:raise ValueError('candidate pool stale or incomplete')
        pool=progress['pool']
    else:pool=planner.candidate_pool()
    choices,routes=planner.choose_sequence(pool)
    (args.output/'selection.json').write_text(json.dumps(dict(choices=choices,routes=routes),ensure_ascii=False))
    planner.build(choices,routes)


if __name__=='__main__':main()
