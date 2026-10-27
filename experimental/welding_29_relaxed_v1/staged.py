"""Three-stage local welding blocks followed by fixed-endpoint transit planning."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from planner import Planner, ROOT, continuous_joint_path, digest, local_ik, pose_error
from contracts import legal_charts

OUTPUT = ROOT / 'build/welding_29_staged_v2'


def select_local_block(blocks, reference_mm):
    """Local correctness takes priority over any point-to-point motion cost."""
    if not blocks:
        raise ValueError('no reachable straight local block')
    return min(blocks, key=lambda b: (
        b['local_max_depth_mm'] > reference_mm,
        max(0., b['local_max_depth_mm'] - reference_mm),
        b['local_mean_soft_cost'],
        abs(b['stroke']['distance_mm'] - 100.)))


def freeze_local_blocks(blocks):
    return copy.deepcopy(blocks)


def straight_knots(model, q, direction, distance):
    pose=model.tcp(q)
    direction=np.asarray(direction,dtype=float)
    direction/=np.linalg.norm(direction)
    knots=[np.asarray(q).copy()]
    for travel in np.linspace(0,distance,max(3,int(np.ceil(distance/2.))+1))[1:]:
        target=pose.copy();target[:3,3]+=direction*travel
        next_q=local_ik(model,target,knots[-1])
        if next_q is None:return None
        knots.append(next_q)
    knots=np.array(knots);spline=CubicSpline(np.linspace(0,1,len(knots)),knots)
    line_error=rotation_error=0.
    for t in np.linspace(0,1,max(101,len(knots)*3)):
        actual_q=spline(t)
        if not model.limits_valid(actual_q):return None
        actual=model.tcp(actual_q)
        line_error=max(line_error,float(np.linalg.norm(actual[:3,3]-pose[:3,3]-direction*distance*t)))
        rotation_error=max(rotation_error,float(np.linalg.norm(Rotation.from_matrix(pose[:3,:3].T@actual[:3,:3]).as_rotvec())))
    if line_error>.02 or rotation_error>2e-4:return None
    return dict(qs=knots.tolist(),distance_mm=float(distance),sign=0.,
                normal_world=direction.tolist(),line_error_mm=line_error,rotation_error_rad=rotation_error,
                endpoint_tcp=model.tcp(knots[-1]).reshape(-1).tolist())


class StagedPlanner(Planner):
    def __init__(self,out=OUTPUT,reference_mm=2.,distance=100.,dwell=.8,fps=30.):
        super().__init__(out,reference_mm,distance,dwell,fps)
        self.stage1=[];self.stage2=[];self.stage3=[]
        self.stage2_sha256=None
        self.rrt_iterations=25000
        self.edge_step=.02
        self.adaptive_rrt=True
        self.force_rrt=True
        self.rrt_seconds=35.

    def posture_candidates(self,row):
        candidates=[]
        for source in row['candidates']:
            if source['robot_id']!=self.rid or not source['fk_candidate_passed']:continue
            q=np.array(source['q_rad'])
            if any(np.linalg.norm(q-np.array(c['q_rad']))<1e-4 for c in candidates):continue
            if not self.model.limits_valid(q):continue
            # Evaluate closure/opening itself before trying any retreat direction.
            checks=[self.cost.check(q,d) for d in np.linspace(0,self.open,9)]
            candidates.append(dict(candidate_index=source['candidate_index'],q_rad=q.tolist(),
                                   endpoint_closed=checks[0],endpoint_open=checks[-1],opening_checks=checks,
                                   opening_max_depth_mm=max(c['observed_max_depth_mm'] for c in checks),
                                   opening_mean_soft_cost=float(np.mean([c['soft_cost'] for c in checks]))))
        return sorted(candidates,key=lambda c:(max(0,c['opening_max_depth_mm']-self.cost.reference),c['opening_mean_soft_cost']))

    def evaluate_line(self,candidate,direction,distance):
        q=np.array(candidate['q_rad'])
        path=straight_knots(self.model,q,direction,distance)
        if path is None:return None
        knots=np.array(path['qs']);spline=CubicSpline(np.linspace(0,1,len(knots)),knots)
        checks=[]
        # Spatial samples every <=2mm, including submillimetre contact boundaries.
        fractions=np.unique(np.r_[np.linspace(0,1,len(knots)),.1/distance,.5/distance,1./distance])
        for fraction in fractions:
            check=self.cost.check(spline(fraction),self.open)
            checks.append(dict(fraction=float(fraction),**check))
        maximum=max(candidate['opening_max_depth_mm'],max(c['observed_max_depth_mm'] for c in checks))
        average=float(np.mean([c['soft_cost'] for c in checks]+[candidate['opening_mean_soft_cost']]))
        result=copy.deepcopy(candidate)
        result.update(stroke=path,local_max_depth_mm=maximum,local_mean_soft_cost=average,
                      local_probes=checks,score=average,within_reference=maximum<=self.cost.reference,
                      endpoint_position_error_mm=float(np.linalg.norm(self.model.tcp(q)[:3,3]-self.current_weld['world_xyz'])))
        return result

    def stage_local_blocks(self):
        previous=self.model.q_initial.copy()
        for index,(weld,row) in enumerate(zip(self.scene['welds'],self.inventory['rows'])):
            self.current_weld=weld
            postures=self.posture_candidates(row)
            self.stage1.append(dict(weld_id=weld['id'],candidates=postures))
            (self.out/'stage1_welding_postures.json').write_text(json.dumps(dict(source_hashes=self.hashes,rows=self.stage1),ensure_ascii=False))
            blocks=[]
            # Source normals first. If they fail, keep welding orientation fixed
            # and test tool-frame lateral straight lines, which are valid TCP
            # line movements without imposing a new welding orientation.
            for candidate in postures:
                if blocks and candidate['opening_max_depth_mm']>min(b['local_max_depth_mm'] for b in blocks)+.05:continue
                pose=self.model.tcp(np.array(candidate['q_rad']))
                directions=[-pose[:3,2],pose[:3,2],pose[:3,0],-pose[:3,0],pose[:3,1],-pose[:3,1]]
                for direction_index,direction in enumerate(directions):
                    for distance in dict.fromkeys([self.distance,50.,30.]):
                        if distance>self.distance:continue
                        result=self.evaluate_line(candidate,direction,distance)
                        if result is not None:
                            result['direction_kind']='tool_z' if direction_index<2 else 'fixed_tool_lateral'
                            result['weld_id']=weld['id'];blocks.append(result)
                            if result['within_reference']:break
                    if blocks and min(b['local_max_depth_mm'] for b in blocks)<=self.cost.reference:break
                if blocks and min(b['local_max_depth_mm'] for b in blocks)<=self.cost.reference:break
            # Fine roll expansion when saved posture inventory cannot achieve a
            # low-interference local block. TCP and surface normal remain fixed.
            if not blocks or min(b['local_max_depth_mm'] for b in blocks)>self.cost.reference:
                base_candidates=postures[:3]
                for base in base_candidates:
                    original=self.model.tcp(np.array(base['q_rad']))
                    for angle in [-30.,-15.,15.,30.,-60.,60.,-90.,90.]:
                        target=original.copy();target[:3,:3]=original[:3,:3]@Rotation.from_euler('z',angle,degrees=True).as_matrix()
                        fit=least_squares(lambda q:pose_error(self.model,q,target),np.array(base['q_rad']),
                                          bounds=(self.model.bounds[0],self.model.bounds[1]),max_nfev=100,
                                          xtol=1e-11,ftol=1e-11,gtol=1e-11)
                        q=fit.x
                        if not self.model.limits_valid(q) or np.linalg.norm(pose_error(self.model,q,target))>1e-5:continue
                        checks=[self.cost.check(q,d) for d in np.linspace(0,self.open,9)]
                        candidate=dict(candidate_index='fine_roll',roll_delta_deg=angle,q_rad=q.tolist(),
                                       endpoint_closed=checks[0],endpoint_open=checks[-1],opening_checks=checks,
                                       opening_max_depth_mm=max(c['observed_max_depth_mm'] for c in checks),
                                       opening_mean_soft_cost=float(np.mean([c['soft_cost'] for c in checks])))
                        if blocks and candidate['opening_max_depth_mm']>min(b['local_max_depth_mm'] for b in blocks)+.05:continue
                        pose=self.model.tcp(q)
                        for direction_index,direction in enumerate([-pose[:3,2],pose[:3,2],pose[:3,0],-pose[:3,0],pose[:3,1],-pose[:3,1]]):
                            for distance in [self.distance,50.,30.]:
                                result=self.evaluate_line(candidate,direction,distance)
                                if result is not None:
                                    result.update(weld_id=weld['id'],direction_kind='fine_roll_fixed_orientation_line')
                                    blocks.append(result)
                                    if result['within_reference']:break
                            if blocks and min(b['local_max_depth_mm'] for b in blocks)<=self.cost.reference:break
                        if blocks and min(b['local_max_depth_mm'] for b in blocks)<=self.cost.reference:break
                    if blocks and min(b['local_max_depth_mm'] for b in blocks)<=self.cost.reference:break
            selected=select_local_block(blocks,self.cost.reference)
            # Pick a constant legal J6 branch now; phase3 never changes the
            # frozen welding posture or its approach/retract motion.
            charts=legal_charts(self.model,np.array(selected['q_rad']),previous)
            for q in charts:
                delta=q-np.array(selected['q_rad']);knots=np.array(selected['stroke']['qs'])+delta
                if all(self.model.limits_valid(k) for k in knots):
                    selected['q_rad']=q.tolist();selected['stroke']['qs']=knots.tolist();break
            selected['considered_local_blocks']=len(blocks)
            selected['entry_exit_same_clearance_pose']=True
            selected['local_frozen']=True
            self.stage2.append(selected);previous=np.array(selected['stroke']['qs'][-1])
            (self.out/'stage2_frozen_local_blocks.json').write_text(json.dumps(dict(source_hashes=self.hashes,reference_mm=self.cost.reference,rows=self.stage2),ensure_ascii=False))
            print(f'STAGE2 {index+1}/29 {weld["id"]}: line {selected["stroke"]["distance_mm"]:.0f}mm '
                  f'error {selected["stroke"]["line_error_mm"]:.6f}mm | '
                  f'max local depth {selected["local_max_depth_mm"]:.3f}mm | choices {len(blocks)}',flush=True)
        self.stage2=freeze_local_blocks(self.stage2)
        self.stage2_sha256=digest(self.out/'stage2_frozen_local_blocks.json')
        return self.stage2

    def edge_evidence(self,qa,qb,spatial_step=None):
        if spatial_step is None:spatial_step=self.edge_step
        qa,qb=np.asarray(qa),np.asarray(qb)
        n=max(4,int(np.ceil(np.max(np.abs(qb-qa))/spatial_step))+1)
        depths=[];costs=[]
        for t in np.linspace(0,1,n):
            q=qa+t*(qb-qa)
            if not self.model.limits_valid(q):return dict(legal=False,maximum=float('inf'),mean=float('inf'))
            c=self.cost.check(q,self.open);depths.append(c['observed_max_depth_mm']);costs.append(c['soft_cost'])
        return dict(legal=True,maximum=max(depths),mean=float(np.mean(costs)),samples=n)

    def route_evidence(self,route):
        rows=[self.edge_evidence(a,b) for a,b in zip(route,route[1:])]
        return dict(legal=all(r['legal'] for r in rows),maximum=max(r['maximum'] for r in rows),
                    mean=float(np.mean([r['mean'] for r in rows])),edges=rows,
                    travel=float(sum(np.linalg.norm(b-a) for a,b in zip(route,route[1:]))))

    def transfer_route(self,qa,qb,index,initial_route=None):
        started=time.monotonic()
        direct=continuous_joint_path(self.model,qa,qb)
        if direct is None:raise RuntimeError('no joint-limit-valid direct fallback')
        baseline=self.route_evidence(direct);best=direct;best_info=baseline
        start_depth=self.cost.check(qa,self.open)['observed_max_depth_mm']
        end_depth=self.cost.check(qb,self.open)['observed_max_depth_mm']
        threshold=max(self.cost.reference,start_depth+.01,end_depth+.01)
        def better(info,previous):
            return (info['maximum']>threshold,max(0,info['maximum']-threshold),info['mean'],info['travel']) < (
                previous['maximum']>threshold,max(0,previous['maximum']-threshold),previous['mean'],previous['travel'])
        if initial_route is not None:
            initial_route=[np.asarray(q) for q in initial_route]
            if not (np.allclose(initial_route[0],qa,atol=1e-9,rtol=0) and
                    np.allclose(initial_route[-1],qb,atol=1e-9,rtol=0)):
                raise ValueError('refinement must preserve frozen clearance endpoints')
            info=self.route_evidence(initial_route)
            if info['legal'] and better(info,best_info):best,best_info=initial_route,info
        if best_info['maximum']>threshold and not self.force_rrt:
            # Cartesian lifted waypoints only belong to the transfer. The local
            # ingress/egress remains frozen, exactly as the user requested.
            pa,pb=self.model.tcp(qa),self.model.tcp(qb)
            for offset in ([0,180,0],[0,350,0],[0,550,0],[0,150,200],[0,150,-200],[200,250,0],[-200,250,0]):
                if time.monotonic()-started>25:break
                lift=[]
                for target,seed in [(pa,qa),(pb,qb)]:
                    target=target.copy();target[:3,3]+=offset
                    fit=least_squares(lambda q:pose_error(self.model,q,target),seed,bounds=(self.model.bounds[0],self.model.bounds[1]),
                                      max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                    charts=legal_charts(self.model,fit.x,seed)
                    if not charts or np.linalg.norm(pose_error(self.model,charts[0],target))>.001:break
                    lift.append(charts[0])
                if len(lift)!=2:continue
                route=[qa,lift[0],lift[1],qb]
                info=self.route_evidence(route)
                if info['legal'] and better(info,best_info):best,best_info=route,info
                if best_info['maximum']<=threshold:break
        # Bounded bidirectional RRT-Connect in the original 6D legal chart.
        attempts=0
        rrt_found=False
        if best_info['maximum']>threshold or self.force_rrt:
            rng=np.random.default_rng(20261008+index)
            trees=[([qa.copy()],[-1]),([qb.copy()],[-1])]
            def feasible_edge(a,b):
                step=min(.07,self.edge_step) if self.adaptive_rrt else .07
                n=max(2,int(np.ceil(np.max(np.abs(b-a))/step))+1)
                for t in np.linspace(0,1,n):
                    q=a+t*(b-a)
                    if not self.model.limits_valid(q) or self.cost.check(q,self.open)['observed_max_depth_mm']>threshold:return False
                return True
            def extend(tree,target):
                nodes,parents=tree
                nearest=int(np.argmin([np.linalg.norm(x-target) for x in nodes]));a=nodes[nearest]
                length=np.linalg.norm(target-a)
                if length<1e-8:return nearest,True
                for step in ([.25,.125,.0625,.03125,.015625] if self.adaptive_rrt else [.25]):
                    b=a+(target-a)*min(1.,step/length)
                    if feasible_edge(a,b):
                        nodes.append(b);parents.append(nearest)
                        return len(nodes)-1,length<=step
                return None,False
            def chain(tree,k):
                path=[]
                while k>=0:path.append(tree[0][k]);k=tree[1][k]
                return path[::-1]
            for iteration in range(self.rrt_iterations):
                if time.monotonic()-started>self.rrt_seconds:break
                side=iteration%2
                if iteration%5==0:target=trees[1-side][0][-1]
                elif self.force_rrt and initial_route is not None and iteration%4==0:
                    target=initial_route[int(rng.integers(len(initial_route)))]+rng.normal(0,.15,6)
                    target=np.clip(target,self.model.bounds[0]+1e-6,self.model.bounds[1]-1e-6)
                elif iteration%3:
                    target=qa+rng.random()*(qb-qa)+rng.normal(0,.45,6)
                    target=np.clip(target,self.model.bounds[0]+1e-6,self.model.bounds[1]-1e-6)
                else:target=rng.uniform(*self.model.bounds)
                k,_=extend(trees[side],target);attempts+=1
                if k is None:continue
                goal=trees[side][0][k]
                for _ in range(45):
                    j,reached=extend(trees[1-side],goal)
                    if j is None:break
                    if reached:
                        first=chain(trees[side],k);second=chain(trees[1-side],j)
                        route=first+second[::-1][1:]
                        if side==1:route=route[::-1]
                        info=self.route_evidence(route)
                        if info['legal'] and info['maximum']<=threshold:
                            rrt_found=True
                            if self.force_rrt or better(info,best_info):best,best_info=route,info
                        break
                if rrt_found:break
        # Shortcut only when it preserves the best observed depth band.
        if len(best)>2:
            rng=np.random.default_rng(1000+index)
            for _ in range(30):
                a,b=sorted(rng.choice(len(best),2,replace=False))
                if b<=a+1:continue
                e=self.edge_evidence(best[a],best[b])
                if e['legal'] and e['maximum']<=best_info['maximum']+.001:
                    best=best[:a+1]+best[b:]
            best_info=self.route_evidence(best)
        row=dict(connection_index=index,from_q_rad=qa.tolist(),to_q_rad=qb.tolist(),
                 route=[q.tolist() for q in best],direct_observed_max_depth_mm=baseline['maximum'],
                 optimized_observed_max_depth_mm=best_info['maximum'],within_reference=best_info['maximum']<=self.cost.reference,
                 allowed_endpoint_band_mm=threshold,within_endpoint_band=best_info['maximum']<=threshold,
                 original_kinematic_limits_valid=best_info['legal'],nodes=len(best),rrt_attempts=attempts,
                 planner='RRT-Connect' if rrt_found else 'diagnostic_fallback',rrt_solved=rrt_found,
                 fallback_retained=not rrt_found and self.force_rrt,
                 seconds=time.monotonic()-started,evidence=best_info)
        self.stage3.append(row)
        (self.out/'stage3_transfer_routes.json').write_text(json.dumps(dict(stage2_sha256=self.stage2_sha256,rows=self.stage3),ensure_ascii=False))
        print(f'STAGE3 {index+1}/29: direct {baseline["maximum"]:.2f} -> {best_info["maximum"]:.2f} mm | '
              f'{len(best)} nodes {attempts} RRT probes {row["seconds"]:.1f}s',flush=True)
        return row['route']

    def plan_transfers(self,blocks):
        routes=[]
        for i,(a,b) in enumerate(zip(blocks,blocks[1:])):
            qa=np.array(a['stroke']['qs'][-1]);qb=np.array(b['stroke']['qs'][-1])
            routes.append(self.transfer_route(qa,qb,i))
        self.loop_route=self.transfer_route(np.array(blocks[-1]['stroke']['qs'][-1]),np.array(blocks[0]['stroke']['qs'][-1]),28)
        return routes

    def build(self,choices,routes):
        # Reuse the source-timed process exporter. Its build() will call this
        # export override; replace its final direct loop before that export.
        self.frozen_choices=choices
        super().build(choices,routes)

    def export(self):
        # Parent build appends a direct final loop; replace it with separately
        # optimized fixed-endpoint phase3 loop before writing final artifacts.
        old=[s for s in self.segments if s['phase']=='loop_return']
        if old:
            cutoff=old[0]['start_time_s']
            self.segments=[s for s in self.segments if s['phase']!='loop_return']
            self.frames=[f for f in self.frames if f['phase']!='loop_return']
            self.at=cutoff
            route=np.array(self.loop_route)
            for qa,qb in zip(route,route[1:]):self.motion([qa,qb],self.open,self.open,'loop_return',28)
        if digest(self.out/'stage2_frozen_local_blocks.json')!=self.stage2_sha256:
            raise RuntimeError('phase3 changed frozen local blocks')
        super().export()
        report=json.loads((self.out/'report.json').read_text())
        summary=report['summary']
        summary.update(schema='user_authorized_staged29_offline_v2',staged_local_blocks_frozen=True,
                       local_blocks_within_reference=sum(b['local_max_depth_mm']<=self.cost.reference for b in self.stage2),
                       local_max_observed_depth_mm=max(b['local_max_depth_mm'] for b in self.stage2),
                       transfer_connections_within_reference=sum(x['within_reference'] for x in self.stage3),
                       transfer_planner='RRT-Connect' if self.force_rrt else 'direct_lift_rrt',
                       transfer_rrt_solved_count=sum(x.get('rrt_solved',False) for x in self.stage3),
                       stage2_sha256=self.stage2_sha256)
        old_path=self.out/'welding_plan_29_relaxed_v1.json.gz'
        plan=json.loads(gzip.decompress(old_path.read_bytes()))
        plan['summary']=plan['status']=summary;plan['schema']=summary['schema']
        for task,block,connection in zip(plan['tasks'],self.stage2,self.stage3):
            task['local_max_observed_depth_mm']=block['local_max_depth_mm']
            task['local_within_reference']=block['local_max_depth_mm']<=self.cost.reference
            task['local_frozen']=True
            task['transfer_planner']=connection.get('planner','direct_lift_rrt')
            task['transfer_rrt_solved']=connection.get('rrt_solved',False)
            task['transfer_fallback_retained']=connection.get('fallback_retained',False)
            task['transfer_max_sampled_depth_mm']=connection['optimized_observed_max_depth_mm']
            task['transfer_is_loop_return']=connection['connection_index']==28
        raw=json.dumps(plan,ensure_ascii=False,separators=(',',':')).encode()
        final=self.out/'welding_plan_29_staged_v2.json.gz';final.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
        old_path.unlink()
        report.update(summary=summary,tasks=plan['tasks'],plan_path=str(final),stage2_sha256=self.stage2_sha256,
                      stage1_artifact='stage1_welding_postures.json',stage2_artifact='stage2_frozen_local_blocks.json',
                      stage3_artifact='stage3_transfer_routes.json')
        (self.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        (self.out/'audit.json').write_text(json.dumps(dict(schema='separate_staged29_visual_audit_v2',plans={final.name:dict(
            sha256=hashlib.sha256(raw).hexdigest(),relative_path=str(final.relative_to(ROOT)),status=summary)}),ensure_ascii=False,indent=2))
        text=(self.out/'结果报告.md').read_text().replace('允许干涉的完整工艺演示','三阶段直线进退枪完整工艺演示')
        text+='\n## 三阶段结果\n\n'
        text+=f'局部段冻结后再规划退枪点连接；局部段 {summary["local_blocks_within_reference"]}/29 在 {self.cost.reference:g} mm 观察干涉参照内，最大 {summary["local_max_observed_depth_mm"]:.3f} mm。\n\n'
        text+=f'点间及闭环连接 {summary["transfer_connections_within_reference"]}/29 在参照内；完整进退枪直线误差见 stage2。所有超过参照的条目仍明确保留为失败/宽松演示。\n'
        text+=f'转移算法 {summary["transfer_planner"]}，RRT-Connect 实际找到路径 {summary["transfer_rrt_solved_count"]}/29；未找到的路线只保留为诊断回放，不算RRT成功。\n'
        (self.out/'结果报告.md').write_text(text,encoding='utf-8')
        print('STAGED SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--reuse-local',action='store_true')
    parser.add_argument('--reuse-transfers',action='store_true')
    parser.add_argument('--refine-transfers',action='store_true',help='refine only over-reference connections, retaining their frozen endpoints and previous best route')
    parser.add_argument('--rrt-iterations',type=int,default=25000)
    parser.add_argument('--edge-step',type=float,default=.02)
    parser.add_argument('--adaptive-rrt',action='store_true',default=True)
    parser.add_argument('--force-rrt',action='store_true',default=True,help='run RRT-Connect for every transfer (default); no Cartesian lift planner')
    parser.add_argument('--rrt-seconds',type=float,default=35.)
    parser.add_argument('--seed-transfers',type=Path,help='previous hash-bound routes used only as sampling guidance and explicit diagnostic fallback')
    args=parser.parse_args()
    planner=StagedPlanner(args.output)
    planner.rrt_iterations=args.rrt_iterations
    planner.edge_step=args.edge_step
    planner.adaptive_rrt=args.adaptive_rrt
    planner.force_rrt=args.force_rrt
    planner.rrt_seconds=args.rrt_seconds
    if args.reuse_local:
        path=args.output/'stage2_frozen_local_blocks.json'
        data=json.loads(path.read_text())
        if data['source_hashes']!=planner.hashes or len(data['rows'])!=29:raise ValueError('local blocks stale/incomplete')
        planner.stage2=freeze_local_blocks(data['rows']);planner.stage2_sha256=digest(path)
    else:planner.stage_local_blocks()
    if args.reuse_transfers or args.refine_transfers or args.seed_transfers:
        data=json.loads((args.seed_transfers or args.output/'stage3_transfer_routes.json').read_text())
        if data['stage2_sha256']!=planner.stage2_sha256 or len(data['rows'])!=29:raise ValueError('transfers stale/incomplete')
        if args.refine_transfers or args.seed_transfers:
            for i,row in enumerate(data['rows']):
                if planner.force_rrt or row['optimized_observed_max_depth_mm']>planner.cost.reference:
                    planner.transfer_route(np.array(row['from_q_rad']),np.array(row['to_q_rad']),i,row['route'])
                else:
                    planner.stage3.append(row)
            routes=[r['route'] for r in planner.stage3[:28]];planner.loop_route=planner.stage3[28]['route']
        else:
            planner.stage3=data['rows'];routes=[r['route'] for r in data['rows'][:28]];planner.loop_route=data['rows'][28]['route']
    else:routes=planner.plan_transfers(planner.stage2)
    planner.build(planner.stage2,routes)


if __name__=='__main__':main()
