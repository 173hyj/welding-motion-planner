"""No-QP ablation: original rays, straight pre-collision release, RRT transfer."""
import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'welding_variable_open_v1'))
sys.path.insert(0,str(HERE.parent/'welding_waypoint_qp_v1'))
sys.path.insert(0,str(HERE.parent/'welding_waypoint_qp_v2'))
from run_qp import opening_frames,longest_blocked_rays
from variable_core import prefix_available
from trace_motion import lift_path as trace_full
from variable_world import VariableToolWorld
from run_geometry import atomic,digest,initialize_worker
from run_initial import assert_frozen_geometry_cache
from tool_world import ROOT
from contracts import ContactTracker,certify_translation
from run_arm import full_probe
from prefix_motion import lift_prefix
from rrt import bidirectional_rrt,edge_samples,path_length,choose_pre_collision
OUT=ROOT/'build/welding_direct_rrt_v1';_ENGINE=None


def geometry_prefix(engine,rid,start,moved,driver,direction,length):
    state=engine.state(rid,start,driver,moved,engine.world.static,'opening')
    context=engine.release_context(rid,start,moved,dict(open_state=state),driver_mm=driver)
    tracker=ContactTracker(context['initial_touching']);direction=np.asarray(direction)
    if any(np.asarray(p['outward_normal'])@direction<=1e-6 for p in context['supports']):return dict(passed=False,reason='not_outward')
    def pose(s):
        t=start.copy();t[:3,3]+=s*direction;return t
    separated=0. if not context['initial_touching'] else None
    if separated is None:
        for s in np.linspace(0.,min(5.,length),int(np.ceil(min(5.,length)/.1))+1):
            state=engine.state(rid,pose(s),driver,moved,engine.world.static,'release',context)
            if state['blocking_pairs'] or not tracker.advance(state['touching'],s):return dict(passed=False,reason='release_blocked',failure_mm=float(s))
            if not state['touching'] and engine.minimum_distance(rid,pose(s),driver,engine.world.static)>.05:
                separated=float(s);break
    if separated is None:return dict(passed=False,reason='not_separated')
    certificate=certify_translation(lambda s:engine.minimum_distance(rid,pose(s),driver,engine.world.static),separated,length)
    return dict(passed=certificate['passed'],separation_mm=separated,certificate=certificate)


def point_job(index,parent_row,qp_row,weld,out,rrt_iterations):
    engine=_ENGINE;started=time.perf_counter();selected=parent_row['initial'];rid=selected['robot_id'];other=selected['other_robot_id'];park=np.asarray(selected['other_q_rad']);model=engine.world.models[rid]
    engine.set_angle(selected['positioner_deg']);moved,_=engine.target(weld,selected);start=model.tcp(np.asarray(selected['q_rad']));candidate={k:selected[k] for k in ('robot_id','roll_deg','axis_sign')}
    seed_start=time.perf_counter();seeds=[]
    for amplitude in (30.,60.,80.,185.):
        if not prefix_available(parent_row['opening_profile'],amplitude):continue
        driver=-amplitude
        opened=engine.state(rid,start,driver,moved,engine.environment.relative_env,'opening')
        exits=engine.search_exits(rid,start,moved,dict(open_state=opened),driver_mm=driver,count=2)
        rays=[dict(r,direction_frame='current_world') for r in exits['exits']]
        if not rays:rays=longest_blocked_rays(engine,rid,start,moved,exits['release_context'],driver)
        for ray in rays:
            trace=trace_full(engine,weld,candidate,selected,ray,driver_mm=driver,opening_frames=opening_frames(parent_row,start,driver))
            last=trace.get('length_mm',0.) if trace['passed'] else trace.get('last_accepted_travel_mm',0.)
            failure=trace.get('travel_mm',last)
            seeds.append(dict(driver_mm=driver,amplitude_mm=amplitude,direction_world=ray['direction_world'],
                final=dict(score=last,last_accepted_travel_mm=last,failure_travel_mm=failure,geometry_failure_mm=ray.get('geometric_failure_mm')),
                original_probe_reason=trace.get('reason','passed'),qp_used=False))
    seed_generation_s=time.perf_counter()-seed_start
    ranked=sorted(seeds,key=lambda t:-t['final']['score']);trials=[];best=None;all_selected=json.loads((ROOT/'build/welding_positioner_static_v3/report.json').read_text())['rows']
    for seed_index,seed in enumerate(ranked):
        driver=seed['driver_mm'];direction=np.asarray(seed['direction_world']);last=seed['final']['last_accepted_travel_mm']
        failure=seed['final'].get('failure_travel_mm') or last;length=min(last,choose_pre_collision(failure,3.))
        if seed['final'].get('geometry_failure_mm') is not None:length=min(length,choose_pre_collision(seed['final']['geometry_failure_mm'],5.))
        if length<5.:continue
        geom=geometry_prefix(engine,rid,start,moved,driver,direction,length)
        prefix=lift_prefix(engine,weld,candidate,selected,dict(direction_world=direction.tolist(),direction_frame='current_world',length_mm=length),
                           driver_mm=driver,opening_frames=opening_frames(parent_row,start,driver))
        trial=dict(seed_index=seed_index,driver_mm=driver,straight_length_mm=length,original_failure_mm=failure,
                   margin_before_failure_mm=failure-length,geometry=geom,prefix_passed=prefix['passed'],rrt_trials=[],passed=False)
        if not geom['passed'] or not prefix['passed']:
            trial['prefix_failure']=prefix if not prefix['passed'] else None;trials.append(trial);continue
        q_start=np.asarray(prefix['q_end_rad']);cache={};checks=0
        def state_valid(q):
            nonlocal checks
            key=tuple(np.round(q,9))
            if key not in cache:
                checks+=1;valid=full_probe(engine,rid,q,other,park,driver,moved,'separated')['passed']
                cache[key]=valid
            return cache[key]
        def edge_valid(a,b):return all(state_valid(q) for q in edge_samples(a,b,np.deg2rad(.05)))
        bank=[np.asarray(r['selected']['q_rad']) for r in all_selected if r.get('selected') and r['selected']['robot_id']==rid]
        bank += [model.q_initial,np.zeros(6),park]
        rng=np.random.default_rng(20261008+index)
        for scale in [3.,5.,10.,20.,35.]:
            bank += [q_start+rng.normal(size=6)*np.deg2rad(scale) for _ in range(24)]
        bank=sorted(bank,key=lambda q:np.linalg.norm(q-q_start));goals=[]
        for q in bank:
            if not model.limits_valid(q):continue
            tcp=model.tcp(q)
            if np.linalg.norm(tcp[:3,3]-start[:3,3])<200.:continue
            if not state_valid(q):continue
            clearance=engine.minimum_distance(rid,tcp,driver,engine.world.static)
            if clearance<5.:continue
            goals.append(q)
            if len(goals)>=3:break
        for gi,goal in enumerate(goals):
            begin=time.perf_counter()
            # Try the edge first; otherwise grow a joint-space RRT with all source checks.
            if edge_valid(q_start,goal):rrt_path=[q_start,goal];stats=dict(reason='direct_edge_free',iterations=0,nodes=2)
            else:rrt_path,stats=bidirectional_rrt(q_start,goal,model,edge_valid,seed=20261008+100*index+gi,max_iterations=rrt_iterations)
            rr=dict(goal_index=gi,goal_q_rad=goal.tolist(),found=rrt_path is not None,stats=stats,wall_s=time.perf_counter()-begin)
            trial['rrt_trials'].append(rr)
            print('STAGED_RRT',parent_row['weld_id'],'SEED',seed_index,'GOAL',gi,'FOUND',rrt_path is not None,'CHECKS',checks,'s',round(rr['wall_s'],2),flush=True)
            if rrt_path is not None:
                # Save actual q/TCP samples along the executed joint path, not straight TCP links.
                frames=list(prefix['frames'])
                dense_valid=True; dense_failure=None
                for a,b in zip(rrt_path,rrt_path[1:]):
                    for q in edge_samples(a,b,np.deg2rad(.05))[1:]:
                        if not state_valid(q):
                            dense_valid=False; dense_failure=dict(q_rad=q.tolist(),edge_start=a.tolist(),edge_end=b.tolist());break
                        tcp=model.tcp(q);frames.append(dict(q_rad=q.tolist(),driver_mm=driver,phase='rrt_transfer',tcp_xyz_mm=tcp[:3,3].tolist(),travel_mm=None))
                    if not dense_valid:break
                rr['dense_final_check_passed']=dense_valid
                if not dense_valid:
                    rr['dense_failure']=dense_failure
                    trial['rrt_trials'][-1]=rr
                    continue
                end=model.tcp(rrt_path[-1]);clearance=engine.minimum_distance(rid,end,driver,engine.world.static)
                name=f'paths/point_{index:02d}_staged.json'
                path=dict(passed=True,robot_id=rid,other_robot_id=other,other_q_rad=park.tolist(),positioner_deg=selected['positioner_deg'],
                      q_start_rad=selected['q_rad'],q_end_rad=rrt_path[-1].tolist(),driver_mm=driver,straight_prefix_mm=length,
                      direction_world=direction.tolist(),prefix_geometry=geom,rrt_nodes=[q.tolist() for q in rrt_path],
                      rrt_joint_path_length_rad=path_length(rrt_path),endpoint_displacement_from_weld_mm=float(np.linalg.norm(end[:3,3]-start[:3,3])),
                      endpoint_tool_clearance_mm=clearance,frames=frames,source_pair_checks=checks,
                      scope='Straight opening/release prefix then joint-space RRT; sampled complete source arms and original limits; no continuous full-system certificate.')
                atomic(out/name,path);best=dict(driver_mm=driver,amplitude_mm=-driver,path_file=name,path_sha256=digest(out/name),straight_prefix_mm=length)
                trial.update(passed=True,path_file=name,path_sha256=digest(out/name));break
        trial['valid_goal_count']=len(goals);trial['source_state_checks']=checks;trials.append(trial)
        atomic(out/f'point_{index:02d}_progress.json',dict(weld_id=parent_row['weld_id'],seeds=len(trials),passed=best is not None,elapsed_s=time.perf_counter()-started))
        if best:break
    result=dict(qp_used=False,original_ray_seed_generation_s=seed_generation_s,original_seeds=seeds,weld_id=parent_row['weld_id'],name=parent_row['name'],initial=selected,complete=True,staged_passed=best is not None,best=best,trials=trials,elapsed_s=time.perf_counter()-started)
    atomic(out/f'point_{index:02d}.json',result);return index,result


def main():
    global _ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--rrt-iterations',type=int,default=700);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('One to four workers')
    assert_frozen_geometry_cache();parent=ROOT/'build/welding_variable_open_v1/report.json';previous=json.loads(parent.read_text())
    baseline=dict(previous['manifest']['source_sha256']);oldcodes=previous['manifest']['code_sha256']
    baseline[str(parent.relative_to(ROOT))]=digest(parent)
    codes={str(p.relative_to(ROOT)):digest(p) for p in [HERE/n for n in ('rrt.py','prefix_motion.py','run_direct.py')]+[HERE.parent/'welding_waypoint_qp_v1/run_qp.py',HERE.parent/'welding_waypoint_qp_v2/trace_motion.py']}
    for name,value in {**baseline,**oldcodes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Frozen source/code changed')
    manifest=dict(source_sha256=baseline,previous_code_sha256=oldcodes,code_sha256=codes,parent_report_sha256=digest(parent),qp_used=False,
          settings=dict(staged_exit=True,straight_start_q_orientation_positioner_park_locked=True,pre_failure_margin_mm=3.,pre_geometry_coarse_margin_mm=5.,
               prefix_min_clearance_mm=.1,transfer_endpoint_min_displacement_mm=200.,transfer_endpoint_min_tool_clearance_mm=5.,rrt_edge_step_deg=.05,final_step_deg=.05,rrt_iterations=args.rrt_iterations))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
    if mp.exists():
        if not args.resume:raise FileExistsError('Use --resume or new output')
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('Settings/source/code changed')
        if (out/'report.json').exists():
            done=json.loads((out/'report.json').read_text())
            if done['summary']['complete']:
                for n,v in done['output_sha256'].items():
                    if digest(out/n)!=v:raise RuntimeError('Saved evidence changed')
                print('STAGED_READ_ONLY',json.dumps(done['summary']));return
    else:atomic(mp,manifest)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());welds={w['id']:w for w in scene['welds']};rows={};jobs=[]
    for i,row in enumerate(previous['rows']):
        if row['best']:continue
        saved=out/f'point_{i:02d}.json'
        if args.resume and saved.exists():rows[i]=json.loads(saved.read_text())
        else:jobs.append((i,row))
    started=time.perf_counter();load_s=0.
    if jobs:
        t=time.perf_counter();_ENGINE=VariableToolWorld(scene);load_s=time.perf_counter()-t;print('STAGED_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            fs=[pool.submit(point_job,i,row,None,welds[row['weld_id']],out,args.rrt_iterations) for i,row in jobs]
            for f in concurrent.futures.as_completed(fs):
                i,rows[i]=f.result();atomic(out/'progress.json',dict(completed=len(rows),total=6,passed=sum(r['staged_passed'] for r in rows.values()),wall_s=time.perf_counter()-started))
    for n,v in {**baseline,**oldcodes,**codes}.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Source/code changed during staged planning')
    outputs={f'point_{i:02d}.json':digest(out/f'point_{i:02d}.json') for i in rows}
    for r in rows.values():
        if r['best']:outputs[r['best']['path_file']]=digest(out/r['best']['path_file'])
    report=dict(manifest=manifest,summary=dict(complete=len(rows)==6,points=len(rows),additional_staged_passed=sum(r['staged_passed'] for r in rows.values()),
         model_load_s=load_s,wall_s=time.perf_counter()-started),rows=[rows[i] for i in sorted(rows)],output_sha256=outputs,
         scope='Original archived welding q, original finite directions and no QP updates. Straight pre-collision prefix followed by RRT. '
               'No continuous full-system safety certificate; original source limits and static assembly exemptions retained.')
    atomic(out/'report.json',report);print('STAGED_SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
