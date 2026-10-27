"""One-QP line-direction repair at failed source-geometry waypoints."""
import argparse
import concurrent.futures
import json
import multiprocessing
import time
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
from variable_world import VariableToolWorld
from variable_motion import lift_path
from variable_core import prefix_available
from run_initial import assert_frozen_geometry_cache
from run_geometry import atomic,digest,initialize_worker
from tool_world import ROOT
from refinement import PoseProblem
from qp_step import solve_step,tangent_basis,updated_direction
from proximity import collect_pairs,proximity_linearization,polygon_linearization

HERE=Path(__file__).parent;OUT=ROOT/'build/welding_waypoint_qp_v1'
AMPLITUDES=[30.,60.,80.,185.]
_ENGINE=None


def opening_frames(row,actual,driver):
    if not prefix_available(row['opening_profile'],-driver):raise ValueError('Opening prefix not certified')
    return [dict(q_rad=row['initial']['q_rad'],driver_mm=float(d),phase='closed' if d==0 else 'opening',
                 tcp_xyz_mm=actual[:3,3].tolist(),travel_mm=0.) for d in row['opening_profile']['passed_drivers_mm'] if d>=driver]


def longest_blocked_rays(engine,rid,start,moved,context,driver,count=2):
    # The previous no-exit point must also contribute a collision waypoint.
    ranked=[]
    for ray_local in engine.rays:
        ray=start[:3,:3]@ray_local
        if any(np.asarray(s['outward_normal'])@ray<=1e-6 for s in context['supports']):continue
        last=0.;failure=None
        for s in [0.,.1,.5,1.,2.,5.,10.,20.,40.,60.,80.,100.]:
            target=start.copy();target[:3,3]+=s*ray
            phase='release' if s<=5. else 'separated'
            state=engine.state(rid,target,driver,moved,engine.environment.relative_env,phase,context if phase=='release' else None)
            if state['blocking_pairs']:
                failure=float(s);break
            last=float(s)
        ranked.append(dict(direction_world=ray.tolist(),direction_frame='current_world',length_mm=100.,
                           geometric_prefix_mm=last,geometric_failure_mm=failure))
    ranked.sort(key=lambda r:-r['geometric_prefix_mm']);chosen=[]
    for r in ranked:
        if any(np.dot(r['direction_world'],x['direction_world'])>np.cos(np.deg2rad(20.)) for x in chosen):continue
        chosen.append(r)
        if len(chosen)==count:break
    return chosen


def build_qp(engine,weld,selected,driver,ray,failure):
    started=time.perf_counter();rid=selected['robot_id'];model=engine.world.models[rid]
    other=selected['other_robot_id'];park=np.asarray(selected['other_q_rad']);q0=np.asarray(selected['q_rad'])
    moved,_=engine.target(weld,selected);start=model.tcp(q0);direction=np.asarray(ray['direction_world']);basis=tangent_basis(direction)
    fail_s=float(failure.get('travel_mm') or ray.get('geometric_failure_mm') or 20.)
    travels=sorted(set([2.,10.,25.,50.,75.,100.]+[max(.1,min(100.,fail_s+x)) for x in (-2.,0.,2.,10.)]))
    n=6*len(travels)+2;pose_rows=[];pose_rhs=[];collision_rows=[];collision_rhs=[]
    lower=[];upper=[];nominals=[];diagnostics=[];q=q0.copy();gradient_s=0.
    open_state=engine.state(rid,start,driver,moved,engine.world.static,'opening')
    context=engine.release_context(rid,start,moved,dict(open_state=open_state),driver_mm=driver)
    for i,s in enumerate(travels):
        target=start.copy();target[:3,3]+=s*direction
        problem=PoseProblem(model,target)
        fit=least_squares(lambda x:problem.evaluate(x)[0],np.clip(q,model.bounds[0]+1e-9,model.bounds[1]-1e-9),
                          jac=lambda x:problem.evaluate(x)[1],bounds=(model.bounds[0],model.bounds[1]),
                          max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        q=fit.x;error,jac=problem.evaluate(q);weights=np.array([1000.,1000.,1000.,500.,500.,500.])
        A=np.zeros((6,n));A[:,6*i:6*i+6]=weights[:,None]*jac;A[:3,-2:]=-s*basis
        pose_rows.append(A);pose_rhs.extend((-weights*error).tolist())
        trust=np.deg2rad(10.)
        lower.extend(np.maximum(model.bounds[0]-q,-trust).tolist());upper.extend(np.minimum(model.bounds[1]-q,trust).tolist())
        phase='release' if s<=5. else 'separated'
        pairs=collect_pairs(engine,rid,q,other,park,driver,moved,phase,context if phase=='release' else None)
        g_start=time.perf_counter();values,gradient=proximity_linearization(engine,rid,q,other,park,driver,pairs);gradient_s+=time.perf_counter()-g_start
        for p,value,g in zip(pairs,values,gradient):
            row=np.zeros(n);row[6*i:6*i+6]=g;collision_rows.append(row);collision_rhs.append(.5-value)
        polygon_rows=polygon_linearization(model,q)
        for poly in polygon_rows:
            row=np.zeros(n);row[6*i:6*i+6]=1000.*poly['gradient'];collision_rows.append(row)
            collision_rhs.append(1000.*(1e-5-poly['signed_distance_rad']))
        nominals.append(dict(travel_mm=s,q_rad=q.tolist()))
        diagnostics.append(dict(travel_mm=s,nominal_limits_valid=model.limits_valid(q),position_error_mm=float(np.linalg.norm(error[:3])*1000.),
             rotation_error_deg=float(np.degrees(np.linalg.norm(error[3:]))),
             mesh_pairs=[dict(pair=p,proxy_mm=float(v),gradient_mm_per_rad=g.tolist()) for p,v,g in zip(pairs,values,gradient)],
             polygon_constraints=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()} for p in polygon_rows]))
    lower.extend([-np.tan(np.deg2rad(20.))]*2);upper.extend([np.tan(np.deg2rad(20.))]*2)
    G=np.asarray(collision_rows) if collision_rows else np.empty((0,n))
    result=solve_step(np.vstack(pose_rows),np.asarray(pose_rhs),G,np.asarray(collision_rhs),lower,upper,n_joint_variables=n-2)
    if result['solved']:
        x=np.asarray(result['step']);local_after=[]
        for i,record in enumerate(nominals):
            qn=np.asarray(record['q_rad'])+x[6*i:6*i+6];diag=diagnostics[i]
            pairs=[r['pair'] for r in diag['mesh_pairs']]
            # Only reevaluate frozen critical pairs, distinct from final full validation.
            values,_=proximity_linearization(engine,rid,qn,other,park,driver,pairs)
            local_after.append(dict(travel_mm=record['travel_mm'],q_rad=qn.tolist(),limits_valid=model.limits_valid(qn),
                                    frozen_pair_proxies_mm=values.tolist()))
        result.update(direction_step_tangent=x[-2:].tolist(),local_after=local_after)
    result.update(nominal_waypoints=nominals,diagnostics=diagnostics,waypoint_count=len(travels),
                  proximity_gradient_wall_s=gradient_s,total_wall_s=time.perf_counter()-started,
                  penetration_proxy='negative maximum among up to100 FCL triangle contacts; local proxy, not globally smooth signed distance')
    return result,basis


def point_job(index,row,weld,out):
    engine=_ENGINE;started=time.perf_counter();selected=row['initial'];rid=selected['robot_id']
    engine.set_angle(selected['positioner_deg']);moved,_=engine.target(weld,selected)
    start=engine.world.models[rid].tcp(np.asarray(selected['q_rad']));candidate={k:selected[k] for k in ('robot_id','roll_deg','axis_sign')}
    records=[];best=None
    for amplitude in AMPLITUDES:
        driver=-amplitude
        if not prefix_available(row['opening_profile'],amplitude):continue
        opened=engine.state(rid,start,driver,moved,engine.environment.relative_env,'opening')
        search=engine.search_exits(rid,start,moved,dict(open_state=opened),driver_mm=driver,count=2)
        rays=[dict(r,direction_frame='current_world') for r in search['exits']]
        no_geometry=not rays
        if no_geometry:rays=longest_blocked_rays(engine,rid,start,moved,search['release_context'],driver)
        frames=opening_frames(row,start,driver)
        for j,ray in enumerate(rays):
            baseline=lift_path(engine,weld,candidate,selected,ray,driver_mm=driver,opening_frames=frames)
            if baseline['passed'] and no_geometry:
                geom=engine.straight_exit(rid,start,moved,search['release_context'],ray['direction_world'],driver_mm=driver)
                if not geom['passed']:
                    baseline=dict(passed=False,reason='tool_geometry_not_certified',travel_mm=geom.get('failure_mm',0.),geometry=geom)
                elif geom['length_mm']!=ray['length_mm']:
                    ray.update(geom)
                    baseline=lift_path(engine,weld,candidate,selected,ray,driver_mm=driver,opening_frames=frames)
            if baseline['passed']:
                # Expanded no-exit direction diagnostic may itself reveal a baseline.
                name=f'paths/point_{index:02d}_amp_{amplitude:g}_unoptimized_{j}.json';atomic(out/name,baseline)
                record=dict(amplitude_mm=amplitude,direction_index=j,baseline_passed=True,qp_used=False,path_file=name,path_sha256=digest(out/name),retreat_passed=True)
                records.append(record);best=dict(record,driver_mm=driver);break
            qp,basis=build_qp(engine,weld,selected,driver,ray,baseline)
            record=dict(amplitude_mm=amplitude,driver_mm=driver,direction_index=j,baseline_passed=False,baseline_failure=baseline,
                        baseline_direction_world=ray['direction_world'],no_geometry_exit_in_previous_search=no_geometry,
                        qp_used=True,qp=qp,line_search=[],retreat_passed=False)
            if qp['solved']:
                for alpha in (1.,.5,.25,.125):
                    direction=updated_direction(ray['direction_world'],basis,qp['direction_step_tangent'],alpha)
                    trial_ray=dict(direction_world=direction.tolist(),direction_frame='current_world',length_mm=100.)
                    # Exact pure translation geometry certificate, then full source-arm lifting.
                    context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
                    geometry=engine.straight_exit(rid,start,moved,context,direction,driver_mm=driver)
                    # Even if geometry fails, compute partial arm path for a before/after diagnostic.
                    if geometry['passed']:trial_ray.update(geometry)
                    path=lift_path(engine,weld,candidate,selected,trial_ray,driver_mm=driver,opening_frames=frames)
                    trial=dict(alpha=alpha,direction_world=direction.tolist(),geometry_passed=geometry['passed'],
                               geometry=geometry,arm_passed=path['passed'],passed=geometry['passed'] and path['passed'])
                    if not path['passed']:trial['arm_failure']=path
                    record['line_search'].append(trial)
                    if trial['passed']:
                        if path['q_start_rad']!=selected['q_rad']:raise RuntimeError('QP changed archived welding q')
                        name=f'paths/point_{index:02d}_amp_{amplitude:g}_ray_{j}.json';atomic(out/name,path)
                        record.update(retreat_passed=True,path_file=name,path_sha256=digest(out/name),selected_alpha=alpha)
                        best=dict(amplitude_mm=amplitude,driver_mm=driver,path_file=name,path_sha256=digest(out/name),qp_used=True)
                        break
            records.append(record)
            atomic(out/f'point_{index:02d}_progress.json',dict(weld_id=row['weld_id'],trials=len(records),amplitude_mm=amplitude,retreat_passed=best is not None,elapsed_s=time.perf_counter()-started))
            print('QP',row['weld_id'],'AMP',amplitude,'RAY',j,'BASE',baseline['reason'],round(baseline.get('travel_mm',0.),3),
                  'QP',qp['status'],'PASS',record['retreat_passed'],'s',round(qp['total_wall_s'],2),flush=True)
            if best:break
        if best:break
    result=dict(weld_id=row['weld_id'],name=row['name'],initial=selected,initial_q_reoptimized=False,
                complete=True,retreat_passed=best is not None,best=best,trials=records,elapsed_s=time.perf_counter()-started)
    atomic(out/f'point_{index:02d}.json',result);return index,result


def main():
    global _ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('1 to4 workers')
    assert_frozen_geometry_cache();parent=ROOT/'build/welding_variable_open_v1/report.json';previous=json.loads(parent.read_text())
    baseline=previous['manifest']['source_sha256'];codes=previous['manifest']['code_sha256']
    for name,value in {**baseline,**codes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Frozen previous input/code changed: '+name)
    newcodes={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}
    manifest=dict(source_sha256=baseline,previous_code_sha256=codes,code_sha256=newcodes,parent_report_sha256=digest(parent),
                  settings=dict(amplitudes_mm=AMPLITUDES,qp_iterations_per_candidate=1,line_search_alpha=[1.,.5,.25,.125],
                  line_direction_optimized=True,start_q_orientation_positioner_park_locked=True,minimum_retreat_mm=100.,collision_linearization_margin_mm=.5))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    mp=out/'manifest.json'
    if mp.exists():
        if not args.resume:raise FileExistsError('Evidence exists; use --resume or new --out')
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('Code/input changed')
        if (out/'report.json').exists():
            done=json.loads((out/'report.json').read_text())
            if done['summary']['complete']:
                for name,value in done['output_sha256'].items():
                    if digest(out/name)!=value:raise RuntimeError('Output changed')
                print('QP_COMPLETED_READ_ONLY',json.dumps(done['summary']));return
    else:atomic(mp,manifest)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());welds={w['id']:w for w in scene['welds']}
    jobs=[];rows={}
    for i,row in enumerate(previous['rows']):
        if row['best']:continue
        saved=out/f'point_{i:02d}.json'
        if args.resume and saved.exists():rows[i]=json.loads(saved.read_text())
        else:jobs.append((i,row))
    started=time.perf_counter();load_s=0.
    if jobs:
        t=time.perf_counter();_ENGINE=VariableToolWorld(scene);load_s=time.perf_counter()-t
        print('QP_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            fs=[pool.submit(point_job,i,row,welds[row['weld_id']],out) for i,row in jobs]
            for future in concurrent.futures.as_completed(fs):
                i,rows[i]=future.result();atomic(out/'progress.json',dict(completed=len(rows),total=6,passed=sum(r['retreat_passed'] for r in rows.values()),wall_s=time.perf_counter()-started))
    for name,value in {**baseline,**codes,**newcodes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Input/code changed during QP experiment')
    outputs={f'point_{i:02d}.json':digest(out/f'point_{i:02d}.json') for i in rows}
    for r in rows.values():
        if r['best']:outputs[r['best']['path_file']]=digest(out/r['best']['path_file'])
    report=dict(manifest=manifest,summary=dict(complete=len(rows)==6,points=len(rows),additional_retreat_passed=sum(r['retreat_passed'] for r in rows.values()),
               actual_qp_solves=sum(t['qp_used'] for r in rows.values() for t in r['trials']),model_load_s=load_s,wall_s=time.perf_counter()-started),
               rows=[rows[i] for i in sorted(rows)],output_sha256=outputs,
               scope='One QP per initial ray/amplitude, shared direction update, original welding q and gun orientation locked. '
                     'QP collisional/slack intermediate states never accepted as executable trajectories. '
                     'Final success requires exact geometric line test plus original full-arm sampled interpolation checks. '
                     'No continuous robot certificate or positioner/inter-weld transfer validation.')
    atomic(out/'report.json',report);print('QP_SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
