"""Direction-only SQP with full-line nonlinear collision and source-limit merit."""
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
from run_qp import opening_frames
from variable_world import VariableToolWorld
from run_geometry import atomic,digest,initialize_worker
from run_initial import assert_frozen_geometry_cache
from tool_world import ROOT
from qp_step import updated_direction
from trace_v3 import lift_path
from reduced import evaluate_line,reduced_step
OUT=ROOT/'build/welding_waypoint_qp_v3';_ENGINE=None


def public_evaluation(e):
    return {k:v for k,v in e.items() if k!='samples'}


def point_job(index,parent_row,old_row,weld,out,max_iterations):
    engine=_ENGINE;started=time.perf_counter();selected=parent_row['initial'];rid=selected['robot_id']
    engine.set_angle(selected['positioner_deg']);moved,_=engine.target(weld,selected)
    start=engine.world.models[rid].tcp(np.asarray(selected['q_rad']));candidate={k:selected[k] for k in ('robot_id','roll_deg','axis_sign')}
    # Start from two strongest v2 outcomes, avoiding nearly identical directions.
    ranked=sorted(old_row['trials'],key=lambda t:-t['final']['score']);seeds=[]
    for t in ranked:
        if any(t['amplitude_mm']==s['amplitude_mm'] and np.dot(t['direction_world'],s['direction_world'])>np.cos(np.deg2rad(10.)) for s in seeds):continue
        seeds.append(t)
        if len(seeds)>=3:break
    trials=[];best=None
    for seed_index,seed in enumerate(seeds):
        amplitude=seed['amplitude_mm'];driver=-amplitude;direction=np.asarray(seed['direction_world']);frames=opening_frames(parent_row,start,driver)
        current=evaluate_line(engine,weld,selected,driver,direction);baseline=public_evaluation(current);history=[];reason='iteration_limit'
        def validate(d):
            opened=engine.state(rid,start,driver,moved,engine.environment.relative_env,'opening')
            context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
            geom=engine.straight_exit(rid,start,moved,context,d,driver_mm=driver)
            ray=dict(direction_world=d.tolist(),direction_frame='current_world',length_mm=100.)
            if geom['passed']:ray.update(geom)
            arm=lift_path(engine,weld,candidate,selected,ray,driver_mm=driver,opening_frames=frames)
            return geom['passed'] and arm['passed'],geom,arm
        initial_pass,geometry,arm=validate(direction)
        passed=initial_pass
        if not passed:
            for iteration in range(max_iterations):
                qp,basis=reduced_step(engine,selected,driver,current,trust_deg=15.)
                record=dict(iteration=iteration+1,qp=qp,baseline=public_evaluation(current),line_search=[])
                if not qp['solved']:history.append(record);reason='qp_not_solved';break
                chosen=None
                for alpha in (1.,.5,.25,.125,.0625):
                    next_direction=updated_direction(direction,basis,qp['direction_step_tangent'],alpha)
                    evaluation=evaluate_line(engine,weld,selected,driver,next_direction)
                    record['line_search'].append(dict(alpha=alpha,**public_evaluation(evaluation)))
                    if evaluation['merit']<current['merit']-max(1e-5,current['merit']*1e-5):
                        if chosen is None or evaluation['merit']<chosen[1]['merit']:chosen=(next_direction,evaluation,alpha)
                if chosen is None:history.append(record);reason='nonlinear_merit_stagnation';break
                direction,current,alpha=chosen;record['accepted_alpha']=alpha;record['accepted']=public_evaluation(current)
                passed,geometry,arm=validate(direction)
                record['full_path_passed']=passed
                record['full_arm_failure']=None if arm['passed'] else {k:v for k,v in arm.items() if k!='accepted_retreat_waypoints'}
                record['geometry']=geometry;history.append(record)
                print('REDUCED_STEP',parent_row['weld_id'],'SEED',seed_index,'ITER',iteration+1,'MERIT',round(current['merit'],5),
                      'FIRST_INVALID',current['first_invalid_sample_mm'],'PASS',passed,flush=True)
                if passed:reason='full_path_passed';break
        trial=dict(amplitude_mm=amplitude,driver_mm=driver,seed_index=seed_index,baseline=baseline,final=public_evaluation(current),
                   direction_world=direction.tolist(),iterations=len(history),history=history,termination_reason=reason,retreat_passed=passed)
        if passed:
            if arm['q_start_rad']!=selected['q_rad']:raise RuntimeError('Archived q0 changed')
            name=f'paths/point_{index:02d}_seed_{seed_index}.json';atomic(out/name,arm)
            best=dict(amplitude_mm=amplitude,driver_mm=driver,path_file=name,path_sha256=digest(out/name),iterations=len(history))
            trial.update(path_file=name,path_sha256=digest(out/name))
        trials.append(trial)
        atomic(out/f'point_{index:02d}_progress.json',dict(weld_id=parent_row['weld_id'],seeds=len(trials),passed=best is not None,merit=current['merit'],elapsed_s=time.perf_counter()-started))
        print('REDUCED_CANDIDATE',parent_row['weld_id'],'SEED',seed_index,'PASS',passed,'MERIT',round(current['merit'],6),reason,flush=True)
        if best:break
    result=dict(weld_id=parent_row['weld_id'],name=parent_row['name'],initial=selected,complete=True,retreat_passed=best is not None,best=best,trials=trials,elapsed_s=time.perf_counter()-started)
    atomic(out/f'point_{index:02d}.json',result);return index,result


def main():
    global _ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-iterations',type=int,default=20);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=4 or not 1<=args.max_iterations<=40:raise ValueError('Invalid worker or iteration settings')
    assert_frozen_geometry_cache();parent=ROOT/'build/welding_variable_open_v1/report.json';previous=json.loads(parent.read_text())
    v2path=ROOT/'build/welding_waypoint_qp_v2/report.json';v2=json.loads(v2path.read_text());old={r['weld_id']:r for r in v2['rows']}
    baseline=previous['manifest']['source_sha256'];old_codes=previous['manifest']['code_sha256']
    files=[HERE/n for n in ('run_reduced.py','reduced.py','trace_v3.py')]+[HERE.parent/'welding_waypoint_qp_v1'/n for n in ('qp_step.py','proximity.py','run_qp.py')]
    codes={str(p.relative_to(ROOT)):digest(p) for p in files}
    for name,value in {**baseline,**old_codes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Frozen source/code changed')
    manifest=dict(source_sha256=baseline,previous_code_sha256=old_codes,code_sha256=codes,parent_report_sha256=digest(parent),v2_report_sha256=digest(v2path),
                 settings=dict(max_qp_iterations=args.max_iterations,direction_only_reduced_qp=True,start_q_orientation_positioner_park_locked=True,
                    initial_seeds='top3 geometrically distinct v2 optimized outcomes',nonlinear_merit='source-limit violation+local mesh penetration/clearance proxies',
                    final_acceptance='exact geometric certificate and full original arm checks',internal_math_ik_can_violate_limits=True))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
    if mp.exists():
        if not args.resume:raise FileExistsError('Use --resume or new output')
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('Input/code/settings changed')
        if (out/'report.json').exists():
            done=json.loads((out/'report.json').read_text())
            if done['summary']['complete']:
                for n,v in done['output_sha256'].items():
                    if digest(out/n)!=v:raise RuntimeError('Output changed')
                print('REDUCED_READ_ONLY',json.dumps(done['summary']));return
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
        print('REDUCED_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            fs=[pool.submit(point_job,i,row,old[row['weld_id']],welds[row['weld_id']],out,args.max_iterations) for i,row in jobs]
            for future in concurrent.futures.as_completed(fs):
                i,rows[i]=future.result();atomic(out/'progress.json',dict(completed=len(rows),total=6,additional_passed=sum(r['retreat_passed'] for r in rows.values()),wall_s=time.perf_counter()-started))
    for n,v in {**baseline,**old_codes,**codes}.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Source/code changed during reduced-QP experiment')
    outputs={f'point_{i:02d}.json':digest(out/f'point_{i:02d}.json') for i in rows}
    for r in rows.values():
        if r['best']:outputs[r['best']['path_file']]=digest(out/r['best']['path_file'])
    report=dict(manifest=manifest,summary=dict(complete=len(rows)==6,points=len(rows),additional_retreat_passed=sum(r['retreat_passed'] for r in rows.values()),
           actual_qp_solves=sum(t['iterations'] for r in rows.values() for t in r['trials']),model_load_s=load_s,wall_s=time.perf_counter()-started),
           rows=[rows[i] for i in sorted(rows)],output_sha256=outputs,
           scope='Direction-only QP after exact pose-Jacobian joint elimination. Internal math branches can be outside source limits only for violation evaluation. '
                 'All reported successes strictly re-lift from archived q0 with original source limits, contact, exact tool geometry and full-arm collision checks. '
                 'No inter-weld/positioner transition or continuous full-system safety certificate.')
    atomic(out/'report.json',report);print('REDUCED_SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
