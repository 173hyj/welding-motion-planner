"""Iterate QPs on the current nonlinear failure, keep original welding q0."""
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
from run_qp import opening_frames,longest_blocked_rays
from variable_world import VariableToolWorld
from variable_core import prefix_available
from run_geometry import atomic,digest,initialize_worker
from run_initial import assert_frozen_geometry_cache
from tool_world import ROOT
from qp_step import updated_direction
from trace_motion import lift_path
from local_qp import local_step
from iterative import optimize_iterations

OUT=ROOT/'build/welding_waypoint_qp_v2'
_ENGINE=None
AMPLITUDES=[30.,60.,80.,185.]


def summarize(evaluation):
    arm=evaluation['arm'];geom=evaluation['geometry']
    return dict(passed=evaluation['passed'],score=evaluation['score'],arm_passed=arm['passed'],geometry_passed=geom['passed'],
                last_accepted_travel_mm=arm.get('last_accepted_travel_mm',arm.get('length_mm',0.)),
                failure_reason=arm.get('reason'),failure_travel_mm=arm.get('travel_mm'),
                geometry_failure_reason=geom.get('reason'),geometry_failure_mm=geom.get('failure_mm'))


def point_job(index,row,weld,out,max_iterations):
    engine=_ENGINE;started=time.perf_counter();selected=row['initial'];rid=selected['robot_id']
    engine.set_angle(selected['positioner_deg']);moved,_=engine.target(weld,selected)
    start=engine.world.models[rid].tcp(np.asarray(selected['q_rad']));candidate={k:selected[k] for k in ('robot_id','roll_deg','axis_sign')}
    trials=[];best=None
    for amplitude in AMPLITUDES:
        if not prefix_available(row['opening_profile'],amplitude):continue
        driver=-amplitude;frames=opening_frames(row,start,driver)
        opened=engine.state(rid,start,driver,moved,engine.environment.relative_env,'opening')
        search=engine.search_exits(rid,start,moved,dict(open_state=opened),driver_mm=driver,count=2)
        rays=[dict(r,direction_frame='current_world') for r in search['exits']]
        if not rays:rays=longest_blocked_rays(engine,rid,start,moved,search['release_context'],driver)
        for ray_index,ray in enumerate(rays):
            cache={};steps=[];all_evaluations=[];current_direction=None
            def key(d):return tuple(np.round(np.asarray(d),12))
            def evaluate(direction):
                k=key(direction)
                if k in cache:return cache[k]
                context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
                geometry=engine.straight_exit(rid,start,moved,context,direction,driver_mm=driver)
                exit_path=dict(direction_world=list(direction),direction_frame='current_world',length_mm=100.)
                if geometry['passed']:exit_path.update(geometry)
                arm=lift_path(engine,weld,candidate,selected,exit_path,driver_mm=driver,opening_frames=frames)
                passed=geometry['passed'] and arm['passed']
                arm_progress=arm.get('length_mm',100.) if arm['passed'] else arm.get('last_accepted_travel_mm',0.)
                if geometry['passed']:geo_progress=100.
                elif geometry.get('failure_mm') is not None:geo_progress=max(0.,min(100.,geometry['failure_mm']-5.))
                else:geo_progress=0.
                progress=min(100.,arm_progress,geo_progress)
                result=dict(passed=passed,score=float(progress),arm=arm,geometry=geometry,direction_world=list(direction))
                cache[k]=result;all_evaluations.append(result);return result
            def step(direction):
                evaluation=evaluate(direction);qp,basis=local_step(engine,weld,selected,driver,np.asarray(direction),evaluation)
                steps.append(qp);proposals=[]
                if qp['solved']:
                    for alpha in (1.,.5,.25,.125,.0625):
                        proposals.append(updated_direction(direction,basis,qp['direction_step_tangent'],alpha).tolist())
                print('SQP_STEP',row['weld_id'],'AMP',amplitude,'RAY',ray_index,'ITER',len(steps),'PROGRESS',round(evaluation['score'],3),
                      'QP',qp['status'],'STEP',qp.get('direction_step_tangent'),flush=True)
                return dict(solved=qp['solved'],proposals=proposals,status=qp['status'],qp_record_index=len(steps)-1)
            # The evaluator and controller can only accept exact nonlinear improvement.
            fit=optimize_iterations(ray['direction_world'],step,evaluate,max_iterations=max_iterations)
            record=dict(amplitude_mm=amplitude,driver_mm=driver,ray_index=ray_index,baseline=summarize(evaluate(ray['direction_world'])),
                        final=summarize(fit['evaluation']),iterations=fit['iterations'],termination_reason=fit['reason'],retreat_passed=fit['passed'],
                        direction_world=fit['state'],qp_steps=steps,
                        history=[dict(iteration=h['iteration'],update=h['update'],candidates=[dict(direction_world=c['state'],evaluation=summarize(c['evaluation'])) for c in h['candidates']],
                                 accepted_direction=h.get('accepted_state')) for h in fit['history']])
            if fit['passed']:
                path=fit['evaluation']['arm']
                if path['q_start_rad']!=selected['q_rad']:raise RuntimeError('Welding start changed')
                name=f'paths/point_{index:02d}_amp_{amplitude:g}_ray_{ray_index}.json';atomic(out/name,path)
                record.update(path_file=name,path_sha256=digest(out/name))
                best=dict(amplitude_mm=amplitude,driver_mm=driver,path_file=name,path_sha256=digest(out/name),iterations=fit['iterations'])
            trials.append(record)
            atomic(out/f'point_{index:02d}_progress.json',dict(weld_id=row['weld_id'],trials=len(trials),passed=best is not None,
                 current_amplitude_mm=amplitude,best_actual_progress_mm=max(t['final']['score'] for t in trials),elapsed_s=time.perf_counter()-started))
            print('SQP_CANDIDATE',row['weld_id'],'AMP',amplitude,'RAY',ray_index,'PASS',fit['passed'],'PROGRESS',round(fit['evaluation']['score'],3),
                  'ITERS',fit['iterations'],fit['reason'],flush=True)
            if best:break
        if best:break
    result=dict(weld_id=row['weld_id'],name=row['name'],initial=selected,complete=True,retreat_passed=best is not None,best=best,
                trials=trials,elapsed_s=time.perf_counter()-started)
    atomic(out/f'point_{index:02d}.json',result);return index,result


def main():
    global _ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-iterations',type=int,default=12);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=4 or not 1<=args.max_iterations<=30:raise ValueError('Invalid workers or iterations')
    assert_frozen_geometry_cache();parent=ROOT/'build/welding_variable_open_v1/report.json';previous=json.loads(parent.read_text())
    baseline=previous['manifest']['source_sha256'];old_codes=previous['manifest']['code_sha256']
    files=[p for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    files += [HERE.parent/'welding_waypoint_qp_v1'/n for n in ('qp_step.py','proximity.py','run_qp.py')]
    codes={str(p.relative_to(ROOT)):digest(p) for p in files}
    for name,value in {**baseline,**old_codes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Frozen source/code differs')
    manifest=dict(source_sha256=baseline,previous_code_sha256=old_codes,code_sha256=codes,parent_report_sha256=digest(parent),
                  settings=dict(amplitudes_mm=AMPLITUDES,max_qp_iterations=args.max_iterations,straight_retreat=True,
                      start_q_positioner_orientation_other_park_locked=True,trust_direction_deg=15.,minimum_retreat_mm=100.,
                      acceptance='only nonlinear prefix progress increase or full path success'))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
    if mp.exists():
        if not args.resume:raise FileExistsError('Use --resume or new --out')
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('Input/code/settings changed')
        if (out/'report.json').exists():
            done=json.loads((out/'report.json').read_text())
            if done['summary']['complete']:
                for name,value in done['output_sha256'].items():
                    if digest(out/name)!=value:raise RuntimeError('Saved output changed')
                print('SQP_READ_ONLY',json.dumps(done['summary']));return
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
        print('SQP_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            fs=[pool.submit(point_job,i,row,welds[row['weld_id']],out,args.max_iterations) for i,row in jobs]
            for future in concurrent.futures.as_completed(fs):
                i,rows[i]=future.result();atomic(out/'progress.json',dict(completed=len(rows),total=6,additional_passed=sum(r['retreat_passed'] for r in rows.values()),wall_s=time.perf_counter()-started))
    for name,value in {**baseline,**old_codes,**codes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Input/code changed during sequential QP')
    outputs={f'point_{i:02d}.json':digest(out/f'point_{i:02d}.json') for i in rows}
    for r in rows.values():
        if r['best']:outputs[r['best']['path_file']]=digest(out/r['best']['path_file'])
    report=dict(manifest=manifest,summary=dict(complete=len(rows)==6,points=len(rows),additional_retreat_passed=sum(r['retreat_passed'] for r in rows.values()),
           actual_qp_solves=sum(t['iterations'] for r in rows.values() for t in r['trials']),model_load_s=load_s,wall_s=time.perf_counter()-started),
           rows=[rows[i] for i in sorted(rows)],output_sha256=outputs,
           scope='Sequential QPs with source-mesh gradients at pose-qualified local states near actual failures; shared straight direction, fixed welding q/orientation. '
                 'Only full exact geometry and source-arm interpolation acceptance counts as success. No continuous full-system safety certificate.')
    atomic(out/'report.json',report);print('SQP_SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
