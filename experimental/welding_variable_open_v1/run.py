"""Sweep partial gun opening without reselecting any initial robot posture."""
import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path
import fcl
import numpy as np
from variable_world import VariableToolWorld
from variable_motion import lift_path
from variable_core import opening_profile, prefix_available, driver_samples, completed_report
from tool_world import ROOT, HERE as OLD
from run_initial import assert_frozen_geometry_cache, extended_source_hashes
from run_arm import full_probe
from run_geometry import atomic, digest, initialize_worker

HERE = Path(__file__).parent
OUT = ROOT/'build/welding_variable_open_v1'
AMPLITUDES = [10.,20.,30.,35.]+[float(a) for a in range(40,181,10)]+[185.]
_ENGINE = None


def electrode_gap(engine, rid, target, driver):
    poses = engine.poses(rid,target,driver)
    electrodes = [pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
    if len(electrodes)!=2: raise RuntimeError('Expected two original source electrodes')
    objects = []
    for pid in electrodes:
        obj = engine.world.robots[rid]['gun:'+pid]; t=poses[pid]
        obj.setTransform(fcl.Transform(t[:3,:3],t[:3,3])); objects.append(obj)
    return float(fcl.distance(*objects,fcl.DistanceRequest(),fcl.DistanceResult()))


def point_job(index, weld, selected, out):
    engine = _ENGINE; begin = time.perf_counter()
    rid=selected['robot_id'];q0=np.asarray(selected['q_rad']);other=selected['other_robot_id'];park=np.asarray(selected['other_q_rad'])
    engine.set_angle(selected['positioner_deg']);moved,_=engine.target(weld,selected)
    actual=engine.world.models[rid].tcp(q0);candidate={k:selected[k] for k in ('robot_id','roll_deg','axis_sign')}
    layout=engine.environment.layout_collisions(selected['positioner_deg'])
    caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
    if layout['blocking_pairs']: raise RuntimeError('Archived layout unexpectedly blocked')
    profile=opening_profile(lambda d: full_probe(engine,rid,q0,other,park,d,moved,'closed' if d==0. else 'opening'),caps)
    trials=[];best=None
    for amplitude in AMPLITUDES:
        started=time.perf_counter();driver=-amplitude
        trial=dict(amplitude_mm=amplitude,driver_mm=driver,opening_passed=prefix_available(profile,amplitude),retreat_passed=False,
                   electrode_mesh_gap_mm=electrode_gap(engine,rid,actual,driver),exit_trials=[],path_file=None)
        if not trial['opening_passed']:
            trial.update(reason='opening_prefix_blocked',failure=profile['failure'])
        else:
            opened=engine.state(rid,actual,driver,moved,engine.environment.relative_env,'opening')
            exits=engine.search_exits(rid,actual,moved,dict(open_state=opened),driver_mm=driver,count=2)
            trial.update(tool_exit_count=len(exits['exits']),rays_tested=exits['rays_tested'],ray_failure_counts=exits['failure_counts'])
            prefix=[dict(q_rad=selected['q_rad'],driver_mm=float(d),phase='closed' if d==0. else 'opening',
                         tcp_xyz_mm=actual[:3,3].tolist(),travel_mm=0.) for d in profile['passed_drivers_mm'] if d>=driver]
            if not prefix or prefix[-1]['driver_mm']!=driver:raise RuntimeError('Requested amplitude is missing from certified prefix')
            for ray in exits['exits']:
                ray=dict(ray,direction_frame='current_world')
                path=lift_path(engine,weld,candidate,selected,ray,driver_mm=driver,opening_frames=prefix)
                if path['passed']:
                    if path['q_start_rad']!=selected['q_rad']:raise RuntimeError('Initial q changed')
                    if np.linalg.norm(np.asarray(path['direction_world'])-ray['direction_world'])>1e-10:raise RuntimeError('Exit direction double transformed')
                    path_name=f'paths/point_{index:02d}_amplitude_{amplitude:g}.json'
                    atomic(out/path_name,path)
                    trial.update(retreat_passed=True,reason='partial_open_and_straight_retreat_passed',path_file=path_name,path_sha256=digest(out/path_name),
                                 length_mm=path['length_mm'],endpoint_clearance_mm=path['endpoint_clearance_mm'])
                    if best is None:best=dict(trial)
                    break
                trial['exit_trials'].append(path)
            if not trial['retreat_passed']:trial['reason']='no_tool_exit_found' if not exits['exits'] else 'continuous_arm_retreat_failed'
        trial['elapsed_s']=time.perf_counter()-started;trials.append(trial)
        with (out/f'point_{index:02d}_events.jsonl').open('a') as stream:stream.write(json.dumps(trial,separators=(',',':'))+'\n')
        atomic(out/f'point_{index:02d}_progress.json',dict(weld_id=weld['id'],completed_amplitudes=len(trials),total_amplitudes=len(AMPLITUDES),
               smallest_successful_tested_amplitude_mm=best['amplitude_mm'] if best else None,elapsed_s=time.perf_counter()-begin))
        print('VARIABLE',index+1,weld['name'],'AMPLITUDE',amplitude,'OPEN',trial['opening_passed'],'RETREAT',trial['retreat_passed'],
              'seconds',round(trial['elapsed_s'],2),flush=True)
    result=dict(weld_id=weld['id'],name=weld['name'],initial=selected,initial_q_reoptimized=False,opening_profile=profile,
                trials=trials,best=best,elapsed_s=time.perf_counter()-begin,complete=True)
    atomic(out/f'point_{index:02d}.json',result)
    return index,result


def main():
    global _ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Use one to four isolated FCL processes')
    assert_frozen_geometry_cache();baseline=extended_source_hashes()
    baseline.update({str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'build/welding_tool_initial_exit_v2/initial_report.json',ROOT/'build/welding_tool_initial_exit_v2/initial_cold_verification.json']})
    code_files=[HERE/n for n in ('run.py','variable_world.py','variable_motion.py','variable_core.py')]
    code_files += [p for p in OLD.glob('*.py') if not p.name.startswith('test_')]
    manifest=dict(source_sha256=baseline,code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_files},
        settings=dict(amplitudes_mm=AMPLITUDES,initial_q_robot_roll_positioner_and_park_locked=True,
                      opening_driver_step_mm=1.,minimum_retreat_mm=100.,retained_directions=2))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    done=completed_report(out/'report.json',manifest,resume=args.resume)
    if done:
        for name,value in done['output_sha256'].items():
            if digest(out/name)!=value:raise RuntimeError('Saved evidence changed: '+name)
        print('COMPLETED_READ_ONLY',json.dumps(done['summary']),flush=True);return
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if not args.resume:raise FileExistsError('Evidence exists; use --resume or a new --out')
        if json.loads(manifest_path.read_text())!=manifest:raise RuntimeError('Source/code/settings changed')
    else:atomic(manifest_path,manifest)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    previous=json.loads((ROOT/'build/welding_positioner_static_v3/report.json').read_text())
    full=json.loads((ROOT/'build/welding_tool_initial_exit_v2/initial_report.json').read_text())
    old_rows={r['weld_id']:r for r in full['rows']};rows={};jobs=[]
    for i,row in enumerate(previous['rows']):
        if not row['passed']:continue
        saved=out/f'point_{i:02d}.json'
        if args.resume and saved.exists():
            record=json.loads(saved.read_text())
            if not record['complete'] or record['initial']!=row['selected']:raise RuntimeError('Incomplete or changed saved point')
            for trial in record['trials']:
                if trial['path_file'] and digest(out/trial['path_file'])!=trial['path_sha256']:raise RuntimeError('Saved path changed')
            rows[i]=record
        else:jobs.append(i)
    start=time.perf_counter();load_s=0.
    if jobs:
        t=time.perf_counter();_ENGINE=VariableToolWorld(scene);load_s=time.perf_counter()-t
        print('VARIABLE_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            fs=[pool.submit(point_job,i,scene['welds'][i],previous['rows'][i]['selected'],out) for i in jobs]
            for future in concurrent.futures.as_completed(fs):
                i,rows[i]=future.result()
                atomic(out/'progress.json',dict(completed_points=len(rows),total_points=21,any_amplitude_passed=sum(r['best'] is not None for r in rows.values()),wall_s=time.perf_counter()-start))
    now=extended_source_hashes();now.update({n:digest(ROOT/n) for n in baseline if n not in now})
    if now!=baseline:raise RuntimeError('Source/historical evidence changed during experiment')
    mismatches=[]
    for row in rows.values():
        trial=row['trials'][-1];old=old_rows[row['weld_id']]
        if trial['opening_passed']!=old['opening_passed'] or trial['retreat_passed']!=old['retreat_passed']:mismatches.append(row['weld_id'])
    counts=[dict(amplitude_mm=a,opening_passed=sum(r['trials'][j]['opening_passed'] for r in rows.values()),
                 open_and_retreat_passed=sum(r['trials'][j]['retreat_passed'] for r in rows.values())) for j,a in enumerate(AMPLITUDES)]
    outputs={f'point_{i:02d}.json':digest(out/f'point_{i:02d}.json') for i in rows}
    for r in rows.values():
        for trial in r['trials']:
            if trial['path_file']:outputs[trial['path_file']]=digest(out/trial['path_file'])
    report=dict(schema='variable_opening_locked_initial_v1',manifest=manifest,summary=dict(complete=len(rows)==21,points=len(rows),
                tested_amplitudes=len(AMPLITUDES),any_amplitude_passed=sum(r['best'] is not None for r in rows.values()),
                full_opening_baseline_matches=not mismatches,baseline_mismatches=mismatches,model_load_s=load_s,wall_s=time.perf_counter()-start),
                by_amplitude=counts,rows=[rows[i] for i in sorted(rows)],output_sha256=outputs,source_and_historical_evidence_unchanged=True,
                scope='Locked original 21 starts; grid minimum only; same sampled source contact and both-arm checks as full-opening v2. '
                      'No complete-system continuous safety proof; source assembly exemptions and prior external axis states retained.')
    atomic(out/'report.json',report)
    print('VARIABLE_SUMMARY',json.dumps(report['summary']),flush=True)
    if mismatches:raise SystemExit(2)


if __name__=='__main__':main()
