"""Finite 5-degree weld-pose search for additional literal 200cm retreats.

An endpoint IK prefilter only rejects candidates. Positive evidence still
requires source contact checks, interval geometry, and full continuous-IK trace.
This is a research experiment, not a replacement for archived pose results.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import time
from pathlib import Path
import copy
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'welding_variable_open_v1')]
import run_retreat_v2 as retreat
from run_geometry import atomic, digest, initialize_worker
from variable_core import opening_profile, prefix_available
from run_arm import full_probe
from run_qp import opening_frames
from analytic_trace_motion_v2 import lift_path
from refinement import roll_target
from analytic_ik_v2 import analytical_seed_bank as solve_seed_bank
from positioner import transform_weld
from probe_neighbor_rays_v17 import kinematic_rejection

ENGINE = None


def point_job(weld_id, out, pose_budget, direction_budget, positioner_deg):
    start_time = time.perf_counter()
    engine = ENGINE
    parent = json.loads((ROOT/'build/welding_variable_open_v1/report.json').read_text())
    original = next(r for r in parent['rows'] if r['weld_id'] == weld_id)
    static = json.loads((ROOT/'build/welding_positioner_static_v3/report.json').read_text())
    candidates = next(r for r in static['rows'] if r['weld_id'] == weld_id)['allowed']
    old = dict(original['initial'],positioner_deg=positioner_deg); rid = old['robot_id']; other = old['other_robot_id']
    layout=engine.environment.layout_collisions(positioner_deg)
    if layout['blocking_pairs']:
        return dict(weld_id=weld_id,initial=old,best=None,passed=False,reason='positioner_layout_blocked')
    weld = next(w for w in engine.scene['welds'] if w['id'] == weld_id)
    delta = engine.set_angle(old['positioner_deg'])
    moved, base = transform_weld(engine.scene, weld, delta)
    model = engine.world.models[rid]
    remote = json.loads((ROOT/'build/welding_retreat_200cm_v2/paths/retreat_16.json').read_text())
    ray_success = np.asarray(remote['direction_world'])
    choices = list(candidates)
    # Try the successful sibling's sign/roll chart, then nearby 5-degree rolls.
    choices.sort(key=lambda c: (min(abs(c['roll_deg']-105), 360-abs(c['roll_deg']-105)),
                                c['axis_sign'] != 1))
    records, best, selected_best = [], None, None
    kinematic_cache = {}
    for ci, candidate in enumerate(choices[:pose_budget]):
        rid = candidate['robot_id']
        other = next(r for r in engine.world.models if r != rid)
        model = engine.world.models[rid]
        old = dict(original['initial'],positioner_deg=positioner_deg,robot_id=rid,other_robot_id=other,
                   q_rad=(original['initial']['q_rad'] if rid == original['initial']['robot_id'] else model.q_initial.tolist()),
                   other_q_rad=[-3.,-.8,.5,0.,0.,float(np.pi/2)])
        target = roll_target(base, candidate['roll_deg'], candidate['axis_sign'])
        solutions, stats = solve_seed_bank(model, target,
            [old['q_rad'], remote['q_start_rad']], random_count=4)
        record = dict(candidate=candidate, ik_stats=stats, starts=len(solutions), trials=[])
        records.append(record)
        print('ALTERNATIVE_POSE', weld_id, ci, candidate['roll_deg'],candidate['axis_sign'],len(solutions),flush=True)
        for solution in solutions:
            selected = dict(old, **candidate, q_rad=solution['q_rad'],
                            target_tcp_matrix=target.ravel().tolist(),
                            actual_tcp_matrix=model.tcp(np.asarray(solution['q_rad'])).ravel().tolist())
            q = np.asarray(solution['q_rad']); actual = model.tcp(q)
            caps = [pid for pid, (_, kind, _) in engine.parts[rid].items() if kind == 'electrode']
            profile = opening_profile(lambda driver: full_probe(engine,rid,q,other,
                np.asarray(old['other_q_rad']),driver,moved,'closed' if driver == 0. else 'opening'),
                caps, maximum_amplitude_mm=80.)
            record['trials'].append(dict(q_rad=q.tolist(), opening_profile=profile, directions=[]))
            state = record['trials'][-1]
            row = dict(initial=selected, best=None, opening_profile=profile, _moved=moved)
            for amplitude in (10., 30., 60., 80.):
                if not prefix_available(profile, amplitude):
                    continue
                driver = -amplitude
                directions = retreat.trial_directions(engine,row,actual,driver,
                                      ROOT/'build/welding_variable_open_v1')
                # Preserve the original bank, choose those nearest the successful
                # sibling's outward ray, keeping the azimuth 5-degree grid.
                directions.sort(key=lambda ray: -float(ray@ray_success))
                for direction in directions[:direction_budget]:
                    end = actual.copy(); end[:3,3] += 2000.*direction
                    ends, endpoint_stats = solve_seed_bank(model,end,
                        [remote['q_end_rad'],q],random_count=0)
                    trial = dict(amplitude_mm=amplitude,direction_world=direction.tolist(),
                                 endpoint_ik_candidates=len(ends),endpoint_ik_stats=endpoint_stats)
                    state['directions'].append(trial)
                    if not ends:
                        trial['reason'] = 'finite_endpoint_ik_prefilter_rejected'; continue
                    clear = True
                    for travel in np.arange(5., 2000.+1e-7, 10.):
                        pose = actual.copy(); pose[:3,3] += travel*direction
                        if engine.state(rid,pose,driver,moved,engine.world.static,'separated')['blocking_pairs']:
                            clear = False; break
                    if not clear:
                        trial['reason'] = 'coarse_geometry_rejected'; continue
                    geom = retreat.geometry_prefix(engine,rid,actual,moved,driver,direction,2000.)
                    if not geom['passed']:
                        trial.update(reason='geometry_certificate_failed',geometry=geom); continue
                    kin_key = (rid,tuple(np.round(np.r_[q,direction],12)))
                    if kin_key not in kinematic_cache:
                        kinematic_cache[kin_key] = kinematic_rejection(model,actual,q,direction)
                    trial['kinematic_prefilter'] = kinematic_cache[kin_key]
                    if not kinematic_cache[kin_key]['passed']:
                        trial['reason'] = 'finite_continuation_prefilter_rejected'; continue
                    path = lift_path(engine,weld,selected,selected,
                        dict(direction_world=direction.tolist(),direction_frame='current_world',length_mm=2000.),
                        driver_mm=driver,opening_frames=opening_frames(row,actual,driver))
                    trial.update(reason=path.get('reason','passed'),arm_passed=path['passed'],
                                 last_accepted_travel_mm=path.get('last_accepted_travel_mm'))
                    print('ALTERNATIVE_TRACE',weld_id,candidate['roll_deg'],candidate['axis_sign'],
                          amplitude,path['passed'],path.get('last_accepted_travel_mm'),flush=True)
                    if path['passed']:
                        name = f'paths/retreat_{weld_id}_angle_{positioner_deg:g}.json'
                        path.update(weld_id=weld_id,fixed_distance_mm=2000.,no_qp=True,
                                    geometry_certificate=geom,welding_posture_reselected=True)
                        atomic(out/name,path)
                        best=dict(driver_mm=driver,amplitude_mm=amplitude,path_file=name,
                                  path_sha256=digest(out/name),length_mm=2000.)
                        selected_best=selected; break
                if best: break
            if best: break
        if best: break
        atomic(out/f'{weld_id}_angle_{positioner_deg:g}_progress.json',dict(completed_poses=ci+1,elapsed_s=time.perf_counter()-start_time))
    result = dict(weld_id=weld_id,initial=selected_best or old,best=best,passed=bool(best),
                  original_initial=original['initial'],welding_posture_reselected=bool(best),required_straight_distance_mm=2000.,
                  alternative_pose_trials=records,elapsed_s=time.perf_counter()-start_time)
    atomic(out/f'{weld_id}_angle_{positioner_deg:g}.json',result)
    print('ALTERNATIVE_POINT',weld_id,bool(best),result['elapsed_s'],flush=True)
    return result


def inventory_job(row, out, angles):
    results=[]
    sequence=[row['initial']['positioner_deg']]+[a for a in angles if a != row['initial']['positioner_deg']]
    for angle in sequence:
        result=point_job(row['weld_id'],out,10000,10000,angle)
        results.append(result)
        if result['best']:break
    return dict(weld_id=row['weld_id'],best_result=next((r for r in results if r['best']),None),
                searched_angles=sequence[:len(results)],records=results)


def main():
    global ENGINE
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,30.,-30.,60.,-60.,90.,-90.,120.,-120.,150.,-150.,180.])
    parser.add_argument('--weld-id',action='append')
    args=parser.parse_args()
    if not 1 <= args.workers <= 2:raise ValueError('One or two workers')
    if args.out.exists():raise FileExistsError('Fresh directory required')
    retreat.assert_frozen_geometry_cache()
    prior_dir=ROOT/'build/welding_retreat_200cm_other_park_v12'
    prior=json.loads((prior_dir/'report.json').read_text())
    if not prior['summary']['complete'] or prior['summary']['required_straight_distance_mm'] != 2000.:raise RuntimeError('Completed2m baseline')
    sources={**prior['manifest']['source_sha256'],**prior['manifest']['code_sha256'],str((prior_dir/'report.json').relative_to(ROOT)):digest(prior_dir/'report.json')}
    static_path=ROOT/'build/welding_positioner_static_v3/report.json'
    sources[str(static_path.relative_to(ROOT))]=digest(static_path)
    for name,sha in prior['output_sha256'].items():
        if digest(prior_dir/name)!=sha:raise RuntimeError('Prior evidence changed')
        sources[str((prior_dir/name).relative_to(ROOT))]=sha
    codes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),HERE/'probe_neighbor_rays_v17.py',HERE/'analytic_ik_v2.py',HERE/'analytic_trace_motion_v2.py')}
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Source changed '+name)
    out=args.out.resolve();out.mkdir(parents=True)
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(distance_mm=2000.,posture_roll_step_deg=5.,angles=args.angles,workers=args.workers,both_robot_assignment=True,explicit_other_park=[-3.,-.8,.5,0.,0.,float(np.pi/2)],no_qp=True,kinematic_prefilter_reject_only=True))
    atomic(out/'manifest.json',manifest)
    rows=copy.deepcopy(prior['rows'])
    for i,row in enumerate(rows):
        if row['best']:
            src=prior_dir/row['best']['path_file'];name=f'paths/reused_{i:02d}.json'
            (out/name).parent.mkdir(parents=True,exist_ok=True);(out/name).write_bytes(src.read_bytes())
            row['best'].update(path_file=name,path_sha256=digest(out/name))
    original=json.loads((ROOT/'build/welding_variable_open_v1/report.json').read_text())
    jobs=[r for r in original['rows'] if not next(x for x in rows if x['weld_id']==r['weld_id'])['best'] and (not args.weld_id or r['weld_id'] in args.weld_id)]
    if args.weld_id and not set(args.weld_id) <= {r['weld_id'] for r in original['rows']}:raise ValueError('Unknown weld')
    ENGINE=retreat.VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    started=time.perf_counter();completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        fs=[pool.submit(inventory_job,r,out,args.angles) for r in jobs]
        for future in concurrent.futures.as_completed(fs):
            result=future.result();completed.append(result)
            if result['best_result']:
                rows=[result['best_result'] if r['weld_id']==result['weld_id'] else r for r in rows]
            atomic(out/'progress.json',dict(finished=len(completed),requested=len(jobs),passed=sum(bool(r['best']) for r in rows),elapsed_s=time.perf_counter()-started))
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Input changed during run')
    output={str(p.relative_to(out)):digest(p) for p in out.glob('OP10L*.json')}
    output.update({str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')})
    summary=dict(complete=True,points=21,passed=sum(bool(r['best']) for r in rows),requested_points=len(jobs),required_straight_distance_mm=2000.,wall_s=time.perf_counter()-started)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,search_records=completed,output_sha256=output,scope='Finite source5deg pose/two-robot/external-axis search. Fixed explicit qualified-by-source-probe other park. Negative not impossibility proof. Positives complete2m source trace require cold audit and15axis transfers.'))
    print('CELL_INVENTORY_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
