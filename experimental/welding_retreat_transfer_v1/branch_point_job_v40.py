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

def point_job(weld_id, out, pose_budget, direction_budget, positioner_deg, other_parks):
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
                   other_q_rad=list(other_parks[other]))
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

