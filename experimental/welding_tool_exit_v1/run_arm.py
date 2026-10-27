"""Lift tool exits to continuous source-legal arm branches and check both arms."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tool_world import ToolWorld, ROOT, HERE, closed_contact_passed
from run_geometry import OUT, atomic, digest, source_hashes, initialize_worker
from arm_path import continuous_ik, interpolation_probes, line_error
from contracts import ContactTracker
sys.path.insert(0,str(ROOT/'experimental/welding_positioner_v3'))
from fast_ik import iter_solutions
from r2_safe_park import additional_self_checks

_ENGINE = None


def resolve_exit_direction(exit_path, delta):
    direction = np.asarray(exit_path['direction_world'],float)
    frame = exit_path.get('direction_frame','source_positioner_zero')
    if frame == 'source_positioner_zero':
        direction = delta[:3,:3]@direction
    elif frame != 'current_world':
        raise ValueError('Unknown exit direction frame: '+frame)
    return direction/np.linalg.norm(direction)


def qualify_full_pairs(pairs, rid, product_id, tool_state, phase):
    if phase not in ('closed','opening','release') or tool_state['blocking_pairs']:
        return list(pairs)
    allowed = {'gun:'+pid for pid,cert in tool_state['certificates'].items() if cert['passed']}
    return [p for p in pairs if not (p.get('checked_robot_id')==rid and p.get('asset_id')==product_id
            and p.get('link') in allowed and 'distance_mm' not in p)]


def full_probe(engine, rid, q, other, park, driver, weld, phase, context=None):
    world = engine.world; model = world.models[rid]
    if not model.limits_valid(q) or not world.models[other].limits_valid(park):
        return dict(passed=False,reason='source_joint_limits',remaining_pairs=[])
    actual = model.tcp(q)
    tool = engine.state(rid,actual,driver,weld,world.static,phase,context)
    world.update(other,park,0.); world.update(rid,q,driver)
    pairs = []
    for checked, state, driver_value in [(rid,q,driver),(other,park,0.)]:
        raw = world.check(checked,state,gun_driver_mm=driver_value,max_pairs=sys.maxsize)
        raw += additional_self_checks(world,checked)
        pairs.extend(dict(p,checked_robot_id=checked) for p in raw)
    remaining = qualify_full_pairs(pairs,rid,weld['product_id'],tool,phase)
    return dict(passed=not remaining and not tool['blocking_pairs'],
        remaining_pairs=remaining,tool_blocking_pairs=tool['blocking_pairs'],
        touching=tool['touching'],certificates=tool['certificates'],
        source_joint_limits_valid=True,actual_tcp_matrix=actual.ravel().tolist(),
        reason='full_source_pair_check' if not remaining else 'full_arm_or_tool_collision')


def start_solution(engine, weld, candidate, target, angle, previous, inventory):
    rid = candidate['robot_id']; model = engine.world.models[rid]
    sources = []
    if previous['selected'] and previous['selected']['robot_id']==rid:
        sources.append(np.asarray(previous['selected']['q_rad']))
    sources += [np.asarray(c['q_rad']) for c in inventory['candidates']
                if c['robot_id']==rid and c['fk_candidate_passed']]
    stats = {}; row = dict(passed=False,selected=None,full_checks=[])
    solutions = iter_solutions(model,target,sources,random_count=16,stats=stats)
    try:
        for solution in solutions:
            q = np.asarray(solution['q_rad'])
            if engine.search.full_check(row,weld,rid,q,angle,candidate['roll_deg'],
                                        candidate['axis_sign'],target,engine.delta):
                return row['selected'],dict(stats)
    finally:
        solutions.close()
    return None,stats


def lift_path(engine, weld, candidate, selected, exit_path):
    rid = candidate['robot_id']; model = engine.world.models[rid]
    q0 = np.asarray(selected['q_rad']); other = selected['other_robot_id']
    park = np.asarray(selected['other_q_rad'])
    moved, _ = engine.target(weld,candidate)
    start = model.tcp(q0); direction = resolve_exit_direction(exit_path,engine.delta)
    length = float(exit_path['length_mm'])
    frames = []; maximum_line_error = 0.; maximum_rotation_error = 0.; maximum_joint_step = 0.
    for driver in np.linspace(0.,-185.,186):
        phase = 'closed' if driver == 0. else 'opening'
        state = full_probe(engine,rid,q0,other,park,float(driver),moved,phase)
        if not state['passed']:
            return dict(passed=False,reason='full_opening_blocked',driver_mm=float(driver),failure=state)
        frames.append(dict(q_rad=q0.tolist(),driver_mm=float(driver),phase=phase,
                           tcp_xyz_mm=start[:3,3].tolist(),travel_mm=0.))
    open_state = engine.state(rid,start,-185.,moved,engine.world.static,'opening')
    context = engine.release_context(rid,start,moved,dict(open_state=open_state))
    for support in context['supports']:
        if np.asarray(support['outward_normal'])@direction <= 1e-6:
            return dict(passed=False,reason='actual_open_state_requires_other_release_direction')
    tracker = ContactTracker(context['initial_touching'])
    q = q0.copy(); travel = 0.; step = 2.
    while travel < length-1e-8:
        requested = min(length,travel+step)
        target = start.copy(); target[:3,3] += requested*direction
        fit = continuous_ik(model,target,q)
        if not fit['passed']:
            if step > .100001:
                step = max(.1,step/2.); continue
            return dict(passed=False,reason='continuous_ik_failed',travel_mm=requested,failure=fit)
        next_q = np.asarray(fit['q_rad']); probes = interpolation_probes(q,next_q)
        probe_data = []; shape_ok = True
        for qi in probes[1:]:
            actual = model.tcp(qi)
            s = float((actual[:3,3]-start[:3,3])@direction)
            lateral = line_error(actual[:3,3],start[:3,3],direction)
            rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@actual[:3,:3]).as_rotvec())))
            if lateral > .05 or rotation > .01 or s < travel-1e-5 or s > requested+.05:
                shape_ok = False; break
            probe_data.append((qi,actual,s,lateral,rotation))
        if not shape_ok:
            if step > .100001:
                step = max(.1,step/2.); continue
            return dict(passed=False,reason='actual_joint_interpolation_off_straight_line',travel_mm=requested)
        for qi,actual,s,lateral,rotation in probe_data:
            phase = 'release' if s <= 5.000001 else 'separated'
            state = full_probe(engine,rid,qi,other,park,-185.,moved,phase,context)
            if not state['passed']:
                return dict(passed=False,reason='full_retreat_blocked',travel_mm=s,failure=state)
            if not tracker.advance(state['touching'],s):
                return dict(passed=False,reason='release_contact_reappeared_or_window_exceeded',travel_mm=s)
            maximum_line_error = max(maximum_line_error,lateral)
            maximum_rotation_error = max(maximum_rotation_error,rotation)
            frames.append(dict(q_rad=qi.tolist(),driver_mm=-185.,phase=phase,
                               tcp_xyz_mm=actual[:3,3].tolist(),travel_mm=s))
        maximum_joint_step = max(maximum_joint_step,float(np.degrees(np.max(abs(next_q-q)))))
        q = next_q; travel = requested; step = min(2.,step*2.)
    clearance = engine.minimum_distance(rid,model.tcp(q),-185.,engine.world.static)
    if clearance < 5.:
        return dict(passed=False,reason='actual_endpoint_clearance_below_5mm',clearance_mm=clearance)
    return dict(passed=True,robot_id=rid,other_robot_id=other,other_q_rad=park.tolist(),
        positioner_deg=selected['positioner_deg'],candidate=candidate,
        q_start_rad=q0.tolist(),q_end_rad=q.tolist(),direction_world=direction.tolist(),
        length_mm=length,endpoint_clearance_mm=clearance,frames=frames,
        maximum_sampled_line_error_mm=maximum_line_error,
        maximum_sampled_rotation_error_deg=maximum_rotation_error,
        maximum_ik_step_joint_change_deg=maximum_joint_step,
        actual_joint_interpolation_checked=True,
        scope='Sampled source-FK joint interpolation and both complete source robots. '
              'Opening/release/arm edges have no continuous collision certificate. '
              'Conditional on existing static assembly-component exemptions; positioner rotation not planned.')


def arm_point(index, geometry, weld, previous, inventory, out):
    engine = _ENGINE; started = time.perf_counter(); trials = []; selected_path = None
    candidates = [c for c in geometry['candidates'] if c['tool_exit_passed']]
    old = previous['selected']
    def rank(c):
        candidate = c['candidate']
        if not old: return (0,0)
        roll = abs(candidate['roll_deg']-old['roll_deg']); roll = min(roll,360-roll)
        return (candidate['robot_id']!=old['robot_id'] or candidate['axis_sign']!=old['axis_sign'],roll)
    candidates.sort(key=rank)
    # The completed v3 stationary search already tested all allowed poses at
    # zero. Reuse its negative finite-search evidence instead of repeating IK.
    angles = [0.] if previous['stationary_passed'] else []
    if old and old['positioner_deg'] not in angles: angles.append(old['positioner_deg'])
    for angle in angles:
        engine.set_angle(angle)
        layout = engine.environment.layout_collisions(angle)
        if layout['blocking_pairs']:
            trials.append(dict(angle_deg=angle,reason='fixture_products_against_fixed_environment',failure=layout))
            continue
        for record in candidates:
            candidate = record['candidate']; rid = candidate['robot_id']
            moved,target = engine.target(weld,candidate)
            tool_state = engine.state(rid,target,0.,moved,engine.world.static,'closed')
            if tool_state['blocking_pairs']:
                trials.append(dict(angle_deg=angle,candidate=candidate,reason='closed_tool_fixed_environment_blocked'))
                continue
            posture,stats = start_solution(engine,weld,candidate,target,angle,previous,inventory)
            if posture is None:
                trials.append(dict(angle_deg=angle,candidate=candidate,reason='no_fully_qualified_start',ik_stats=stats))
                continue
            for exit_path in record['exits']:
                path = lift_path(engine,weld,candidate,posture,exit_path)
                if path['passed']:
                    selected_path = path; break
                trials.append(dict(angle_deg=angle,candidate=candidate,
                                   direction_local_baseline=exit_path['direction_world'],**path))
            atomic(out/f'arm_point_{index:02d}_progress.json',dict(weld_id=weld['id'],
                   trials=len(trials),angle_deg=angle,elapsed_s=time.perf_counter()-started))
            if selected_path: break
        if selected_path: break
    result = dict(weld_id=weld['id'],name=weld['name'],geometry_sha256=geometry['_sha256'],
        passed=selected_path is not None,path=selected_path,trials=trials,angles_tested=angles,
        stationary_negative_search_reused=not previous['stationary_passed'],
        elapsed_s=time.perf_counter()-started,
        scope='0 deg if a source-qualified stationary posture exists, otherwise prior per-point static angle. '
              'Finite orientations/2 retained directions; failure does not prove infeasibility.')
    atomic(out/f'arm_point_{index:02d}.json',result)
    print('ARM',index+1,weld['id'],'PASS',result['passed'],'trials',len(trials),
          'seconds',round(result['elapsed_s'],2),flush=True)
    return index,result


def main():
    global _ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--geometry-out',type=Path,default=OUT)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--point',type=int)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4: raise ValueError('Use 1 to 4 workers')
    geom_out = args.geometry_out.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    baseline=source_hashes(); scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    previous=json.loads((ROOT/'build/welding_positioner_static_v3/report.json').read_text())
    geometry_manifest=json.loads((geom_out/'manifest.json').read_text())
    if baseline != geometry_manifest['baseline_sha256']: raise RuntimeError('Geometry source/historical hashes differ')
    inventory=json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
    maps={r['weld_id']:r for r in inventory['rows']}
    indices=[args.point-1] if args.point else list(range(29)); geometries={}; rows={}; jobs=[]
    for i in indices:
        path=geom_out/f'point_{i:02d}.json'
        geometry=json.loads(path.read_text());geometry['_sha256']=digest(path);geometries[i]=geometry
        if not geometry['complete']: raise RuntimeError('Only completed point geometry can be lifted')
        saved=out/f'arm_point_{i:02d}.json'
        if args.resume and saved.exists():
            rows[i]=json.loads(saved.read_text());continue
        if not geometry['passed']:
            rows[i]=dict(weld_id=geometry['weld_id'],name=geometry['name'],passed=False,path=None,
                         status='no_tool_geometry_exit',geometry_sha256=geometry['_sha256'],trials=[])
            atomic(saved,rows[i]);continue
        jobs.append(i)
    manifest=dict(source_sha256=baseline,geometry_manifest_sha256=digest(geom_out/'manifest.json'),
        point_geometry_sha256={str(i):g['_sha256'] for i,g in geometries.items()},
        code_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'arm_path.py',HERE/'tool_world.py',HERE/'contracts.py']})
    path=out/'arm_manifest.json'
    if path.exists():
        if not args.resume: raise FileExistsError('Arm evidence exists; use --resume or a new --out')
        if json.loads(path.read_text())!=manifest: raise RuntimeError('Arm input/code changed; use a new --out')
    atomic(path,manifest)
    started=time.perf_counter();load_s=0.
    if jobs:
        load_start=time.perf_counter();_ENGINE=ToolWorld(scene);load_s=time.perf_counter()-load_start
        print('ARM_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            futures=[pool.submit(arm_point,i,geometries[i],scene['welds'][i],previous['rows'][i],
                                 maps[scene['welds'][i]['id']],out) for i in jobs]
            for future in concurrent.futures.as_completed(futures):
                i,rows[i]=future.result()
                atomic(out/'arm_progress.json',dict(completed=len(rows),total=len(indices),
                     passed=sum(r['passed'] for r in rows.values()),elapsed_s=time.perf_counter()-started))
    if source_hashes()!=baseline: raise RuntimeError('Source or historical evidence changed')
    report=dict(schema='source_tool_exit_arm_lift_v1',manifest=manifest,
        summary=dict(reported_points=len(rows),passed_points=sum(r['passed'] for r in rows.values()),
                     model_load_s=load_s,wall_s=time.perf_counter()-started,complete=len(rows)==29),
        rows=[rows[i] for i in sorted(rows)],source_and_historical_evidence_unchanged=True)
    atomic(out/'arm_report.json',report)
    print('ARM_SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
