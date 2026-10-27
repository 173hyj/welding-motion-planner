"""Directly test archived initial configurations; never reselect welding IK."""
import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
from tool_world import ToolWorld, ROOT, HERE
from run_arm import full_probe, lift_path
from run_geometry import source_hashes, atomic, digest, initialize_worker

OUT = ROOT/'build/welding_tool_initial_exit_v2'
_ENGINE = None


def extended_source_hashes():
    paths = set(source_hashes())
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    adapted = json.loads((ROOT/'build/welding_positioner_static_v2/geometry_audit.json').read_text())['scene']
    for asset in scene['assets']+adapted['assets']:
        if asset.get('url'): paths.add('build/'+asset['url'])
        for link in asset.get('robot',{}).get('link_models',{}).values():
            paths.add('build/'+link['url'])
    paths.update(['build/robot.json','build/fixtureL.json','build/AP31-M7-S1-00-00.json',
        'build_scene.py','collision_audit.py','welding_kinematics.py','gun_kinematics.py',
        'experimental/welding_positioner_v1/positioner.py',
        'experimental/welding_positioner_v2/world.py','experimental/welding_positioner_v2/search.py',
        'experimental/welding_endpoints_v1/refinement.py',
        'r2_safe_park.py',
        'build/eight_hour_v17/experimental_release/multi_point_map/source_face_radius_v1.py',
        'build/eight_hour_v17/experimental_release/multi_point_map/source_face_world_adapter_v1.py'])
    return {name:digest(ROOT/name) for name in sorted(paths)}


def assert_frozen_geometry_cache():
    """Prevent old prepare_scene from entering its historical rewrite branch."""
    path=ROOT/'build/welding_positioner_static_v2/geometry_audit.json'
    audit=json.loads(path.read_text())
    for section in ('source_sha256','mesh_sha256'):
        for name,expected in audit[section].items():
            if digest(ROOT/name)!=expected:
                raise RuntimeError('Frozen v2 geometry cache differs; refusing historical rebuild: '+name)
    return True


def initial_job(index, weld, selected, out):
    engine = _ENGINE; started = time.perf_counter()
    rid = selected['robot_id']; q0=np.asarray(selected['q_rad']); other=selected['other_robot_id']
    park=np.asarray(selected['other_q_rad']); engine.set_angle(selected['positioner_deg'])
    moved,_=engine.target(weld,selected); actual=engine.world.models[rid].tcp(q0)
    candidate={k:selected[k] for k in ('robot_id','roll_deg','axis_sign')}
    layout=engine.environment.layout_collisions(selected['positioner_deg'])
    result=dict(weld_id=weld['id'],name=weld['name'],initial=dict(robot_id=rid,q_rad=q0.tolist(),
        other_robot_id=other,other_q_rad=park.tolist(),roll_deg=selected['roll_deg'],
        axis_sign=selected['axis_sign'],positioner_deg=selected['positioner_deg']),
        initial_q_reoptimized=False,opening_passed=False,retreat_passed=False,
        opening_checked_samples=0,exit_trials=[],path=None)
    if layout['blocking_pairs']:
        result.update(reason='source_layout_blocked',failure=layout)
    else:
        opening_failure=None; initial_certified=True
        for driver in np.linspace(0.,-185.,186):
            phase='closed' if driver==0 else 'opening'
            state=full_probe(engine,rid,q0,other,park,float(driver),moved,phase)
            result['opening_checked_samples']+=1
            if driver==0:
                caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
                initial_certified=all(state.get('certificates',{}).get(pid,{}).get('passed',False) for pid in caps)
            if not state['passed'] or not initial_certified:
                opening_failure=dict(driver_mm=float(driver),state=state);break
        if opening_failure:
            result.update(reason='direct_initial_opening_blocked',failure=opening_failure)
        else:
            result['opening_passed']=True
            open_state=engine.state(rid,actual,-185.,moved,engine.environment.relative_env,'opening')
            exits=engine.search_exits(rid,actual,moved,dict(open_state=open_state),count=2)
            result['tool_exit_count']=len(exits['exits']);result['rays_tested']=exits['rays_tested']
            result['ray_failure_counts']=exits['failure_counts']
            for exit_path in exits['exits']:
                # These rays were just searched at the already-set external
                # angle. Geometry-report rays, in contrast, originate at 0°.
                exit_path=dict(exit_path,direction_frame='current_world')
                path=lift_path(engine,weld,candidate,selected,exit_path)
                if path['passed']:
                    if path['q_start_rad'] != selected['q_rad'] or path['frames'][0]['q_rad'] != selected['q_rad']:
                        raise RuntimeError('Archived initial configuration changed')
                    if np.linalg.norm(np.asarray(path['direction_world'])-exit_path['direction_world'])>1e-10:
                        raise RuntimeError('World exit ray was transformed more than once')
                    result.update(retreat_passed=True,path=path,reason='direct_initial_open_and_retreat_passed')
                    break
                result['exit_trials'].append(path)
            if not result['retreat_passed']:
                result['reason']='no_tool_exit_found' if not exits['exits'] else 'continuous_arm_retreat_failed'
    result['elapsed_s']=time.perf_counter()-started
    atomic(out/f'initial_point_{index:02d}.json',result)
    print('INITIAL',index+1,weld['id'],'OPEN',result['opening_passed'],'RETREAT',result['retreat_passed'],
          result['reason'],'seconds',round(result['elapsed_s'],2),flush=True)
    return index,result


def main():
    global _ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Use 1 to 4 isolated workers')
    assert_frozen_geometry_cache()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    baseline=extended_source_hashes()
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    previous=json.loads((ROOT/'build/welding_positioner_static_v3/report.json').read_text())
    manifest=dict(source_sha256=baseline,initial_report_sha256=digest(ROOT/'build/welding_positioner_static_v3/report.json'),
        code_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'run_arm.py',HERE/'arm_path.py',HERE/'tool_world.py',HERE/'contracts.py']},
        settings=dict(initial_q_locked=True,initial_robot_roll_positioner_and_park_locked=True,
                      opening_driver_step_mm=1.,minimum_retreat_mm=100.,retained_directions=2))
    path=out/'initial_manifest.json'
    if path.exists():
        if not args.resume:raise FileExistsError('Use --resume or another --out to preserve evidence')
        if json.loads(path.read_text())!=manifest:raise RuntimeError('Input/code changed; use another --out')
    atomic(path,manifest)
    jobs=[];rows={}
    for i,row in enumerate(previous['rows']):
        if not row['passed']:continue
        saved=out/f'initial_point_{i:02d}.json'
        if args.resume and saved.exists():rows[i]=json.loads(saved.read_text())
        else:jobs.append(i)
    started=time.perf_counter();load_s=0.
    if jobs:
        begin=time.perf_counter();_ENGINE=ToolWorld(scene);load_s=time.perf_counter()-begin
        print('INITIAL_MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            futures=[pool.submit(initial_job,i,scene['welds'][i],previous['rows'][i]['selected'],out) for i in jobs]
            for future in concurrent.futures.as_completed(futures):
                i,rows[i]=future.result()
                atomic(out/'initial_progress.json',dict(completed=len(rows),total=21,
                    opening_passed=sum(r['opening_passed'] for r in rows.values()),
                    retreat_passed=sum(r['retreat_passed'] for r in rows.values())))
    if extended_source_hashes()!=baseline:raise RuntimeError('Actual source meshes/dependencies or historical evidence changed')
    report=dict(schema='archived_initial_direct_opening_and_straight_retreat_v2',manifest=manifest,
        summary=dict(initial_configurations=21,completed=len(rows),
            opening_passed=sum(r['opening_passed'] for r in rows.values()),
            open_and_retreat_passed=sum(r['retreat_passed'] for r in rows.values()),
            stationary_initials=sum(r['initial']['positioner_deg']==0 for r in rows.values()),
            prior_positioner_initials=sum(r['initial']['positioner_deg']!=0 for r in rows.values()),
            model_load_s=load_s,wall_s=time.perf_counter()-started,complete=len(rows)==21),
        rows=[rows[i] for i in sorted(rows)],source_and_historical_evidence_unchanged=True,
        scope='Exactly the 21 v3 selected q/robot/roll/positioner/other-arm parks; no welding posture IK reselection. '
              'Retreat uses incremental local IK and two tool-geometric straight directions, at least100 mm. '
              'Full source pairs checked at sampled opening and actual q-interpolation probes, not continuous robot certification. '
              'Existing assembly-component exemptions retained; no positioner rotation path.')
    atomic(out/'initial_report.json',report)
    print('INITIAL_SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
