"""Fresh relative geometry -> complete weld/open/straight-retreat actions in one search."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE)]
from run_integrated_retreat_v118 import load_raw_world
from mixed_bridge_v95 import SIGNS
from signed_wrist_model_v40 import branch_parks
from analytic_ik_v2 import analytical_seed_bank
from run_geometry import atomic,digest,initialize_worker
from joint_search_v121 import jobs_for_stage,positioner_angles,try_configuration,asset_mesh_paths
from integrated_retreat_v117 import online_retreat_ids

ENGINE=None;SETTINGS=None;OUT=None;HINTS={}


def search_point(index,row,weld,stage):
    started=time.perf_counter();row=copy.deepcopy(row);hint=HINTS.get(weld['id'])
    angles=[0] if stage=='stationary' else positioner_angles()
    if hint and stage=='positioner' and hint['positioner_deg'] in angles:
        angles=[hint['positioner_deg']]+[a for a in angles if a!=hint['positioner_deg']]
    parks={rid:[q] for rid,q in branch_parks(ENGINE.world.models,SIGNS).items()}
    for rid,model in ENGINE.world.models.items():
        for q in [np.array([-3.,-np.pi/12,0.,0.,0.,SIGNS[rid]*np.pi/2]),model.q_initial]:
            if model.limits_valid(q) and not any(np.array_equal(q,p) for p in parks[rid]):parks[rid].append(q.tolist())
    expired=False
    for angle in angles:
        if time.perf_counter()-started>SETTINGS['point_seconds']:expired=True;break
        layout=ENGINE.environment.layout_collisions(angle)
        ENGINE.set_angle(angle)
        if layout['blocking_pairs']:
            row['angle_trials'].append(dict(positioner_deg=angle,reason='layout_collision'));continue
        candidates=list(row['allowed'])
        # Prioritize rolls spread around the full cyclic domain; this is ordering only.
        centers=[0,180,90,270,45,225,135,315]
        def rank(c):
            hinted=hint and all(c[k]==hint[k] for k in ('robot_id','roll_deg','axis_sign'))
            return (not hinted,min((min(abs(c['roll_deg']-x),360-abs(c['roll_deg']-x)),i) for i,x in enumerate(centers)),c['robot_id'],c['axis_sign'])
        candidates.sort(key=rank);cache={};trial=dict(positioner_deg=angle,poses=0,ik_configurations=0,closed_passed=0,ray_trials=0);row['angle_trials'].append(trial)
        for candidate in candidates[:SETTINGS['poses']]:
            if time.perf_counter()-started>SETTINGS['point_seconds']:expired=True;break
            rid=candidate['robot_id'];moved,target=ENGINE.target(weld,candidate);model=ENGINE.world.models[rid]
            if ENGINE.search.gun_pairs(rid,target,moved,ENGINE.environment.fixed_env):continue
            trial['poses']+=1;fits,_=analytical_seed_bank(model,target,[model.q_initial])
            for fit in fits[:SETTINGS['configurations']]:
                trial['ik_configurations']+=1
                result=try_configuration(ENGINE,weld,candidate,fit['q_rad'],parks,{**SETTINGS,'angle_deg':angle},cache)
                trial['closed_passed']+=int(result['closed_passed']);trial['ray_trials']+=len(result['attempts'])
                log=OUT/f'search_{index:02d}.jsonl'
                with log.open('a') as stream:
                    record={k:v for k,v in result.items() if k not in ('path','selected')}
                    record.update(stage=stage,positioner_deg=angle);stream.write(json.dumps(record,separators=(',',':'))+'\n')
                if result['passed']:
                    path=result['path'];selected=result['selected'];name=f'paths/retreat_{index:02d}.json';atomic(OUT/name,path);h=digest(OUT/name)
                    row.update(passed=True,initial=selected,best=dict(path_file=name,path_sha256=h,length_mm=1000.,driver_mm=path['driver_mm'],amplitude_mm=abs(path['driver_mm'])),
                        online_acceptance=dict(passed=True,path_sha256=h,validation=path['online_validation']),stationary_passed=angle==0)
                    break
            if row['passed']:break
        atomic(OUT/f'point_{index:02d}_progress.json',dict(weld_id=weld['id'],stage=stage,angle=angle,passed=row['passed'],elapsed_s=time.perf_counter()-started,trial=trial))
        print('JOINT_ANGLE',weld['id'],stage,angle,'PASS',row['passed'],'CLOSED',trial['closed_passed'],'RAYS',trial['ray_trials'],'SECONDS',round(time.perf_counter()-started,2),flush=True)
        if row['passed'] or expired:break
    row[stage+'_done']=True;row.setdefault('stage_worker_s',{})[stage]=time.perf_counter()-started
    row['status']='complete_action_accepted' if row['passed'] else ('finite_budget_expired' if expired else 'complete_action_not_found_in_finite_budget')
    if not row['passed']:row.setdefault('initial',None);row.setdefault('best',None)
    atomic(OUT/f'point_{index:02d}_{stage}.json',row)
    print('JOINT_POINT',weld['id'],stage,row['passed'],row['status'],round(time.perf_counter()-started,3),flush=True)
    return index,row


def main():
    global ENGINE,SETTINGS,OUT,HINTS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--directions',type=int,default=32)
    p.add_argument('--poses',type=int,default=48);p.add_argument('--configurations',type=int,default=2)
    p.add_argument('--point-seconds',type=float,default=240.)
    p.add_argument('--pose-hints',type=Path,help='Optional previously selected robot/roll/angle only; no q/ray/open/path reuse')
    args=p.parse_args()
    if not 1<=args.workers<=4 or not 1<=args.directions<=2522 or args.poses<1 or args.configurations<1 or args.point_seconds<=0:raise ValueError('Invalid finite budgets')
    OUT=args.out.resolve()
    if OUT.exists():raise FileExistsError('Use fresh directory')
    begin=time.perf_counter();OUT.mkdir(parents=True);initialize_worker()
    SETTINGS=dict(directions=args.directions,poses=args.poses,configurations=args.configurations,point_seconds=args.point_seconds,amplitudes=[10.,30.,60.,80.,120.,140.,180.])
    if args.pose_hints:
        hints=json.loads(args.pose_hints.read_text())
        HINTS={r['weld_id']:{k:r['initial'][k] for k in ('robot_id','roll_deg','axis_sign','positioner_deg')} for r in hints['rows'] if r.get('best')}
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());ENGINE=load_raw_world(scene,SIGNS);load_s=time.perf_counter()-begin
    deps=[Path(__file__),HERE/'joint_search_v121.py',HERE/'run_integrated_retreat_v118.py',HERE/'integrated_retreat_v117.py',HERE/'analytic_ik_v2.py',HERE/'analytic_trace_motion_v2.py',HERE/'probe_near_retreat_v72.py',HERE/'probe_neighbor_rays_v17.py',HERE/'signed_wrist_model_v40.py',HERE/'mixed_bridge_v95.py',
          ROOT/'experimental/welding_tool_exit_v1/verify_initial.py',ROOT/'experimental/welding_tool_exit_v1/tool_world.py',ROOT/'experimental/welding_tool_exit_v1/run_arm.py',ROOT/'experimental/welding_tool_exit_v1/arm_path.py',ROOT/'experimental/welding_tool_exit_v1/contracts.py',
          ROOT/'experimental/welding_positioner_v2/search.py',ROOT/'experimental/welding_positioner_v2/world.py',ROOT/'experimental/welding_positioner_v2/geometry.py',ROOT/'experimental/welding_positioner_v2/schedule.py',
          ROOT/'experimental/welding_positioner_v1/positioner.py',ROOT/'experimental/welding_variable_open_v1/variable_world.py',ROOT/'experimental/welding_variable_open_v1/variable_core.py',ROOT/'experimental/welding_direct_rrt_v1/run_direct.py',ROOT/'scripts/audit_contact_release_v17.py',ROOT/'welding_kinematics.py',ROOT/'gun_kinematics.py',ROOT/'collision_audit.py',ROOT/'build_scene.py',ROOT/'r2_safe_park.py']
    codes={str(f.relative_to(ROOT)):digest(f) for f in deps}
    sources=dict(ENGINE.environment.audit['source_sha256']);sources.update(ENGINE.environment.audit['mesh_sha256'])
    source_files=[ROOT/'build/scene_placement_corrected_v1.json',ROOT/'build/robot.json',ROOT/'build/fixtureL.json',ROOT/'build/welding_endpoints_v1_corrected_static.json',ROOT/'build/welding_positioner_static_v2/geometry_audit.json',ROOT/'source/psz/EngData/engData_ver2.xml']
    source_files.extend(asset_mesh_paths(scene,ROOT))
    if args.pose_hints:source_files.append(args.pose_hints.resolve())
    sources.update({str(f.relative_to(ROOT)):digest(f) for f in source_files})
    for n,h in {**sources,**codes}.items():
        if digest(ROOT/n)!=h:raise ValueError('Source/code mismatch '+n)
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(**SETTINGS,workers=args.workers,per_robot_wrist_sign=SIGNS,
        distance_mm=1000.,posture_roll_step_deg=5,positioner_grid_step_deg=5,positioner_angle_order=positioner_angles(),
        acceptance='Closed pose + full opening prefix + complete 1000mm online straight retreat',stationary_barrier='Complete actions only',
        prior_pose_hints=bool(HINTS),prior_q_ray_opening_or_path_reused=False,raw_fixture_geometry=True,independent_cold_required=False))
    atomic(OUT/'manifest.json',manifest)
    screen_begin=time.perf_counter();screen=[ENGINE.search.relative_screen(w) for w in scene['welds']];screen_s=time.perf_counter()-screen_begin
    atomic(OUT/'geometry_screen.json',dict(rows=screen,summary=dict(source_points=len(screen),qualified=sum(bool(r['allowed']) for r in screen),discarded=sum(not r['allowed'] for r in screen),wall_s=screen_s)))
    rows=[r for r in screen if r['allowed']];welds={w['id']:w for w in scene['welds']};timings={}
    print('JOINT_GEOMETRY',len(screen),len(rows),'LOAD',load_s,'SCREEN',screen_s,flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        for stage in ('stationary','positioner'):
            started=time.perf_counter();jobs=[pool.submit(search_point,i,rows[i],welds[rows[i]['weld_id']],stage) for i in jobs_for_stage(rows,stage)]
            for f in concurrent.futures.as_completed(jobs):
                i,row=f.result();rows[i]=row
                atomic(OUT/'progress.json',dict(stage=stage,finished=sum(bool(r.get(stage+'_done')) for r in rows),qualified=len(rows),passed=sum(r['passed'] for r in rows),elapsed_s=time.perf_counter()-begin))
            timings[stage+'_wall_s']=time.perf_counter()-started
    for i,row in enumerate(rows):
        row['required_straight_distance_mm']=1000.;row['fresh_cold_required']=False;row['online_validation_required']=True
        atomic(OUT/f'point_{i:02d}.json',row)
    for n,h in {**sources,**codes}.items():
        if digest(ROOT/n)!=h:raise ValueError('Input changed during joint search '+n)
    outputs={str(f.relative_to(OUT)):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('manifest.json','progress.json','report.json') and not f.name.endswith('_progress.json')}
    summary=dict(complete=True,source_points=29,geometry_qualified=len(rows),geometry_discarded=len(screen)-len(rows),points=len(rows),requested_points=len(rows),passed=sum(r['passed'] for r in rows),required_straight_distance_mm=1000.,
        online_validated_retreats=sum(r['passed'] for r in rows),model_load_s=load_s,geometry_screen_s=screen_s,**timings,wall_s=time.perf_counter()-begin,
        independent_cold_performed=False,full_inventory_action_planned=all(r['passed'] for r in rows))
    report=dict(manifest=manifest,summary=summary,rows=rows,output_sha256=outputs,scope='Unified finite pose/open/ray/IK search: close-only candidates never accepted. No historical q/ray/open/path reuse. Original geometry with existing assembly exemptions; sampled online acceptance, not continuous/dynamic or independent cold proof.')
    online_retreat_ids(report,OUT);atomic(OUT/'report.json',report);print('JOINT_SUMMARY',summary,flush=True)
    if not summary['full_inventory_action_planned']:raise SystemExit(2)


if __name__=='__main__':main()
