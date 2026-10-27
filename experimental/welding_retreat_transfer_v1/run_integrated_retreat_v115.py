"""Regenerate selected 1000 mm seed rays; integrate final gates or matched legacy mode."""
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'welding_variable_open_v1')]
import branch_point_job_v64 as search
import run_retreat_100cm_endpoint_v101 as fallback
import analytic_trace_motion_v2 as legacy
import integrated_retreat_v114 as online
import verify_initial
from variable_world import VariableToolWorld
from variable_core import driver_samples
from signed_wrist_model_v40 import SignedWristModel
from run_geometry import atomic,digest,initialize_worker

ENGINE=None
MODE=None


def load_raw_world(scene,signs):
    engine=verify_initial.fresh_source_world(scene);engine.__class__=VariableToolWorld
    for rid,model in list(engine.world.models.items()):engine.world.models[rid]=SignedWristModel(model,signs[rid])
    spec=importlib.util.spec_from_file_location('online_independent_xml_fk',ROOT/'scripts/audit_contact_release_v17.py')
    raw_fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw_fk)
    raw=json.loads((ROOT/'build/robot.json').read_text());joints=raw_fk.joints_from_raw(raw)[:6]
    tcps={a['id']:raw_fk.source_tcp_home(raw,a) for a in scene['assets'] if a['kind']=='robot'}
    engine.online_tcp=lambda rid,q:engine.world.models[rid].base@raw_fk.flange_source(q,joints)@tcps[rid]
    return engine


def generate_point(index,row,seed_out,out,settings,directions):
    begun=time.perf_counter();selected=dict(row['initial']);rid=selected['robot_id'];other=selected['other_robot_id']
    seed=json.loads((seed_out/row['best']['path_file']).read_text())
    ENGINE.set_angle(selected['positioner_deg']);weld=next(w for w in ENGINE.scene['welds'] if w['id']==row['weld_id'])
    moved,_=ENGINE.target(weld,selected);start=ENGINE.world.models[rid].tcp(np.asarray(selected['q_rad']))
    driver=seed['driver_mm'];ray=dict(direction_world=seed['direction_world'],direction_frame='current_world',length_mm=1000.)
    geometry=search.retreat.geometry_prefix(ENGINE,rid,start,moved,driver,np.asarray(ray['direction_world']),1000.)
    path=None;profile=None;attempts=[]
    if geometry['passed']:
        if MODE=='online':path=online.lift_path(ENGINE,weld,selected,selected,ray,driver_mm=driver)
        else:
            caps=[pid for pid,(_,kind,_) in ENGINE.parts[rid].items() if kind=='electrode']
            profile=search.opening_profile(lambda d:search.full_probe(ENGINE,rid,np.asarray(selected['q_rad']),other,
                np.asarray(selected['other_q_rad']),d,moved,'closed' if d==0. else 'opening'),caps,maximum_amplitude_mm=abs(driver))
            if search.prefix_available(profile,abs(driver)):
                path=legacy.lift_path(ENGINE,weld,selected,selected,ray,driver_mm=driver,
                    opening_frames=search.opening_frames(dict(initial=selected,opening_profile=profile),start,driver))
        attempts.append(dict(strategy='regenerate_selected_pose_ray',passed=bool(path and path['passed']),reason=path.get('reason') if path else None))
    if not path or not path['passed']:
        if MODE!='online':raise RuntimeError('Matched legacy seed failed: '+row['weld_id'])
        fallback.ENGINE=ENGINE;search.lift_path=online.lift_path
        path,changed,profile,trials=fallback.search_pending(row,settings['seeds'],settings['parks'],directions)
        attempts.append(dict(strategy='finite_fallback_search',trials=trials))
        if path:
            selected=changed;geometry=path['geometry_certificate']
    result=dict(weld_id=row['weld_id'],initial=selected,original_initial=row.get('original_initial',row['initial']),
        passed=bool(path and path['passed']),best=None,opening_profile=profile,required_straight_distance_mm=1000.,
        fresh_cold_required=MODE!='online',online_validation_required=MODE=='online',per_path_inactive_park_authoritative=True)
    if result['passed']:
        path.update(weld_id=row['weld_id'],fixed_distance_mm=1000.,geometry_certificate=geometry)
        if MODE=='online':online.check_path_contract(path,selected,1000.)
        name=f'paths/retreat_{index:02d}.json';atomic(out/name,path);h=digest(out/name)
        result['best']=dict(path_file=name,path_sha256=h,length_mm=1000.,driver_mm=path['driver_mm'],amplitude_mm=abs(path['driver_mm']))
        if MODE=='online':result['online_acceptance']=dict(passed=True,path_sha256=h,validation=path['online_validation'])
    result['elapsed_s']=time.perf_counter()-begun
    atomic(out/f'point_{index:02d}.json',result);atomic(out/f'trials_{index:02d}.json',attempts)
    print('INTEGRATED_POINT',MODE,row['weld_id'],result['passed'],round(result['elapsed_s'],3),flush=True)
    return index,result


def main():
    global ENGINE,MODE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-retreat',type=Path,default=ROOT/'build/welding_retreat_100cm_diverse_v102')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['online','legacy'],default='online')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--directions',type=int,default=256)
    args=p.parse_args();MODE=args.mode
    if not 1<=args.workers<=4 or not 1<=args.directions<=2522:raise ValueError('Invalid worker/direction budget')
    out=args.out.resolve();seed_out=args.seed_retreat.resolve()
    if out.exists():raise FileExistsError('Use fresh output directory')
    total_started=time.perf_counter();parent=json.loads((seed_out/'report.json').read_text())
    if not parent['summary']['complete'] or len(parent['rows'])!=21 or not all(r['best'] for r in parent['rows']):raise ValueError('21 selected seed rays required')
    sources={**parent['manifest']['source_sha256'],**parent['manifest']['code_sha256'],str((seed_out/'report.json').relative_to(ROOT)):digest(seed_out/'report.json')}
    for name,h in parent['output_sha256'].items():
        if digest(seed_out/name)!=h:raise ValueError('Seed output changed '+name)
        sources[str((seed_out/name).relative_to(ROOT))]=h
    code_paths=[Path(__file__),HERE/'integrated_retreat_v114.py',HERE/'analytic_trace_motion_v2.py',HERE/'analytic_ik_v2.py',HERE/'run_retreat_100cm_endpoint_v101.py',
        ROOT/'experimental/welding_tool_exit_v1/verify_initial.py',ROOT/'experimental/welding_tool_exit_v1/run_arm.py',ROOT/'experimental/welding_tool_exit_v1/arm_path.py',
        ROOT/'experimental/welding_variable_open_v1/variable_core.py',ROOT/'scripts/audit_contact_release_v17.py']
    codes={str(p.relative_to(ROOT)):digest(p) for p in code_paths}
    for n,h in {**sources,**codes}.items():
        if digest(ROOT/n)!=h:raise ValueError('Source/code changed '+n)
    setup_s=time.perf_counter()-total_started;out.mkdir(parents=True)
    signs=parent['manifest']['settings']['per_robot_wrist_sign']
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(mode=MODE,workers=args.workers,distance_mm=1000.,
        seed_reuse='Only selected welding poses, opening amplitudes and ray directions; no saved path/cold result reuse',
        generation='Fresh analytic continuation',source_geometry='Original raw fixture triangles',per_robot_wrist_sign=signs,
        opening_step_mm=.5 if MODE=='online' else 1.,saved_edge_midpoints=MODE=='online',independent_xml_fk=MODE=='online',
        separate_retreat_cold_required=MODE!='online',direction_budget=args.directions))
    atomic(out/'manifest.json',manifest);initialize_worker();search.retreat.assert_frozen_geometry_cache()
    begun=time.perf_counter();scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    ENGINE=load_raw_world(scene,signs);load_s=time.perf_counter()-begun
    fallback.ENGINE=ENGINE;search.ENGINE=ENGINE
    if MODE=='online':search.lift_path=online.lift_path
    rows=[None]*21;trace_started=time.perf_counter()
    seeds=parent['manifest']['settings'].get('seeds',[])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        jobs=[pool.submit(generate_point,i,row,seed_out,out,dict(seeds=seeds if isinstance(seeds,list) else seeds.get(row['weld_id'],[]),parks=parent['manifest']['settings']['per_robot_inactive_park']),args.directions) for i,row in enumerate(parent['rows'])]
        for f in concurrent.futures.as_completed(jobs):
            i,row=f.result();rows[i]=row;atomic(out/'progress.json',dict(finished=sum(r is not None for r in rows),passed=sum(bool(r and r['best']) for r in rows),elapsed_s=time.perf_counter()-total_started))
    trace_s=time.perf_counter()-trace_started
    for n,h in {**sources,**codes}.items():
        if digest(ROOT/n)!=h:raise ValueError('Inputs changed during generation '+n)
    outputs={str(p.relative_to(out)):digest(p) for p in out.rglob('*.json') if p.name not in ('report.json','manifest.json','progress.json')}
    summary=dict(complete=True,points=21,requested_points=21,passed=sum(bool(r['best']) for r in rows),required_straight_distance_mm=1000.,
        online_validated_retreats=sum(bool(r.get('online_acceptance')) for r in rows),independent_cold_performed=False,
        setup_s=setup_s,model_load_s=load_s,trace_wall_s=trace_s,wall_s=time.perf_counter()-total_started)
    report=dict(manifest=manifest,summary=summary,rows=rows,output_sha256=outputs,
        scope='Selected pose/ray seed regeneration benchmark, not from-zero geometry/pose search. Online source acceptance is not independent cold. Sampled geometry with existing source assembly exemptions.')
    if MODE=='online':online.online_retreat_ids(report,out)
    atomic(out/'report.json',report);print('INTEGRATED_SUMMARY',summary,flush=True)
    if summary['passed']!=21:raise SystemExit(2)


if __name__=='__main__':main()
