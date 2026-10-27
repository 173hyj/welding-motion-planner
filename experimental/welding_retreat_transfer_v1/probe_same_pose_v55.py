"""Try source-legal IK branches of retained exact poses before new pose sweeps."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v40 as search
from signed_wrist_model_v40 import SignedWristModel,branch_parks,path_compatibility
from analytic_ik_v2 import solver_for
from probe_neighbor_rays_v17 import kinematic_rejection
from run_geometry import atomic,digest,initialize_worker

ROOT=search.ROOT
HERE=Path(__file__).resolve().parent


def same_source_pose(model,reference,q):
    reference,q=np.asarray(reference),np.asarray(q)
    return bool(reference.shape==(6,) and q.shape==(6,) and np.isfinite(q).all() and
        np.isfinite(reference).all() and model.limits_valid(q) and
        np.allclose(model.tcp(reference),model.tcp(q),atol=1e-6,rtol=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();baseline,original,out=(p.resolve() for p in (args.baseline,args.original,args.out))
    if out.exists():raise FileExistsError('Fresh output required')
    previous=json.loads((baseline/'report.json').read_text());old=json.loads((original/'report.json').read_text())
    if (len(previous['rows'])!=21 or {r['weld_id'] for r in previous['rows']}!={r['weld_id'] for r in old['rows']} or
        previous['summary']['required_straight_distance_mm']!=2000. or old['summary']['required_straight_distance_mm']!=2000.):
        raise ValueError('Same original all21 2000mm inventory required')
    sources={};rows=copy.deepcopy(previous['rows']);originals={r['weld_id']:r for r in old['rows']}
    for directory,report in ((baseline,previous),(original,old)):
        sources.update(report['manifest']['source_sha256']);sources.update(report['manifest']['code_sha256'])
        sources[str((directory/'report.json').relative_to(ROOT))]=digest(directory/'report.json')
        for n,s in report['output_sha256'].items():sources[str((directory/n).relative_to(ROOT))]=s
    codes={str((HERE/n).relative_to(ROOT)):digest(HERE/n) for n in
        (Path(__file__).name,'branch_point_job_v40.py','signed_wrist_model_v40.py',
         'analytic_ik_v2.py','analytic_trace_motion_v2.py','probe_neighbor_rays_v17.py')}
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s:raise ValueError('Changed input '+n)
    out.mkdir(parents=True)
    for n,s in previous['output_sha256'].items():
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((baseline/n).read_bytes())
    initialize_worker();search.retreat.assert_frozen_geometry_cache()
    engine=search.retreat.VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    models=engine.world.models;ids=sorted(models);signs={ids[0]:-1,ids[1]:1};parks=branch_parks(models,signs)
    for rid,m in list(models.items()):models[rid]=SignedWristModel(m,signs[rid])
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(distance_mm=2000.,
        per_robot_wrist_sign=signs,per_robot_inactive_park=parks,exact_pose_preserved=True,
        no_midpath_wrap=True,no_negative_reuse=True,fresh_cold_required=True))
    atomic(out/'manifest.json',manifest);started=time.perf_counter();records=[]
    for i,row in enumerate(rows):
        wid=row['weld_id'];source=originals[wid]
        if row['best'] or not source['best']:continue
        path=json.loads((original/source['best']['path_file']).read_text());rid=path['robot_id'];model=models[rid]
        q0=np.asarray(path['q_start_rad']);pose=model.tcp(q0);solutions=solver_for(model).solve(pose,q0)
        record=dict(weld_id=wid,original_start_solutions=len(solutions),trials=[]);records.append(record)
        print('SAME_POSE_START',wid,len(solutions),flush=True)
        weld=next(w for w in engine.scene['welds'] if w['id']==wid)
        for q in solutions:
            if not same_source_pose(model,q0,q):raise ValueError('Candidate changes exact pose or source limits')
            kin=kinematic_rejection(model,pose,q,np.asarray(path['direction_world']))
            trial=dict(q_rad=q.tolist(),kinematic_prefilter=kin);record['trials'].append(trial)
            if not kin['passed']:continue
            selected=copy.deepcopy(source['initial']);other=path['other_robot_id']
            selected.update(q_rad=q.tolist(),other_q_rad=parks[other],actual_tcp_matrix=model.tcp(q).ravel().tolist())
            engine.set_angle(selected['positioner_deg']);moved,_=engine.target(weld,selected)
            caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
            profile=search.opening_profile(lambda d:search.full_probe(engine,rid,q,other,np.asarray(parks[other]),d,moved,
                'closed' if d==0. else 'opening'),caps,maximum_amplitude_mm=80.)
            driver=path['driver_mm'];trial['opening_profile']=profile
            if not search.prefix_available(profile,-driver):continue
            geom=search.retreat.geometry_prefix(engine,rid,pose,moved,driver,np.asarray(path['direction_world']),2000.)
            trial['geometry_passed']=geom['passed']
            if not geom['passed']:continue
            warm=search.lift_path(engine,weld,selected,selected,dict(direction_world=path['direction_world'],
                direction_frame='current_world',length_mm=2000.),driver_mm=driver,
                opening_frames=search.opening_frames(dict(initial=selected,opening_profile=profile),pose,driver))
            trial.update(passed=warm['passed'],reason=warm.get('reason'),last_accepted_travel_mm=warm.get('last_accepted_travel_mm'))
            if not warm['passed']:continue
            warm.update(weld_id=wid,fixed_distance_mm=2000.,no_qp=True,geometry_certificate=geom,same_source_pose_reselected=True)
            if path_compatibility(warm,signs,parks)!=(True,True):raise ValueError('Branch/park changed in saved trace')
            name=f'paths/same_pose_{i:02d}.json';atomic(out/name,warm)
            row['original_initial']=copy.deepcopy(source.get('original_initial',source['initial']))
            row.update(initial=selected,best=dict(source['best'],path_file=name,path_sha256=digest(out/name)),
                opening_profile=profile,mixed_wrist_branch_compatible=True,fresh_cold_required=True,
                selection_status='same_pose_candidate_not_yet_cold_verified')
            print('SAME_POSE_PASSED',wid,flush=True);break
        atomic(out/(wid+'_attempt.json'),record)
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s:raise ValueError('Changed source while searching '+n)
    outputs={str(p.relative_to(out)):digest(p) for p in [*(out/'paths').glob('*.json'),*out.glob('*_attempt.json')]}
    summary=dict(complete=True,points=21,passed=sum(bool(r['best']) for r in rows),required_straight_distance_mm=2000.,
        branch_compatible=sum(bool(r['best']) for r in rows),historical_chain_intact=False,fresh_cold_required=True,
        retained_existing_candidates=14,wall_s=time.perf_counter()-started)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,search_records=records,output_sha256=outputs,
        scope='Same original TCP poses and directions; source signed IK, complete opening and 2000mm trace. '
              'Original all21 identities and14 path candidates retained; finite negatives not impossibility. '
              'New cold and true endpoint transfers required. Historical dependency loss remains disclosed.'))
    print('SAME_POSE_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
