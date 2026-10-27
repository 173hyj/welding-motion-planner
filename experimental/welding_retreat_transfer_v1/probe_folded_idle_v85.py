"""Recheck actual other-arm blocked GW7 rays with source-legal folded parks."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from analytic_ik_v2 import analytical_continuous_ik
from compact_inventory_v82 import compact_rows, link_candidate
from folded_idle_park_v84 import park_bank, blocked_rays
from signed_wrist_model_v40 import SignedWristModel, path_compatibility
from probe_mixed_wrist_v65 import replace_with_positive
from run_geometry import atomic, digest, initialize_worker

ROOT=search.ROOT
HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists(): raise FileExistsError('Fresh evidence required')
    baseline=ROOT/'build/welding_retreat_200cm_deferred_sphere_v83'
    source=baseline/'report.json'
    parent=json.loads(source.read_text())
    if (not parent['summary']['complete'] or len(parent['rows'])!=21 or
            len({r['weld_id'] for r in parent['rows']})!=21 or
            parent['summary']['required_straight_distance_mm']!=2000.):
        raise ValueError('Completed full21 literal200cm inventory required')
    wid='OP10L-SUB1-POINT-GW-7'
    evidence_path=ROOT/'build/welding_retreat_200cm_deferred_sphere_v81'/ (wid+'_sphere.json')
    evidence=json.loads(evidence_path.read_text())
    if evidence['weld_id']!=wid: raise ValueError('Wrong evidence identity')
    rays=blocked_rays(evidence,3)
    if not rays: raise ValueError('Real other-arm failures required')
    rows=compact_rows(parent['rows'],source,ROOT)
    original=next(r for r in rows if r['weld_id']==wid)
    if original['best']: raise ValueError('Do not replace an existing positive')
    sources={**parent['manifest']['source_sha256'],**parent['manifest']['code_sha256'],
        str(source.relative_to(ROOT)):digest(source),str(evidence_path.relative_to(ROOT)):digest(evidence_path)}
    sources.update({str((baseline/n).relative_to(ROOT)):s for n,s in parent['output_sha256'].items()})
    codes={str((HERE/n).relative_to(ROOT)):digest(HERE/n) for n in
        [Path(__file__).name,'folded_idle_park_v84.py','compact_inventory_v82.py',
         'branch_point_job_v64.py','analytic_ik_v2.py','analytic_trace_motion_v2.py',
         'signed_wrist_model_v40.py','probe_mixed_wrist_v65.py']}
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s: raise ValueError('Input changed '+n)
    out.mkdir(parents=True)
    for i,row in enumerate(rows):
        if row.get('retained_existing_candidate'):
            row['retained_existing_candidate']=link_candidate(row['weld_id'],row['retained_existing_candidate'],baseline,out,f'paths/retained_{i:02d}.json')
        if row['best']:
            row['best']=link_candidate(row['weld_id'],row['best'],baseline,out,f'paths/reused_{i:02d}.json')
    signs=parent['manifest']['settings']['per_robot_wrist_sign']
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(
        distance_mm=2000.,per_robot_wrist_sign=signs,ray_budget=3,
        inactive_first_axis_rad=[-3.,-2.,-1.,0.,1.,2.,3.],
        inactive_second_deg=[-45.,-15.,15.,45.,75.],inactive_third_deg=[-90.,-45.,0.,45.,75.],
        per_robot_inactive_park=parent['manifest']['settings']['per_robot_inactive_park'],
        per_path_inactive_park_authoritative=True,original_source_gates_unchanged=True,
        sparse_checks_reject_only=True,inactive_arm_locked_through_retreat=True,
        no_midpath_wrap=True,fresh_cold_and_actual15axis_required=True,rays=rays))
    atomic(out/'manifest.json',manifest)
    initialize_worker();started=time.perf_counter()
    search.retreat.assert_frozen_geometry_cache()
    engine=search.retreat.VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    models=engine.world.models
    for rid,model in list(models.items()): models[rid]=SignedWristModel(model,signs[rid])
    weld=next(w for w in engine.scene['welds'] if w['id']==wid)
    attempts=[];replacement=None
    for ri,case in enumerate(rays):
        seed,trial=case['seed'],case['trial'];angle=seed['positioner_deg']
        if engine.environment.layout_collisions(angle)['blocking_pairs']: continue
        delta=engine.set_angle(angle);moved,base=search.transform_weld(engine.scene,weld,delta)
        rid=seed['candidate']['robot_id'];other=next(k for k in models if k!=rid)
        model=models[rid];q=np.asarray(seed['q_rad']);start=model.tcp(q)
        target=search.roll_target(base,seed['candidate']['roll_deg'],seed['candidate']['axis_sign'])
        if seed['candidate']['roll_deg']%5 or not model.limits_valid(q) or not np.allclose(start,target,atol=1e-6,rtol=0):
            raise ValueError('Exact source weld seed mismatch')
        direction=np.asarray(trial['direction_world']);driver=-trial['amplitude_mm']
        if not np.isclose(np.linalg.norm(direction),1.,atol=1e-12): raise ValueError('Unit straight ray required')
        # Source IK continuation at2mm, then reject-only full-arm samples every100mm.
        sparse=[];previous=q.copy();kin_ok=True
        for travel in np.arange(2.,2000.+1e-7,2.):
            pose=start.copy();pose[:3,3]+=travel*direction
            fit=analytical_continuous_ik(model,pose,previous)
            if not fit['passed']: kin_ok=False;break
            previous=np.asarray(fit['q_rad'])
            if travel%100==0: sparse.append((float(travel),previous.copy()))
        if not kin_ok:
            attempts.append(dict(ray_index=ri,reason='finite_sparse_kinematic_rejected'));continue
        for pi,park_list in enumerate(park_bank(models[other],signs[other])):
            park=np.asarray(park_list)
            selected=dict(original['initial'],**seed['candidate'],positioner_deg=angle,robot_id=rid,
                other_robot_id=other,q_rad=q.tolist(),other_q_rad=park_list,
                actual_tcp_matrix=start.ravel().tolist(),target_tcp_matrix=target.ravel().tolist())
            record=dict(ray_index=ri,park_index=pi,other_q_rad=park_list,passed=False)
            attempts.append(record)
            closed=search.full_probe(engine,rid,q,other,park,0.,moved,'closed')
            if not closed['passed']:
                record.update(reason='source_closed_rejected',pairs=closed.get('remaining_pairs',[]));continue
            blocked=None
            for travel,qi in sparse:
                check=search.full_probe(engine,rid,qi,other,park,driver,moved,'separated')
                if not check['passed']:
                    blocked=dict(travel_mm=travel,pairs=check.get('remaining_pairs',[]));break
            if blocked:
                record.update(reason='sparse_source_rejected',failure=blocked);continue
            caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
            profile=search.opening_profile(lambda d:search.full_probe(engine,rid,q,other,park,d,moved,
                'closed' if d==0. else 'opening'),caps,maximum_amplitude_mm=80.)
            if not search.prefix_available(profile,-driver):
                record['reason']='source_opening_rejected';continue
            geom=search.retreat.geometry_prefix(engine,rid,start,moved,driver,direction,2000.)
            if not geom['passed']:
                record['reason']='geometry_prefix_rejected';continue
            path=search.lift_path(engine,weld,selected,selected,
                dict(direction_world=direction.tolist(),direction_frame='current_world',length_mm=2000.),
                driver_mm=driver,opening_frames=search.opening_frames(dict(initial=selected,opening_profile=profile),start,driver))
            record.update(passed=path['passed'],reason=path.get('reason','passed'),
                last_accepted_travel_mm=path.get('last_accepted_travel_mm'),failure=path.get('failure'))
            print('FOLDED_IDLE_TRACE',ri,pi,record['passed'],record['reason'],record.get('last_accepted_travel_mm'),flush=True)
            if path['passed']:
                expected=copy.deepcopy(parent['manifest']['settings']['per_robot_inactive_park']);expected[other]=park_list
                if path_compatibility(path,signs,expected)!=(True,True): raise ValueError('Path changed sign/park')
                path.update(weld_id=wid,fixed_distance_mm=2000.,no_qp=True,geometry_certificate=geom,inactive_park_reselected=True)
                name='paths/gw7_folded_idle_2000mm.json';atomic(out/name,path)
                replacement=dict(weld_id=wid,initial=selected,opening_profile=profile,passed=True,
                    best=dict(path_file=name,path_sha256=digest(out/name),length_mm=2000.,driver_mm=driver,amplitude_mm=-driver),
                    original_initial=original.get('original_initial',original['initial']),required_straight_distance_mm=2000.,
                    mixed_wrist_branch_compatible=True,per_path_inactive_park_authoritative=True)
                break
        atomic(out/'progress.json',dict(finished_rays=ri+1,attempts=len(attempts),new_positive=bool(replacement),elapsed_s=time.perf_counter()-started))
        if replacement: break
    if replacement: rows=[replace_with_positive(r,replacement) if r['weld_id']==wid else r for r in rows]
    atomic(out/'attempts.json',dict(weld_id=wid,attempts=attempts))
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s: raise ValueError('Input changed during search '+n)
    outputs={str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')}
    outputs['attempts.json']=digest(out/'attempts.json')
    summary=dict(complete=True,points=21,requested_points=21,passed=sum(bool(r['best']) for r in rows),
        branch_compatible=sum(bool(r['best']) for r in rows),required_straight_distance_mm=2000.,
        retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),
        new_positives=int(bool(replacement)),historical_chain_intact=False,wall_s=time.perf_counter()-started)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,output_sha256=outputs,
        scope='Finite alternate inactive-arm parks; exact active weld pose and straight2m ray retained. '
        'All source opening/geometry/fulltrace gates required. Sparse positives never accepted alone. '
        'Each path freezes its own park; changed parks require fresh cold and actual15axis transfers. '
        'All21 identities and14 originals retained; finite negatives do not prove infeasibility.'))
    print('FOLDED_IDLE_SUMMARY',summary,flush=True)


if __name__=='__main__': main()
