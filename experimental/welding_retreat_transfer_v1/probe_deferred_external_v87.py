"""Fresh finite external-angle domain, preserving each selected path's idle park."""
import argparse
import json
import time
from pathlib import Path
import branch_point_job_v64 as search
from compact_inventory_v82 import compact_rows, link_candidate
from probe_mixed_wrist_v65 import replace_with_positive
from signed_wrist_model_v40 import SignedWristModel, path_compatibility
from positive_angles_v35 import normalize_result
from run_geometry import atomic, digest, initialize_worker

ROOT=search.ROOT
HERE=Path(__file__).resolve().parent
ANGLES={'OP10L-SUB1-POINT-GW-6':[100.,110.],
        'OP10L-SUB2-POINT-GW-4':[140.,160.],
        'OP10L-SUB2-POINT-GW-6':[140.,160.],
        'OP10L-SUB3-POINT-GW-14':[130.,140.]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists(): raise FileExistsError('Fresh evidence required')
    baseline=ROOT/'build/welding_retreat_200cm_folded_idle_v86'
    source=baseline/'report.json';parent=json.loads(source.read_text())
    if (not parent['summary']['complete'] or len(parent['rows'])!=21 or
            len({r['weld_id'] for r in parent['rows']})!=21 or
            parent['summary']['required_straight_distance_mm']!=2000. or
            {r['weld_id'] for r in parent['rows'] if not r['best']}!=set(ANGLES)):
        raise ValueError('Completed full21 literal2m inventory with expected remaining4 required')
    rows=compact_rows(parent['rows'],source,ROOT)
    sources={**parent['manifest']['source_sha256'],**parent['manifest']['code_sha256'],
             str(source.relative_to(ROOT)):digest(source)}
    sources.update({str((baseline/n).relative_to(ROOT)):s for n,s in parent['output_sha256'].items()})
    codes={str((HERE/n).relative_to(ROOT)):digest(HERE/n) for n in
        [Path(__file__).name,'branch_point_job_v64.py','compact_inventory_v82.py',
         'probe_mixed_wrist_v65.py','signed_wrist_model_v40.py','positive_angles_v35.py',
         'rejection_gates_v63.py','analytic_ik_v2.py','analytic_trace_motion_v2.py']}
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s: raise ValueError('Changed input '+n)
    out.mkdir(parents=True)
    for i,row in enumerate(rows):
        if row.get('retained_existing_candidate'):
            row['retained_existing_candidate']=link_candidate(row['weld_id'],row['retained_existing_candidate'],baseline,out,f'paths/retained_{i:02d}.json')
        if row['best']:
            row['best']=link_candidate(row['weld_id'],row['best'],baseline,out,f'paths/reused_{i:02d}.json')
    signs=parent['manifest']['settings']['per_robot_wrist_sign']
    parks=parent['manifest']['settings']['per_robot_inactive_park']
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(
        distance_mm=2000.,angles_by_weld=ANGLES,posture_roll_step_deg=5.,
        both_robot_assignment=True,full_direction_bank=True,negative_evidence_reused=False,
        per_robot_wrist_sign=signs,per_robot_inactive_park=parks,
        selected_per_path_park_preserved=True,no_midpath_wrap=True,original_source_gates_unchanged=True,
        fresh_cold_and_actual15axis_required=True))
    atomic(out/'manifest.json',manifest)
    initialize_worker();started=time.perf_counter()
    search.retreat.assert_frozen_geometry_cache()
    search.ENGINE=search.retreat.VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    for rid,model in list(search.ENGINE.world.models.items()):
        search.ENGINE.world.models[rid]=SignedWristModel(model,signs[rid])
    records=[];new=0
    for wi,(wid,angles) in enumerate(ANGLES.items()):
        record=dict(weld_id=wid,attempts=[]);records.append(record)
        for angle in angles:
            result=normalize_result(search.point_job(wid,out,10000,10000,angle,parks),wid,angle)
            name=f'{wid}_angle_{angle:g}.json';atomic(out/name,result)
            record['attempts'].append(dict(positioner_deg=angle,passed=bool(result['best']),report_file=name,elapsed_s=result.get('elapsed_s')))
            if result['best']:
                path=json.loads((out/result['best']['path_file']).read_text())
                if path_compatibility(path,signs,parks)!=(True,True): raise ValueError('New path changed proposed sign/park')
                old=next(r for r in rows if r['weld_id']==wid)
                result['original_initial']=old.get('original_initial',old['initial'])
                result['mixed_wrist_branch_compatible']=True
                rows=[replace_with_positive(r,result) if r['weld_id']==wid else r for r in rows]
                new+=1;break
        atomic(out/'progress.json',dict(finished=17+wi+1,requested=21,compatible=sum(bool(r['best']) for r in rows),elapsed_s=time.perf_counter()-started))
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s: raise ValueError('Input changed during search '+n)
    outputs={str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')}
    outputs.update({p.name:digest(p) for p in out.glob('OP10L*_angle_*.json') if not p.name.endswith('_progress.json')})
    summary=dict(complete=True,points=21,requested_points=21,passed=sum(bool(r['best']) for r in rows),
        branch_compatible=sum(bool(r['best']) for r in rows),required_straight_distance_mm=2000.,
        retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),
        new_positives=new,historical_chain_intact=False,wall_s=time.perf_counter()-started)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,search_records=records,output_sha256=outputs,
        scope='Fresh finite external angles on remaining4 identities; old17 selected paths retained with their actual idle parks. '
        'Original source fullopening/geometry/full2m trace required. All21 identities and14 original candidates retained. '
        'Finite negatives do not prove infeasibility; positives require owncold and actual15axis transfers/finalcold.'))
    print('DEFERRED_EXTERNAL_SUMMARY',summary,flush=True)


if __name__=='__main__': main()
