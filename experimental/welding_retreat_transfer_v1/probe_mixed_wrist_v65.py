"""Fresh-source inventory with early kinematic rejection and exact original retention."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from retained_candidates_v58 import copy_candidate
from signed_wrist_model_v40 import SignedWristModel, branch_parks, path_compatibility
from positive_angles_v35 import angle_sequence, normalize_result
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def replace_with_positive(original, result):
    if original['weld_id'] != result['weld_id'] or not result.get('passed') or not result.get('best'):
        raise ValueError('Positive of the same weld identity required')
    selected = copy.deepcopy(result)
    if original.get('retained_existing_candidate'):
        selected['retained_existing_candidate'] = copy.deepcopy(original['retained_existing_candidate'])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retreat-out',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--priority-weld',help='Visit this deferred weld first without reducing scope')
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,30.,-30.,60.,-60.,90.,-90.,120.,-120.,150.,-150.,180.])
    args = parser.parse_args()
    source,out = args.retreat_out.resolve(),args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    parent = json.loads((source/'report.json').read_text())
    if (not parent['summary']['complete'] or parent['summary']['required_straight_distance_mm'] != 2000.
            or len(parent['rows']) != 21 or len({r['weld_id'] for r in parent['rows']}) != 21):
        raise ValueError('Full 21-weld completed 2000mm inventory required')
    angle_sequence(0.,args.angles)
    sources = {**parent['manifest']['source_sha256'],**parent['manifest']['code_sha256'],
               str((source/'report.json').relative_to(ROOT)):digest(source/'report.json')}
    sources.update({str((source/n).relative_to(ROOT)):s for n,s in parent['output_sha256'].items()})
    codes = {str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),HERE/'branch_point_job_v64.py',HERE/'rejection_gates_v63.py',HERE/'retained_candidates_v58.py',
          HERE/'signed_wrist_model_v40.py',HERE/'positive_angles_v35.py',HERE/'analytic_ik_v2.py',
          HERE/'analytic_trace_motion_v2.py',HERE/'probe_neighbor_rays_v17.py',HERE/'probe_mixed_wrist_v41.py']}
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name) != sha:
            raise ValueError('Changed source '+name)
    out.mkdir(parents=True)
    rows = copy.deepcopy(parent['rows'])
    old_paths = {}
    for i,row in enumerate(rows):
        row['mixed_wrist_branch_compatible'] = False
        if row.get('retained_existing_candidate'):
            row['retained_existing_candidate'] = copy_candidate(row['weld_id'],
                row['retained_existing_candidate'],source,out,f'paths/retained_{i:02d}.json')
        if row['best']:
            path = source/row['best']['path_file']
            old_paths[row['weld_id']] = json.loads(path.read_text())
            name = f'paths/reused_{i:02d}.json'
            (out/name).parent.mkdir(parents=True,exist_ok=True)
            (out/name).write_bytes(path.read_bytes())
            row['best'].update(path_file=name,path_sha256=digest(out/name))
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    search.ENGINE = search.retreat.VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    models = search.ENGINE.world.models
    ids = sorted(models)
    if ids != ['F46171CC-24BD-45DD-82E8-21711C110EE9','F7690943-8A45-40E4-9AD3-F92856F982FA']:
        raise ValueError('Expected exact original R2/R1 source identities')
    signs = {ids[0]:-1,ids[1]:1}
    parks = branch_parks(models,signs)
    for rid,model in list(models.items()):
        models[rid] = SignedWristModel(model,signs[rid])
    manifest = dict(source_sha256=sources,code_sha256=codes,settings=dict(
        distance_mm=2000.,posture_roll_step_deg=5.,per_robot_wrist_sign=signs,
        per_robot_inactive_park=parks,angles=args.angles,both_robot_assignment=True,
        source_coupling_unchanged=True,no_midpath_wrap=True,negative_evidence_reused=False,
        old_positive_retained_on_finite_failure=True,full_direction_bank=True,
        reject_gate_order='coarse_then_kinematic_then_fine_geometry',
        exact_local_kinematic_cache_key=True,priority_weld=args.priority_weld,
        original_full_distance_candidates_retained=sum(bool(r.get('retained_existing_candidate')) for r in rows)))
    atomic(out/'manifest.json',manifest)
    started = time.perf_counter()
    records = []
    pending = []
    # First reuse or recompute all compatible active-arm traces; no movement of
    # an inactive arm is inserted into a retreat. Every new node has a fixed park.
    for i,row in enumerate(rows):
        wid = row['weld_id']
        path = old_paths.get(wid)
        record = dict(weld_id=wid,source_positive=bool(path),attempts=[])
        records.append(record)
        if path:
            active_ok,park_ok = path_compatibility(path,signs,parks)
            if active_ok and park_ok:
                row['mixed_wrist_branch_compatible'] = True
                record['unchanged_path_reused'] = True
                print('MIXED_REUSED',wid,flush=True)
                continue
            if active_ok:
                print('MIXED_RETRACE_START',wid,flush=True)
                selected = dict(row['initial'],other_q_rad=parks[path['other_robot_id']])
                rid,other = selected['robot_id'],selected['other_robot_id']
                weld = next(w for w in search.ENGINE.scene['welds'] if w['id']==wid)
                delta = search.ENGINE.set_angle(selected['positioner_deg'])
                moved,_ = search.transform_weld(search.ENGINE.scene,weld,delta)
                q = np.asarray(selected['q_rad']); actual = models[rid].tcp(q)
                caps = [pid for pid,(_,kind,_) in search.ENGINE.parts[rid].items() if kind=='electrode']
                profile = search.opening_profile(lambda driver:search.full_probe(search.ENGINE,rid,q,other,
                    np.asarray(selected['other_q_rad']),driver,moved,'closed' if driver==0. else 'opening'),
                    caps,maximum_amplitude_mm=80.)
                driver = path['driver_mm']; trial_row = dict(initial=selected,opening_profile=profile)
                if search.prefix_available(profile,-driver):
                    geom = search.retreat.geometry_prefix(search.ENGINE,rid,actual,moved,driver,
                                                          np.asarray(path['direction_world']),2000.)
                    if geom['passed']:
                        warm = search.lift_path(search.ENGINE,weld,selected,selected,
                               dict(direction_world=path['direction_world'],direction_frame='current_world',length_mm=2000.),
                               driver_mm=driver,opening_frames=search.opening_frames(trial_row,actual,driver))
                        record['warm_trace'] = dict(passed=warm['passed'],reason=warm.get('reason'),
                                           last_accepted_travel_mm=warm.get('last_accepted_travel_mm'))
                        if warm['passed']:
                            warm.update(weld_id=wid,fixed_distance_mm=2000.,no_qp=True,geometry_certificate=geom,
                                        inactive_park_reselected=True)
                            if path_compatibility(warm,signs,parks)!=(True,True):
                                raise ValueError('Warm trace left selected branch/park domain')
                            name = f'paths/mixed_retrace_{wid}.json'
                            atomic(out/name,warm)
                            row.setdefault('original_initial',copy.deepcopy(row['initial']))
                            row.update(initial=selected,opening_profile=profile,mixed_wrist_branch_compatible=True)
                            row['best'].update(path_file=name,path_sha256=digest(out/name),length_mm=2000.)
                            print('MIXED_RETRACE_PASSED',wid,flush=True)
                            continue
                    else:
                        record['warm_geometry_failed'] = True
                else:
                    record['warm_opening_failed'] = True
        pending.append(i)
    atomic(out/'progress.json',dict(finished=21-len(pending),requested=21,
          branch_compatible=sum(r['mixed_wrist_branch_compatible'] for r in rows),
          selected_complete_retreats=sum(bool(r['best']) for r in rows),
          retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),elapsed_s=time.perf_counter()-started))
    # Different park and search domains: old failed point-angle searches cannot
    # serve as negative results for this batch.
    # Prefer reselecting existing full-distance nodes so fresh endpoint
    # connections can be investigated before the previously negative nodes.
    if args.priority_weld and args.priority_weld not in {r['weld_id'] for r in rows}:
        raise ValueError('Unknown priority weld identity')
    pending.sort(key=lambda i: (rows[i]['weld_id'] != args.priority_weld,
                               not bool(rows[i].get('retained_existing_candidate'))))
    for ji,index in enumerate(pending):
        row = rows[index];wid = row['weld_id']
        sequence = angle_sequence(row['initial']['positioner_deg'],args.angles)
        print('MIXED_POINT_START',wid,sequence,flush=True)
        for angle in sequence:
            result = normalize_result(search.point_job(wid,out,10000,10000,angle,parks),wid,angle)
            atomic(out/f'{wid}_angle_{angle:g}.json',result)
            records[index]['attempts'].append(dict(positioner_deg=angle,passed=bool(result['best']),
                  report_file=f'{wid}_angle_{angle:g}.json',elapsed_s=result.get('elapsed_s')))
            if result['best']:
                path = json.loads((out/result['best']['path_file']).read_text())
                if path_compatibility(path,signs,parks)!=(True,True):
                    raise ValueError('New path left selected source wrist/park domains')
                result['mixed_wrist_branch_compatible'] = True
                rows[index] = replace_with_positive(rows[index],result)
                break
        atomic(out/'progress.json',dict(finished=21-len(pending)+ji+1,requested=21,
              branch_compatible=sum(r['mixed_wrist_branch_compatible'] for r in rows),
              selected_complete_retreats=sum(bool(r['best']) for r in rows),
          retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),elapsed_s=time.perf_counter()-started))
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name) != sha:
            raise ValueError('Source changed during mixed branch search '+name)
    outputs = {str(p.relative_to(out)):digest(p) for p in out.glob('OP10L*.json')}
    outputs.update({str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')})
    summary = dict(complete=True,points=21,passed=sum(bool(r['best']) for r in rows),
                   required_straight_distance_mm=2000.,requested_points=21,
                   branch_compatible=sum(r['mixed_wrist_branch_compatible'] for r in rows),
                   newly_searched_points=len(pending),
                   retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),
                   historical_chain_intact=False,wall_s=time.perf_counter()-started)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,search_records=records,
           output_sha256=outputs,scope='Fresh source-matched chain after explicitly disclosed historical dependency loss; '
           'Source limits/coupling/geometry unchanged; complete opening and 2000mm trace checked. '
           'Finite failed searches do not establish impossibility. Old positive retained and marked '
           'or separately retained on finite failed re-selection. All new nodes require independent cold '
           'and real 15-axis transfers; neither parking jumps nor midpath wrap permitted.'))
    print('MIXED_WRIST_SUMMARY',summary,flush=True)


if __name__ == '__main__':
    main()
