"""Retry completed source-collision candidates with an explicit alternate park."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np

import probe_analytic_retreat_v2 as exp
from analytic_trace_motion_v2 import lift_path
from run_geometry import atomic, digest
from run_arm import full_probe
from run_qp import opening_frames
from variable_core import opening_profile, prefix_available

ROOT = exp.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate-budget', type=int, default=40)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Preserve evidence')
    exp.retreat.assert_frozen_geometry_cache()
    prior_path = ROOT/'build/welding_retreat_200cm_other_park_v12/report.json'
    prior = json.loads(prior_path.read_text())
    if not prior['summary']['complete'] or prior['summary']['required_straight_distance_mm'] != 2000.:
        raise RuntimeError('Completed 2000mm baseline required')
    evidence_path = args.evidence.resolve()
    evidence = json.loads(evidence_path.read_text())
    original_path = ROOT/'build/welding_variable_open_v1/report.json'
    original = json.loads(original_path.read_text())
    old = dict(evidence['initial'])
    if evidence.get('best') or evidence.get('required_straight_distance_mm') != 2000.:
        raise RuntimeError('Require completed failed same-distance evidence')
    remote_row = next(r for r in prior['rows'] if r['best'] and
                      r['initial']['other_robot_id'] == old['other_robot_id'])
    remote_path = prior_path.parent/remote_row['best']['path_file']
    remote = json.loads(remote_path.read_text())
    if old['other_robot_id'] != remote['other_robot_id']:
        raise RuntimeError('Alternate park belongs to another robot')
    park = np.asarray(remote['other_q_rad'])
    sources = {**prior['manifest']['source_sha256'], **prior['manifest']['code_sha256']}
    for p in (prior_path, evidence_path, original_path, remote_path):
        sources[str(p.relative_to(ROOT))] = digest(p)
    for name, value in prior['output_sha256'].items():
        p = prior_path.parent/name
        if digest(p) != value:
            raise RuntimeError('Baseline output changed')
        sources[str(p.relative_to(ROOT))] = value
    codes = {str(p.relative_to(ROOT)): digest(p) for p in
             (Path(__file__), HERE/'analytic_ik_v2.py', HERE/'analytic_trace_motion_v2.py')}
    for name, value in {**sources, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Source changed')
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        distance_mm=2000., alternate_other_park=park.tolist(),
        candidate_budget=args.candidate_budget, scope='Completed explicit-angle collision candidates; alternate source-legal other-arm park'))
    atomic(out/'manifest.json', manifest)
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    engine = exp.retreat.VariableToolWorld(scene)
    layout = engine.environment.layout_collisions(old['positioner_deg'])
    if layout['blocking_pairs']:
        raise RuntimeError('Requested positioner layout blocked')
    engine.set_angle(old['positioner_deg'])
    model = engine.world.models[old['robot_id']]
    weld = next(w for w in scene['welds'] if w['id'] == evidence['weld_id'])
    candidates = []
    for pose in evidence['alternative_pose_trials']:
        for initial in pose['trials']:
            for trial in initial['directions']:
                if trial.get('reason') == 'full_retreat_blocked':
                    candidates.append((pose, initial, trial))
    candidates.sort(key=lambda x: -(x[2].get('last_accepted_travel_mm') or 0))
    records, best, best_initial = [], None, None
    seen = set()
    started = time.perf_counter()
    profiles = {}
    for pose, initial, trial in candidates:
        key = tuple(np.round(np.r_[initial['q_rad'], trial['direction_world'], trial['amplitude_mm']], 10))
        if key in seen:
            continue
        seen.add(key)
        if len(records) >= args.candidate_budget:
            break
        q = np.asarray(initial['q_rad'])
        target = model.tcp(q)
        selected = dict(old, **pose['candidate'], q_rad=q.tolist(), other_q_rad=park.tolist(),
                        target_tcp_matrix=target.ravel().tolist(), actual_tcp_matrix=target.ravel().tolist())
        moved, _ = engine.target(weld, selected)
        pkey = tuple(q)
        if pkey not in profiles:
            caps = [pid for pid, (_, kind, _) in engine.parts[old['robot_id']].items() if kind == 'electrode']
            profiles[pkey] = opening_profile(lambda driver: full_probe(engine, old['robot_id'], q,
                old['other_robot_id'], park, driver, moved, 'closed' if driver == 0. else 'opening'),
                caps, maximum_amplitude_mm=80.)
        profile = profiles[pkey]
        record = dict(candidate=pose['candidate'], q_rad=q.tolist(), other_q_rad=park.tolist(),
                      amplitude_mm=trial['amplitude_mm'], direction_world=trial['direction_world'])
        records.append(record)
        if not prefix_available(profile, trial['amplitude_mm']):
            record.update(passed=False, reason='alternate_park_opening_rejected')
            continue
        driver = -trial['amplitude_mm']
        ray = dict(direction_world=trial['direction_world'], direction_frame='current_world', length_mm=2000.)
        geom = exp.retreat.geometry_prefix(engine, old['robot_id'], target, moved, driver,
                                          np.asarray(trial['direction_world']), 2000.)
        if not geom['passed']:
            record.update(passed=False, reason='geometry_certificate_rejected', geometry=geom)
            continue
        row = dict(initial=selected, best=None, opening_profile=profile)
        path = lift_path(engine, weld, selected, selected, ray, driver_mm=driver,
                         opening_frames=opening_frames(row, target, driver))
        record.update(passed=path['passed'], reason=path.get('reason'),
                      last_accepted_travel_mm=path.get('last_accepted_travel_mm'), failure=path.get('failure'))
        print('ALTERNATE_PARK_TRIAL', len(records), record['passed'], record['reason'],
              record.get('last_accepted_travel_mm'), flush=True)
        atomic(out/'progress.json', dict(finished=len(records), best=bool(best), elapsed_s=time.perf_counter()-started))
        if path['passed']:
            name = f'paths/retreat_other_park_{evidence["weld_id"]}.json'
            path.update(weld_id=evidence['weld_id'], fixed_distance_mm=2000., no_qp=True,
                        geometry_certificate=geom, other_park_reselected=True)
            atomic(out/name, path)
            best = dict(driver_mm=driver, amplitude_mm=trial['amplitude_mm'], path_file=name,
                        path_sha256=digest(out/name), length_mm=2000.)
            best_initial = selected
            break
    rows = copy.deepcopy(prior['rows'])
    for i, row in enumerate(rows):
        if row['best']:
            name = f'paths/reused_retreat_{i:02d}.json'
            atomic(out/name, json.loads((prior_path.parent/row['best']['path_file']).read_text()))
            row['best'].update(path_file=name, path_sha256=digest(out/name))
    if best:
        result = dict(weld_id=evidence['weld_id'], initial=best_initial, original_initial=old,
                      best=best, passed=True, required_straight_distance_mm=2000.,
                      other_park_reselected=True, trials=records)
        rows = [result if r['weld_id'] == result['weld_id'] else r for r in rows]
    atomic(out/'trials.json', dict(rows=records))
    outputs = {str(p.relative_to(out)): digest(p) for p in (out/'paths').glob('*.json')}
    outputs['trials.json'] = digest(out/'trials.json')
    for name, value in {**sources, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Inputs changed during run')
    summary = dict(complete=True, points=21, passed=sum(bool(r['best']) for r in rows),
                   required_straight_distance_mm=2000., trials=len(records), wall_s=time.perf_counter()-started)
    atomic(out/'report.json', dict(manifest=manifest, summary=summary, rows=rows, output_sha256=outputs,
          scope='Explicit alternate parked robot state; no teleport motion claim. Finite diagnostic collision-candidate retries. Every positive complete 2000mm path requires cold audit, then both-arm transfers.'))
    print('ALTERNATE_PARK_SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
