"""Change only the idle arm park for three known source-legal GW8 rays."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from signed_wrist_model_v40 import SignedWristModel
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh candidate output required')
    baseline = ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v66/report.json'
    parent = json.loads(baseline.read_text())
    diagnostic = ROOT / 'build/welding_gw8_block_diagnostic_v67/report.json'
    evidence = json.loads(diagnostic.read_text())
    if not evidence['summary']['complete'] or len(evidence['rows']) != 3:
        raise ValueError('Completed source collision diagnosis required')
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
        **evidence['manifest']['source_sha256'], **evidence['manifest']['code_sha256']}
    for p in [baseline, diagnostic]:
        sources[str(p.relative_to(ROOT))] = digest(p)
    for name, sha in evidence['output_sha256'].items():
        sources[str((diagnostic.parent / name).relative_to(ROOT))] = sha
    codes = {str((HERE / n).relative_to(ROOT)): digest(HERE / n) for n in
        [Path(__file__).name, 'branch_point_job_v64.py', 'signed_wrist_model_v40.py',
         'analytic_trace_motion_v2.py', 'analytic_ik_v2.py', 'rejection_gates_v63.py']}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed source: ' + name)
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        distance_mm=2000., positioner_deg=60., inactive_first_axis_candidates_rad=[0., -1., -2., 1., 2.],
        active_pose_and_rays_preserved=True, original_source_gates_unchanged=True,
        per_robot_wrist_sign=parent['manifest']['settings']['per_robot_wrist_sign'],
        inactive_arm_frozen_within_each_retreat=True,
        different_parks_need_actual_15_axis_transfers=True, fresh_cold_required=True))
    atomic(out / 'manifest.json', manifest)
    started = time.perf_counter()
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    engine = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    models = engine.world.models
    signs = manifest['settings']['per_robot_wrist_sign']
    parks = parent['manifest']['settings']['per_robot_inactive_park']
    for rid, model in list(models.items()):
        models[rid] = SignedWristModel(model, signs[rid])
    wid = 'OP10L-SUB1-POINT-GW-8'
    original = next(r for r in parent['rows'] if r['weld_id'] == wid)
    weld = next(w for w in engine.scene['welds'] if w['id'] == wid)
    delta = engine.set_angle(60.)
    moved, base = search.transform_weld(engine.scene, weld, delta)
    attempts = []
    best = None
    selected_best = None
    for first_axis in manifest['settings']['inactive_first_axis_candidates_rad']:
        for case_index, record in enumerate(evidence['rows']):
            case = record['case']
            rid = case['robot_id']
            other = next(k for k in models if k != rid)
            park = np.asarray(parks[other]).copy()
            park[0] = first_axis
            if not models[other].limits_valid(park):
                raise ValueError('Proposed idle park violates source bounds/coupling/sign')
            q = np.asarray(case['q_start_rad'])
            actual = models[rid].tcp(q)
            target = search.roll_target(base, 345., -1)
            if not models[rid].limits_valid(q) or np.max(abs(actual - target)) > 1e-6:
                raise ValueError('Archived active pose no longer matches source weld frame')
            selected = copy.deepcopy(original['initial'])
            selected.update(robot_id=rid, other_robot_id=other, q_rad=q.tolist(), other_q_rad=park.tolist(),
                positioner_deg=60., roll_deg=345., axis_sign=-1,
                actual_tcp_matrix=actual.ravel().tolist(), target_tcp_matrix=target.ravel().tolist())
            caps = [pid for pid, (_, kind, _) in engine.parts[rid].items() if kind == 'electrode']
            profile = search.opening_profile(lambda d: search.full_probe(engine, rid, q, other,
                park, d, moved, 'closed' if d == 0. else 'opening'), caps, maximum_amplitude_mm=80.)
            driver = case['driver_mm']
            trial = dict(inactive_q1_rad=first_axis, case_index=case_index,
                opening_passed=search.prefix_available(profile, -driver))
            attempts.append(trial)
            if not trial['opening_passed']:
                continue
            direction = np.asarray(case['direction_world'])
            geom = search.retreat.geometry_prefix(engine, rid, actual, moved, driver, direction, 2000.)
            trial['geometry_passed'] = geom['passed']
            if not geom['passed']:
                continue
            path = search.lift_path(engine, weld, selected, selected,
                dict(direction_world=direction.tolist(), direction_frame='current_world', length_mm=2000.),
                driver_mm=driver,
                opening_frames=search.opening_frames(dict(initial=selected, opening_profile=profile), actual, driver))
            trial.update(passed=path['passed'], reason=path.get('reason'),
                last_accepted_travel_mm=path.get('last_accepted_travel_mm'),
                failure=path.get('failure'))
            print('GW8_IDLE_PARK_TRACE', first_axis, case_index, path['passed'],
                path.get('reason'), path.get('last_accepted_travel_mm'), flush=True)
            if path['passed']:
                path.update(weld_id=wid, fixed_distance_mm=2000., no_qp=True,
                    geometry_certificate=geom, inactive_park_reselected=True)
                name = 'paths/gw8_variable_park_2000mm.json'
                atomic(out / name, path)
                best = dict(path_file=name, path_sha256=digest(out / name), length_mm=2000.,
                    driver_mm=driver, amplitude_mm=-driver)
                selected_best = selected
                break
        atomic(out / 'progress.json', dict(attempts=len(attempts), passed=bool(best),
            elapsed_s=time.perf_counter()-started))
        if best:
            break
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source changed during search: ' + name)
    atomic(out / 'candidate_report.json', dict(manifest=manifest,
        summary=dict(complete=True, passed=bool(best), wall_s=time.perf_counter()-started,
            fresh_cold_required=True, full21_goal_complete=False),
        row=dict(weld_id=wid, initial=selected_best, best=best, passed=bool(best)),
        attempts=attempts, output_sha256={str(p.relative_to(out)): digest(p) for p in (out / 'paths').glob('*.json')},
        scope='Only idle-arm first axis changed, active source pose and known rays preserved. '
              'Complete source opening/geometry/2000mm trace required; positives require fresh cold and '
              'actual full15axis transitions from other parks. Finite negatives are not infeasibility proofs.'))
    print('GW8_IDLE_PARK_SUMMARY', bool(best), time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    main()
