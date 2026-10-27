"""Refine a failed near-endpoint ray on the existing five-degree sphere."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from probe_neighbor_rays_v17 import neighbor_bank, kinematic_rejection
from signed_wrist_model_v40 import SignedWristModel, path_compatibility
from probe_mixed_wrist_v65 import replace_with_positive
from retained_candidates_v58 import copy_candidate
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def near_seed(archive, minimum):
    seeds = []
    for pose in archive['alternative_pose_trials']:
        for trial in pose['trials']:
            for ray in trial['directions']:
                if (ray.get('reason') == 'full_retreat_blocked' and
                        minimum <= ray.get('last_accepted_travel_mm', 0.) < 2000.):
                    seeds.append(dict(candidate=pose['candidate'], q_start_rad=trial['q_rad'], ray=ray))
    if not seeds:
        raise ValueError('A failed near-endpoint trace is required')
    return copy.deepcopy(max(seeds, key=lambda s: s['ray']['last_accepted_travel_mm']))


def release_neighbors(start, reference, supports, count):
    if not 1 <= count <= 2522:
        raise ValueError('Finite five-degree direction budget required')
    bank = neighbor_bank(start, reference, 2522)
    allowed = [d for d in bank if all(np.asarray(s['outward_normal']) @ d > 1e-6 for s in supports)]
    return np.asarray(allowed[:count]).reshape((-1, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--directions', type=int, default=256)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    baseline = ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v66'
    input_path = ROOT / 'build/welding_retreat_200cm_refined_angles_v69/OP10L-SUB1-POINT-GW-1_angle_75.json'
    diagnostic_path = ROOT / 'build/welding_gw1_near_endpoint_diagnostic_v71/report.json'
    parent = json.loads((baseline / 'report.json').read_text())
    archive = json.loads(input_path.read_text())
    diagnosis = json.loads(diagnostic_path.read_text())
    if not diagnosis['summary']['complete']:
        raise ValueError('Completed source diagnosis required')
    if len(parent['rows']) != 21 or parent['summary']['required_straight_distance_mm'] != 2000.:
        raise ValueError('Full21 literal2m inventory required')
    seed = near_seed(archive, 1900.)
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
               **diagnosis['manifest']['source_sha256']}
    for p in (baseline / 'report.json', input_path, diagnostic_path):
        sources[str(p.relative_to(ROOT))] = digest(p)
    for name, sha in parent['output_sha256'].items():
        sources[str((baseline / name).relative_to(ROOT))] = sha
    codes = {str((HERE / n).relative_to(ROOT)): digest(HERE / n) for n in
             [Path(__file__).name, 'probe_neighbor_rays_v17.py', 'branch_point_job_v64.py',
              'probe_mixed_wrist_v65.py', 'retained_candidates_v58.py', 'analytic_ik_v2.py',
              'analytic_trace_motion_v2.py', 'signed_wrist_model_v40.py']}
    for n, sha in {**sources, **codes}.items():
        if digest(ROOT / n) != sha:
            raise ValueError('Changed input ' + n)
    out.mkdir(parents=True)
    rows = copy.deepcopy(parent['rows'])
    for i, row in enumerate(rows):
        if row.get('retained_existing_candidate'):
            row['retained_existing_candidate'] = copy_candidate(row['weld_id'],
                row['retained_existing_candidate'], baseline, out, f'paths/retained_{i:02d}.json')
        if row['best']:
            src = baseline / row['best']['path_file']
            name = f'paths/reused_{i:02d}.json'
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_bytes(src.read_bytes())
            row['best'].update(path_file=name, path_sha256=digest(out / name))
    signs = parent['manifest']['settings']['per_robot_wrist_sign']
    parks = parent['manifest']['settings']['per_robot_inactive_park']
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        distance_mm=2000., azimuth_step_deg=5, elevation_step_deg=5,
        direction_budget=args.directions, positioner_deg=75., fixed_weld_pose=True,
        per_robot_wrist_sign=signs, per_robot_inactive_park=parks,
        original_source_gates_unchanged=True, no_midpath_wrap=True,
        endpoint_collision_gate='reject_only', fresh_cold_required=True))
    atomic(out / 'manifest.json', manifest)
    started = time.perf_counter()
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    engine = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    models = engine.world.models
    for rid, model in list(models.items()):
        models[rid] = SignedWristModel(model, signs[rid])
    wid = archive['weld_id']
    original = next(r for r in rows if r['weld_id'] == wid)
    weld = next(w for w in engine.scene['welds'] if w['id'] == wid)
    if engine.environment.layout_collisions(75.)['blocking_pairs']:
        raise ValueError('Blocked fixed source layout')
    delta = engine.set_angle(75.)
    moved, base = search.transform_weld(engine.scene, weld, delta)
    rid = seed['candidate']['robot_id']
    other = next(k for k in models if k != rid)
    q = np.asarray(seed['q_start_rad'])
    start = models[rid].tcp(q)
    target = search.roll_target(base, seed['candidate']['roll_deg'], seed['candidate']['axis_sign'])
    if not models[rid].limits_valid(q) or not np.allclose(start, target, atol=1e-6, rtol=0):
        raise ValueError('Seed must match exact legal source weld pose')
    selected = dict(original['initial'], **seed['candidate'],
                    other_robot_id=other, q_rad=q.tolist(), other_q_rad=parks[other],
                    positioner_deg=75., actual_tcp_matrix=start.ravel().tolist(),
                    target_tcp_matrix=target.ravel().tolist())
    caps = [pid for pid, (_, kind, _) in engine.parts[rid].items() if kind == 'electrode']
    profile = search.opening_profile(lambda d: search.full_probe(engine, rid, q, other,
        np.asarray(parks[other]), d, moved, 'closed' if d == 0. else 'opening'), caps,
        maximum_amplitude_mm=80.)
    records = []
    best = None
    kin_cache = {}
    for amplitude in (60., 80., 30., 10.):
        if not search.prefix_available(profile, amplitude):
            continue
        driver = -amplitude
        open_state = engine.state(rid, start, driver, moved, engine.world.static, 'opening')
        context = engine.release_context(rid, start, moved, dict(open_state=open_state), driver_mm=driver)
        directions = release_neighbors(start, np.asarray(seed['ray']['direction_world']),
                                       context['supports'], args.directions)
        for di, direction in enumerate(directions):
            key = tuple(direction)
            record = dict(amplitude_mm=amplitude, direction_index=di, direction_world=direction.tolist())
            records.append(record)
            if key not in kin_cache:
                kin_cache[key] = kinematic_rejection(models[rid], start, q, direction)
            kin = kin_cache[key]
            if not kin['passed']:
                record['reason'] = 'finite_continuation_prefilter_rejected'
                continue
            endpoint = search.full_probe(engine, rid, np.asarray(kin['q_end_rad']), other,
                np.asarray(parks[other]), driver, moved, 'separated')
            if not endpoint['passed']:
                record.update(reason='endpoint_source_collision_rejected', failure=endpoint)
                continue
            geom = search.retreat.geometry_prefix(engine, rid, start, moved, driver, direction, 2000.)
            if not geom['passed']:
                record.update(reason='geometry_prefix_rejected', geometry=geom)
                continue
            path = search.lift_path(engine, weld, selected, selected,
                dict(direction_world=direction.tolist(), direction_frame='current_world', length_mm=2000.),
                driver_mm=driver, opening_frames=search.opening_frames(
                    dict(initial=selected, opening_profile=profile), start, driver))
            record.update(reason=path.get('reason', 'passed'), passed=path['passed'],
                          last_accepted_travel_mm=path.get('last_accepted_travel_mm'), failure=path.get('failure'))
            print('NEAR_TRACE', amplitude, di, path['passed'], path.get('last_accepted_travel_mm'), flush=True)
            if path['passed']:
                if path_compatibility(path, signs, parks) != (True, True):
                    raise ValueError('Path changed branch or idle park')
                path.update(weld_id=wid, fixed_distance_mm=2000., no_qp=True, geometry_certificate=geom)
                name = 'paths/near_retreat_gw1.json'
                atomic(out / name, path)
                best = dict(path_file=name, path_sha256=digest(out / name), length_mm=2000.,
                            driver_mm=driver, amplitude_mm=amplitude)
                replacement = dict(weld_id=wid, initial=selected, best=best, passed=True,
                    opening_profile=profile, mixed_wrist_branch_compatible=True,
                    required_straight_distance_mm=2000., original_initial=original['initial'])
                rows = [replace_with_positive(r, replacement) if r['weld_id'] == wid else r for r in rows]
                break
            atomic(out / 'progress.json', dict(attempts=len(records), amplitude_mm=amplitude,
                direction_index=di, new_positive=False, elapsed_s=time.perf_counter()-started))
        if best:
            break
        atomic(out / 'progress.json', dict(attempts=len(records), amplitude_mm=amplitude,
            new_positive=bool(best), elapsed_s=time.perf_counter()-started))
        print('NEAR_AMPLITUDE_FINISHED', amplitude, len(records), bool(best), flush=True)
    atomic(out / 'trials.json', dict(seed=seed, records=records, opening_profile=profile))
    for n, sha in {**sources, **codes}.items():
        if digest(ROOT / n) != sha:
            raise ValueError('Changed input during search ' + n)
    outputs = {str(p.relative_to(out)): digest(p) for p in (out / 'paths').glob('*.json')}
    outputs['trials.json'] = digest(out / 'trials.json')
    summary = dict(complete=True, points=21, requested_points=21, passed=sum(bool(r['best']) for r in rows),
        branch_compatible=sum(bool(r['best']) for r in rows), required_straight_distance_mm=2000.,
        retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),
        new_positive=bool(best), historical_chain_intact=False, wall_s=time.perf_counter()-started)
    atomic(out / 'report.json', dict(manifest=manifest, summary=summary, rows=rows, output_sha256=outputs,
        scope='Local finite five-degree sphere refinement, exact weld pose preserved. Full original '
              'opening/geometry/source-arm trace required. Endpoint gate only rejects; fresh cold and '
              'real full15axis transitions still required. All21 identities and original14 retained; '
              'finite negatives are not impossibility proofs.'))
    print('NEAR_SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
