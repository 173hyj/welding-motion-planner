"""Fresh 1000mm endpoint certification, preserving the entire 21-point inventory."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import branch_point_job_v64 as search
from distance_prefix_v97 import crop_exact_prefix
from signed_wrist_model_v40 import SignedWristModel
from probe_near_retreat_v72 import release_neighbors
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent
DISTANCE = 1000.
ENGINE = None


def refresh_prefix(row, old_path):
    engine = ENGINE
    path = crop_exact_prefix(old_path, DISTANCE)
    selected = row['initial']; rid = selected['robot_id']
    engine.set_angle(selected['positioner_deg'])
    weld = next(w for w in engine.scene['welds'] if w['id'] == row['weld_id'])
    moved, _ = engine.target(weld, selected)
    model = engine.world.models[rid]
    start = model.tcp(np.asarray(selected['q_rad']))
    direction = np.asarray(path['direction_world'])
    driver = path['driver_mm']
    endpoint = model.tcp(np.asarray(path['q_end_rad']))
    clearance = engine.minimum_distance(rid, endpoint, driver, engine.world.static)
    probe = search.full_probe(engine, rid, np.asarray(path['q_end_rad']),
        path['other_robot_id'], np.asarray(path['other_q_rad']), driver, moved, 'separated')
    if clearance < 5. or not probe['passed']:
        return None, dict(reason='fresh_1000mm_endpoint_rejected', clearance_mm=clearance, probe=probe)
    geom = search.retreat.geometry_prefix(engine, rid, start, moved, driver, direction, DISTANCE)
    if not geom['passed']:
        return None, dict(reason='fresh_1000mm_geometry_rejected', geometry=geom)
    errors = []
    rotations = []
    for frame in path['frames']:
        actual = model.tcp(np.asarray(frame['q_rad']))
        offset = actual[:3, 3] - start[:3, 3]
        errors.append(float(np.linalg.norm(offset - (offset @ direction) * direction)))
        rotations.append(float(np.degrees(np.linalg.norm(Rotation.from_matrix(
            start[:3, :3].T @ actual[:3, :3]).as_rotvec()))))
    path.update(passed=True, geometry_certificate=geom, endpoint_clearance_mm=clearance,
        maximum_saved_frame_line_error_mm=max(errors),
        maximum_saved_frame_rotation_error_deg=max(rotations),
        fresh_cold_required=True, exact_saved_prefix=True)
    return path, dict(reason='fresh_endpoint_and_tool_geometry_passed',
                      full_prefix_collision_audit='pending independent cold verification')


def search_pending(row, seeds, parks, directions):
    engine = ENGINE; wid = row['weld_id']
    weld = next(w for w in engine.scene['welds'] if w['id'] == wid)
    initial_seed = dict(positioner_deg=row['initial']['positioner_deg'],
        candidate={k: row['initial'][k] for k in ('robot_id', 'roll_deg', 'axis_sign')},
        q_rad=row['initial']['q_rad'], reference=None)
    records = []
    for si, seed in enumerate([initial_seed] + seeds):
        angle = seed['positioner_deg']; candidate = seed['candidate']; rid = candidate['robot_id']
        model = engine.world.models[rid]; q = np.asarray(seed['q_rad'])
        trial = dict(seed=seed, trials=[]); records.append(trial)
        if not model.limits_valid(q) or engine.environment.layout_collisions(angle)['blocking_pairs']:
            trial['reason'] = 'source_limits_or_layout_rejected'; continue
        engine.set_angle(angle)
        other = next(r for r in engine.world.models if r != rid)
        selected = dict(row['initial'], **candidate, positioner_deg=angle,
                        q_rad=q.tolist(), other_robot_id=other, other_q_rad=parks[other])
        moved, _ = engine.target(weld, selected); start = model.tcp(q)
        selected.update(actual_tcp_matrix=start.ravel().tolist())
        # Candidate identity must represent the original five-degree weld pose.
        delta = engine.delta
        _, base = search.transform_weld(engine.scene, weld, delta)
        target = search.roll_target(base, candidate['roll_deg'], candidate['axis_sign'])
        if candidate['roll_deg'] % 5 or not np.allclose(start, target, atol=1e-6, rtol=0):
            raise ValueError('Seed does not match five-degree source weld target')
        selected['target_tcp_matrix'] = target.ravel().tolist()
        caps = [pid for pid, (_, kind, _) in engine.parts[rid].items() if kind == 'electrode']
        profile = search.opening_profile(lambda d: search.full_probe(engine, rid, q, other,
            np.asarray(parks[other]), d, moved, 'closed' if d == 0. else 'opening'),
            caps, maximum_amplitude_mm=185.)
        trial['opening_profile'] = profile
        print('ONE_METER_SEED', wid, si, angle, candidate, flush=True)
        for amplitude in (10., 30., 60., 80., 90., 100., 120., 140., 160., 180., 185.):
            if not search.prefix_available(profile, amplitude): continue
            driver = -amplitude
            state = engine.state(rid, start, driver, moved, engine.world.static, 'opening')
            context = engine.release_context(rid, start, moved, dict(open_state=state), driver_mm=driver)
            reference = np.asarray(seed['reference'] if seed['reference'] is not None
                                   else context['preferred_outward_normal'])
            bank = release_neighbors(start, reference, context['supports'], directions)
            for di, direction in enumerate(bank):
                t = dict(amplitude_mm=amplitude, direction_index=di, direction_world=direction.tolist())
                trial['trials'].append(t)
                end = start.copy(); end[:3, 3] += DISTANCE * direction
                ends, _ = search.solve_seed_bank(model, end, [q], random_count=0)
                if not ends: t['reason'] = 'finite_1000mm_endpoint_ik_rejected'; continue
                # Reject-only: no straight trace can finish at a blocked endpoint.
                # Every positive still uses unchanged geometry and full sampled trace.
                if not any(search.full_probe(engine, rid, np.asarray(fit['q_rad']), other,
                        np.asarray(parks[other]), driver, moved, 'separated')['passed'] for fit in ends):
                    t['reason'] = 'all_finite_endpoint_source_candidates_blocked'; continue
                blocked = False
                for s in np.arange(5., DISTANCE + 1e-7, 10.):
                    pose = start.copy(); pose[:3, 3] += s * direction
                    if engine.state(rid, pose, driver, moved, engine.world.static, 'separated')['blocking_pairs']:
                        blocked = True; break
                if blocked: t['reason'] = 'coarse_tool_geometry_rejected'; continue
                geom = search.retreat.geometry_prefix(engine, rid, start, moved, driver, direction, DISTANCE)
                if not geom['passed']: t['reason'] = 'tool_geometry_certificate_rejected'; continue
                path = search.lift_path(engine, weld, selected, selected,
                    dict(direction_world=direction.tolist(), direction_frame='current_world', length_mm=DISTANCE),
                    driver_mm=driver, opening_frames=search.opening_frames(
                        dict(initial=selected, opening_profile=profile), start, driver))
                t.update(passed=path['passed'], reason=path.get('reason', 'passed'),
                         last_accepted_travel_mm=path.get('last_accepted_travel_mm'))
                print('ONE_METER_TRACE', wid, si, amplitude, di, path['passed'],
                      path.get('last_accepted_travel_mm'), flush=True)
                if path['passed']:
                    path.update(weld_id=wid, fixed_distance_mm=DISTANCE, no_qp=True, geometry_certificate=geom)
                    return path, selected, profile, records
    return None, None, None, records


def point_job(index, row, baseline, out, seeds, parks, directions):
    started = time.perf_counter(); old_best = row.get('best')
    records = []; path = None; selected = row['initial']; profile = row.get('opening_profile')
    if old_best:
        old = json.loads((baseline / old_best['path_file']).read_text())
        if digest(baseline / old_best['path_file']) != old_best['path_sha256']:
            raise ValueError('Archived source path changed')
        path, record = refresh_prefix(row, old); records.append(record)
    if path is None:
        path, changed, changed_profile, searched = search_pending(row, seeds, parks, directions)
        records.extend(searched)
        if path:
            selected, profile = changed, changed_profile
    result = dict(weld_id=row['weld_id'], initial=selected, original_initial=row.get('original_initial', row['initial']),
        opening_profile=profile, best=None, passed=path is not None, required_straight_distance_mm=DISTANCE,
        previous_2000mm_candidate=old_best, fresh_cold_required=True,
        per_path_inactive_park_authoritative=True, elapsed_s=time.perf_counter()-started)
    if path:
        name = f'paths/retreat_{index:02d}.json'; atomic(out / name, path)
        result['best'] = dict(path_file=name, path_sha256=digest(out / name), length_mm=DISTANCE,
            driver_mm=path['driver_mm'], amplitude_mm=abs(path['driver_mm']))
    atomic(out / f'trials_{index:02d}.json', records)
    atomic(out / f'point_{index:02d}.json', result)
    print('ONE_METER_POINT', row['weld_id'], result['passed'], result['elapsed_s'], flush=True)
    return index, result


def main():
    global ENGINE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--directions', type=int, default=256)
    args = p.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.directions <= 2522:
        raise ValueError('Finite search and worker budget required')
    out = args.out.resolve()
    if out.exists(): raise FileExistsError('Fresh output required')
    baseline = ROOT / 'build/welding_retreat_200cm_large_new_seeds_v93'
    parent = json.loads((baseline / 'report.json').read_text())
    if not parent['summary']['complete'] or len(parent['rows']) != 21:
        raise ValueError('Completed full identity inventory required')
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256']}
    sources[str((baseline / 'report.json').relative_to(ROOT))] = digest(baseline / 'report.json')
    for n, value in parent['output_sha256'].items(): sources[str((baseline / n).relative_to(ROOT))] = value
    codes = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), HERE / 'distance_prefix_v97.py',
        HERE / 'analytic_trace_motion_v2.py', HERE / 'analytic_ik_v2.py', HERE / 'probe_near_retreat_v72.py']}
    for n, value in {**sources, **codes}.items():
        if digest(ROOT / n) != value: raise ValueError('Changed source ' + n)
    settings = parent['manifest']['settings']
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(distance_mm=DISTANCE,
        direction_budget=args.directions, roll_step_deg=5, sphere_step_deg=5, no_qp=True,
        per_robot_wrist_sign=settings['per_robot_wrist_sign'],
        per_robot_inactive_park=settings['per_robot_inactive_park'],
        per_path_inactive_park_authoritative=True, fresh_cold_required=True, no_2000mm_endpoint_transfer_reuse=True))
    out.mkdir(parents=True); atomic(out / 'manifest.json', manifest)
    started = time.perf_counter(); initialize_worker(); search.retreat.assert_frozen_geometry_cache()
    ENGINE = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    for rid, model in list(ENGINE.world.models.items()):
        ENGINE.world.models[rid] = SignedWristModel(model, settings['per_robot_wrist_sign'][rid])
    rows = [None] * 21
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
        jobs = [pool.submit(point_job, i, row, baseline, out,
            settings['seeds'].get(row['weld_id'], []), settings['per_robot_inactive_park'], args.directions)
            for i, row in enumerate(parent['rows'])]
        for f in concurrent.futures.as_completed(jobs):
            i, row = f.result(); rows[i] = row
            atomic(out / 'progress.json', dict(finished=sum(r is not None for r in rows), requested=21,
                passed=sum(bool(r and r['best']) for r in rows), elapsed_s=time.perf_counter()-started))
    for n, value in {**sources, **codes}.items():
        if digest(ROOT / n) != value: raise ValueError('Inputs changed during run ' + n)
    outputs = {str(p.relative_to(out)): digest(p) for p in out.rglob('*.json') if p.name not in
               ('manifest.json', 'progress.json', 'report.json')}
    summary = dict(complete=True, points=21, requested_points=21, passed=sum(bool(r['best']) for r in rows),
        required_straight_distance_mm=DISTANCE, historical_chain_intact=False, wall_s=time.perf_counter()-started)
    atomic(out / 'report.json', dict(manifest=manifest, summary=summary, rows=rows, output_sha256=outputs,
        scope='1000mm exact saved prefixes with fresh endpoint/full-source and interval tool gates; pending identities get fresh finite 5degree ray searches. '
              'Independent cold verification required for every positive; historical missing evidence remains disclosed. '
              'All21 identities retained; negatives are not infeasibility proofs; sampled collision validation.'))
    print('ONE_METER_SUMMARY', summary, flush=True)


if __name__ == '__main__': main()
