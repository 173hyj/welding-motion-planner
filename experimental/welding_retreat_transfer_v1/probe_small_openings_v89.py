"""Smaller nonzero source-valid opening domain on remaining4, unchanged source acceptance."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from signed_wrist_model_v40 import SignedWristModel, path_compatibility
from compact_inventory_v82 import compact_rows, link_candidate
from probe_mixed_wrist_v65 import replace_with_positive
from probe_near_retreat_v72 import release_neighbors
from probe_neighbor_rays_v17 import kinematic_rejection
from seed_selector_v79 import select_seeds
from seed_selector_v79 import rank_seeds

def opening_amplitudes():
    return (1., 2., 3., 4.)
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent
HISTORY = ['mixed_wrist_v44', 'refined_angles_v69', 'local_external_v74',
           'local_external_v76', 'local_external_v77', 'deferred_external_v87']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 4:
        raise ValueError('Finite seed budget required')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    baseline = ROOT / 'build/welding_retreat_200cm_deferred_external_v87'
    parent = json.loads((baseline / 'report.json').read_text())
    if (not parent['summary']['complete'] or len(parent['rows']) != 21 or
            len({r['weld_id'] for r in parent['rows']}) != 21 or
            parent['summary']['required_straight_distance_mm'] != 2000.):
        raise ValueError('Full21 completed literal2m inventory required')
    rows = compact_rows(parent['rows'], baseline / 'report.json', ROOT)
    pending = [r['weld_id'] for r in rows if not r['best']]
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
               str((baseline / 'report.json').relative_to(ROOT)): digest(baseline / 'report.json')}
    for n, sha in parent['output_sha256'].items():
        sources[str((baseline / n).relative_to(ROOT))] = sha
    choices = {wid: [] for wid in pending}
    for tag in HISTORY:
        folder = ROOT / ('build/welding_retreat_200cm_' + tag)
        for wid in pending:
            for p in sorted(folder.glob(wid + '_angle_*.json')):
                if p.name.endswith('_progress.json'):
                    continue
                r = json.loads(p.read_text())
                name = str(p.relative_to(ROOT))
                selected = select_seeds([(name, r)], wid, 10000)
                if selected:
                    choices[wid].extend(selected)
                    sources[name] = digest(p)
    choices = {wid: rank_seeds(items, args.seeds) for wid, items in choices.items()}
    if any(not items for items in choices.values()):
        raise ValueError('Every deferred identity needs an archived seed')
    codes = {str((HERE / n).relative_to(ROOT)): digest(HERE / n) for n in
             [Path(__file__).name, 'compact_inventory_v82.py', 'seed_selector_v79.py', 'seed_extension_v81.py', 'probe_near_retreat_v72.py',
              'probe_neighbor_rays_v17.py', 'branch_point_job_v64.py',
              'probe_mixed_wrist_v65.py', 'retained_candidates_v58.py',
              'analytic_ik_v2.py', 'analytic_trace_motion_v2.py', 'signed_wrist_model_v40.py']}
    for n, sha in {**sources, **codes}.items():
        if digest(ROOT / n) != sha:
            raise ValueError('Changed input ' + n)
    out.mkdir(parents=True)
    for i, row in enumerate(rows):
        if row.get('retained_existing_candidate'):
            row['retained_existing_candidate'] = link_candidate(row['weld_id'],
                row['retained_existing_candidate'], baseline, out, f'paths/retained_{i:02d}.json')
        if row['best']:
            name = f'paths/reused_{i:02d}.json'
            row['best'] = link_candidate(row['weld_id'], row['best'], baseline, out, name)
    signs = parent['manifest']['settings']['per_robot_wrist_sign']
    parks = parent['manifest']['settings']['per_robot_inactive_park']
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        distance_mm=2000., per_robot_wrist_sign=signs, per_robot_inactive_park=parks,
        seed_budget=args.seeds, opening_amplitudes_mm=opening_amplitudes(), prior_exact_seed_exclusion=False, fresh_small_nonzero_openings=True, excluded_batches=[], compact_search_metadata=True, immutable_path_hardlinks=True, direction_budget=2522, azimuth_step_deg=5,
        elevation_step_deg=5, posture_roll_step_deg=5, archived_negatives_reused=False,
        seeds=choices, fresh_cold_required=True, original_source_gates_unchanged=True,
        per_seed_exact_kinematic_cache=True, no_midpath_wrap=True))
    atomic(out / 'manifest.json', manifest)
    started = time.perf_counter()
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    engine = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    models = engine.world.models
    for rid, model in list(models.items()):
        models[rid] = SignedWristModel(model, signs[rid])
    records = []
    for wi, wid in enumerate(pending):
        original = next(r for r in rows if r['weld_id'] == wid)
        weld = next(w for w in engine.scene['welds'] if w['id'] == wid)
        point = dict(weld_id=wid, seeds=[], passed=False)
        records.append(point)
        replacement = None
        for si, seed in enumerate(choices[wid]):
            angle = seed['positioner_deg']
            case = dict(seed=seed, trials=[])
            point['seeds'].append(case)
            if engine.environment.layout_collisions(angle)['blocking_pairs']:
                case['reason'] = 'source_layout_blocked'
                continue
            delta = engine.set_angle(angle)
            moved, base = search.transform_weld(engine.scene, weld, delta)
            rid = seed['candidate']['robot_id']
            other = next(k for k in models if k != rid)
            q = np.asarray(seed['q_rad'])
            model = models[rid]
            start = model.tcp(q)
            target = search.roll_target(base, seed['candidate']['roll_deg'], seed['candidate']['axis_sign'])
            if (seed['candidate']['roll_deg'] % 5 != 0 or not model.limits_valid(q) or
                    not np.allclose(start, target, atol=1e-6, rtol=0)):
                raise ValueError('Seed must match exact legal source five-degree weld pose')
            selected = dict(original['initial'], **seed['candidate'], other_robot_id=other,
                positioner_deg=angle, q_rad=q.tolist(), other_q_rad=parks[other],
                actual_tcp_matrix=start.ravel().tolist(), target_tcp_matrix=target.ravel().tolist())
            caps = [pid for pid, (_, kind, _) in engine.parts[rid].items() if kind == 'electrode']
            profile = search.opening_profile(lambda d: search.full_probe(engine, rid, q, other,
                np.asarray(parks[other]), d, moved, 'closed' if d == 0. else 'opening'),
                caps, maximum_amplitude_mm=80.)
            case['opening_profile'] = profile
            kin_cache = {}
            print('DEFERRED_SPHERE_START', wid, si, angle, seed['candidate'], flush=True)
            for amplitude in opening_amplitudes():
                if not search.prefix_available(profile, amplitude):
                    continue
                driver = -amplitude
                state = engine.state(rid, start, driver, moved, engine.world.static, 'opening')
                context = engine.release_context(rid, start, moved, dict(open_state=state), driver_mm=driver)
                bank = release_neighbors(start, np.asarray(seed['reference']), context['supports'], 2522)
                for di, direction in enumerate(bank):
                    if di % 32 == 0:
                        atomic(out / 'progress.json', dict(finished=21-len(pending)+wi,
                            requested=21, compatible=sum(bool(r['best']) for r in rows),
                            weld_id=wid, seed_index=si, amplitude_mm=amplitude, direction_index=di,
                            elapsed_s=time.perf_counter()-started))
                    trial = dict(amplitude_mm=amplitude, direction_index=di,
                                 direction_world=direction.tolist())
                    case['trials'].append(trial)
                    end = start.copy()
                    end[:3, 3] += 2000. * direction
                    ends, _ = search.solve_seed_bank(model, end, [q], random_count=0)
                    if not ends:
                        trial['reason'] = 'finite_endpoint_ik_rejected'
                        continue
                    clear = True
                    for travel in np.arange(5., 2000. + 1e-7, 10.):
                        pose = start.copy()
                        pose[:3, 3] += travel * direction
                        if engine.state(rid, pose, driver, moved, engine.world.static, 'separated')['blocking_pairs']:
                            clear = False
                            break
                    if not clear:
                        trial['reason'] = 'coarse_tool_geometry_rejected'
                        continue
                    key = tuple(direction)
                    if key not in kin_cache:
                        kin_cache[key] = kinematic_rejection(model, start, q, direction)
                    kin = kin_cache[key]
                    if not kin['passed']:
                        trial.update(reason='finite_continuation_rejected',
                                     last_kinematic_travel_mm=kin.get('travel_mm'))
                        continue
                    end_state = search.full_probe(engine, rid, np.asarray(kin['q_end_rad']), other,
                        np.asarray(parks[other]), driver, moved, 'separated')
                    if not end_state['passed']:
                        trial.update(reason='endpoint_source_collision_rejected',
                                     pairs=end_state.get('remaining_pairs', []),
                                     tool_pairs=end_state.get('tool_blocking_pairs', []))
                        continue
                    geom = search.retreat.geometry_prefix(engine, rid, start, moved, driver, direction, 2000.)
                    if not geom['passed']:
                        trial['reason'] = 'geometry_prefix_rejected'
                        continue
                    path = search.lift_path(engine, weld, selected, selected,
                        dict(direction_world=direction.tolist(), direction_frame='current_world', length_mm=2000.),
                        driver_mm=driver, opening_frames=search.opening_frames(
                            dict(initial=selected, opening_profile=profile), start, driver))
                    trial.update(reason=path.get('reason', 'passed'), passed=path['passed'],
                        last_accepted_travel_mm=path.get('last_accepted_travel_mm'),
                        pairs=(path.get('failure') or {}).get('remaining_pairs', []))
                    print('DEFERRED_SPHERE_TRACE', wid, si, amplitude, di,
                          path['passed'], path.get('last_accepted_travel_mm'), flush=True)
                    if not path['passed']:
                        continue
                    if path_compatibility(path, signs, parks) != (True, True):
                        raise ValueError('Positive changed branch or inactive park')
                    path.update(weld_id=wid, fixed_distance_mm=2000., no_qp=True, geometry_certificate=geom)
                    name = f'paths/full_sphere_{wid}.json'
                    atomic(out / name, path)
                    replacement = dict(weld_id=wid, initial=selected, opening_profile=profile,
                        best=dict(path_file=name, path_sha256=digest(out / name), length_mm=2000.,
                            driver_mm=driver, amplitude_mm=amplitude), passed=True,
                        required_straight_distance_mm=2000., mixed_wrist_branch_compatible=True,
                        original_initial=original.get('original_initial', original['initial']))
                    point['passed'] = True
                    break
                if replacement:
                    break
            if replacement:
                break
        if replacement:
            rows = [replace_with_positive(r, replacement) if r['weld_id'] == wid else r for r in rows]
        atomic(out / (wid + '_sphere.json'), point)
        atomic(out / 'progress.json', dict(finished=21-len(pending)+wi+1, requested=21,
            compatible=sum(bool(r['best']) for r in rows), elapsed_s=time.perf_counter()-started))
        print('DEFERRED_SPHERE_POINT', wid, bool(replacement), flush=True)
    for n, sha in {**sources, **codes}.items():
        if digest(ROOT / n) != sha:
            raise ValueError('Input changed during search ' + n)
    outputs = {str(p.relative_to(out)): digest(p) for p in (out / 'paths').glob('*.json')}
    outputs.update({p.name: digest(p) for p in out.glob('OP10L*_sphere.json')})
    summary = dict(complete=True, points=21, requested_points=21,
        passed=sum(bool(r['best']) for r in rows), branch_compatible=sum(bool(r['best']) for r in rows),
        required_straight_distance_mm=2000.,
        retained_original_candidates=sum(bool(r.get('retained_existing_candidate')) for r in rows),
        new_positives=sum(p['passed'] for p in records), historical_chain_intact=False,
        wall_s=time.perf_counter()-started)
    atomic(out / 'report.json', dict(manifest=manifest, summary=summary, rows=rows,
        output_sha256=outputs, scope='Compact inventory retaining SHA-bound source reports. Fresh full5degree sphere at a finite archived pose bank; '
        'old negatives only rank seeds and are not reused. Original opening/geometry/full2m '
        'trace required. All21 identities and14originals retained. Positives still require own '
        'cold and actualfull15axis transfer/finalcold. Finite negatives do not prove infeasibility.'))
    print('DEFERRED_SPHERE_SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
