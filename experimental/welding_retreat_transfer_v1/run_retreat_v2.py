"""Fixed-distance, fixed-orientation straight retreat from archived weld poses."""
import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/'welding_variable_open_v1'),
               str(HERE.parent/'welding_waypoint_qp_v1'),
               str(HERE.parent/'welding_waypoint_qp_v2'),
               str(HERE.parent/'welding_direct_rrt_v1')]
from variable_world import VariableToolWorld
from variable_core import prefix_available
from run_geometry import atomic, digest, initialize_worker
from run_initial import assert_frozen_geometry_cache
from trace_motion import lift_path
from run_qp import opening_frames
from run_direct import geometry_prefix

_ENGINE = None


def trial_directions(engine, row, target, driver, parent_dir):
    directions = []
    if row['best'] and row['best']['driver_mm'] == driver:
        archived = json.loads((parent_dir/row['best']['path_file']).read_text())
        directions.append(np.asarray(archived['direction_world']))
    opened = engine.state(row['initial']['robot_id'], target, driver, row['_moved'],
                          engine.environment.relative_env, 'opening')
    context = engine.release_context(row['initial']['robot_id'], target, row['_moved'],
                                     dict(open_state=opened), driver_mm=driver)
    normal = np.asarray(context['preferred_outward_normal'])
    bank = [target[:3, :3]@ray for ray in engine.rays]
    bank.sort(key=lambda ray: -float(ray@normal))
    for direction in bank:
        if any(float(np.asarray(p['outward_normal'])@direction) <= 1e-6 for p in context['supports']):
            continue
        if any(np.linalg.norm(direction-other) < 1e-8 for other in directions):
            continue
        directions.append(direction)
    # Greedy farthest-point ordering spreads the retained trials across the
    # unchanged source direction bank rather than clustering near one normal.
    remaining = np.asarray(directions)
    if len(remaining):
        order = [0]
        chosen = np.zeros(len(remaining), dtype=bool); chosen[0] = True
        nearest = 1. - remaining@remaining[0]
        while len(order) < len(remaining):
            scores = np.where(chosen, -np.inf, nearest)
            index = int(np.argmax(scores)); order.append(index); chosen[index] = True
            nearest = np.minimum(nearest, 1.-remaining@remaining[index])
        directions = [remaining[i] for i in order]
    return directions


def point_job(index, row, weld, parent_dir, out, distance, arm_budget):
    engine = _ENGINE
    started = time.perf_counter()
    selected = row['initial']
    rid = selected['robot_id']
    engine.set_angle(selected['positioner_deg'])
    moved, _ = engine.target(weld, selected)
    row = dict(row, _moved=moved)
    start = engine.world.models[rid].tcp(np.asarray(selected['q_rad']))
    amplitudes = [row['best']['amplitude_mm']] if row['best'] else []
    amplitudes += [a for a in (30., 60., 80., 185.) if a not in amplitudes]
    trials, best = [], None
    geometry_reasons = {}
    geometry_queries = 0
    for amplitude in amplitudes:
        if not prefix_available(row['opening_profile'], amplitude):
            continue
        driver = -amplitude
        directions = trial_directions(engine, row, start, driver, parent_dir)
        arm_trials = 0
        for direction in directions:
            # Fast reject only; positives require local contact release and interval certification.
            clear = True
            for s in np.arange(5., distance+1e-7, 10.):
                pose = start.copy()
                pose[:3, 3] += s*direction
                geometry_queries += 1
                state = engine.state(rid, pose, driver, moved, engine.world.static, 'separated')
                if state['blocking_pairs']:
                    clear = False
                    geometry_reasons['ray_collision'] = geometry_reasons.get('ray_collision', 0)+1
                    break
            if not clear:
                continue
            geom = geometry_prefix(engine, rid, start, moved, driver, direction, distance)
            if not geom['passed']:
                reason = geom.get('reason', 'interval_not_certified')
                geometry_reasons[reason] = geometry_reasons.get(reason, 0)+1
                continue
            ray = dict(direction_world=direction.tolist(), direction_frame='current_world', length_mm=distance)
            path = lift_path(engine, weld, selected, selected, ray, driver_mm=driver,
                             opening_frames=opening_frames(row, start, driver))
            trial = dict(amplitude_mm=amplitude, direction_world=direction.tolist(), geometry=geom,
                         arm_passed=path['passed'], reason=path.get('reason', 'passed'),
                         failed_travel_mm=path.get('failure_travel_mm'),
                         last_accepted_travel_mm=path.get('last_accepted_travel_mm'),
                         failure=path.get('failure'))
            trials.append(trial)
            arm_trials += 1
            if path['passed']:
                name = f'paths/retreat_{index:02d}.json'
                path.update(weld_id=row['weld_id'], fixed_distance_mm=distance,
                            no_qp=True, geometry_certificate=geom)
                atomic(out/name, path)
                best = dict(driver_mm=driver, amplitude_mm=amplitude,
                            path_file=name, path_sha256=digest(out/name), length_mm=distance)
                break
            if arm_trials >= arm_budget:
                break
        if best:
            break
        print('RETREAT_AMPLITUDE', row['weld_id'], amplitude, 'ARM_TRIALS', arm_trials, flush=True)
    result = dict(weld_id=row['weld_id'], initial=selected, best=best, passed=best is not None,
                  required_straight_distance_mm=distance, trials=trials,
                  geometry_rejection_counts=geometry_reasons, geometry_queries=geometry_queries,
                  finite_arm_candidates_per_amplitude=arm_budget,
                  elapsed_s=time.perf_counter()-started)
    atomic(out/f'point_{index:02d}.json', result)
    print('RETREAT_POINT', row['weld_id'], 'PASS', result['passed'],
          'SECONDS', round(result['elapsed_s'], 2), flush=True)
    return index, result


def main():
    global _ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance-mm', type=float, required=True)
    parser.add_argument('--unit-confirmation', required=True,
                        help='Record the user-confirmed distance text; never silently infer cm vs mm')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--arm-candidates', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weld-id', action='append')
    parser.add_argument('--reuse-result', type=Path)
    args = parser.parse_args()
    if args.distance_mm <= 0 or not np.isfinite(args.distance_mm) or not 1 <= args.workers <= 4 or args.arm_candidates < 1:
        raise ValueError('Positive distance and candidate budget; 1–4 workers required')
    assert_frozen_geometry_cache()
    parent_dir = ROOT/'build/welding_variable_open_v1'
    parent = json.loads((parent_dir/'report.json').read_text())
    baseline = dict(parent['manifest']['source_sha256'])
    baseline['build/welding_variable_open_v1/report.json'] = digest(parent_dir/'report.json')
    baseline['source/psz/EngData/engData_ver2.xml'] = digest(ROOT/'source/psz/EngData/engData_ver2.xml')
    for row in parent['rows']:
        if row['best']:
            name = row['best']['path_file']
            expected = row['best']['path_sha256']
            if digest(parent_dir/name) != expected or parent['output_sha256'][name] != expected:
                raise RuntimeError('Archived direction source changed')
            baseline[str((parent_dir/name).relative_to(ROOT))] = expected
    prior = None
    if args.reuse_result:
        reuse = args.reuse_result.resolve()
        prior = json.loads((reuse/'report.json').read_text())
        if not prior['summary']['complete'] or prior['summary']['required_straight_distance_mm'] != args.distance_mm:
            raise RuntimeError('Only completed same-distance prior runs may be reused')
        if any(r['best'] for r in prior['rows']):
            raise RuntimeError('This refinement runner only reuses sealed negative rows')
        baseline[str((reuse/'report.json').relative_to(ROOT))] = digest(reuse/'report.json')
        for name, value in prior['output_sha256'].items():
            if digest(reuse/name) != value:
                raise RuntimeError('Prior run output changed')
            baseline[str((reuse/name).relative_to(ROOT))] = value
    if args.weld_id and not prior:
        raise ValueError('Targeted refinement must reuse the complete prior inventory')
    known_ids = {r['weld_id'] for r in parent['rows']}
    if args.weld_id and not set(args.weld_id) <= known_ids:
        raise ValueError('Unknown target weld identity')

    codes = dict(parent['manifest']['code_sha256'])
    direct = json.loads((ROOT/'build/welding_direct_rrt_v1/manifest.json').read_text())
    codes.update(direct['code_sha256'])
    codes[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    manifest = dict(source_sha256=baseline, code_sha256=codes, no_qp=True,
                    settings=dict(distance_mm=args.distance_mm, unit_confirmation=args.unit_confirmation,
                                  arm_candidates_per_amplitude=args.arm_candidates,
                                  direction_order='greedy_farthest_point_on_original_bank',
                                  target_weld_ids=args.weld_id, reused_result=str(args.reuse_result) if args.reuse_result else None))
    for name, value in {**baseline, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Frozen source/code changed: '+name)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mp = out/'manifest.json'
    if mp.exists():
        if not args.resume or json.loads(mp.read_text()) != manifest:
            raise RuntimeError('Use new output or matching --resume')
        if (out/'report.json').exists():
            done = json.loads((out/'report.json').read_text())
            if done['summary']['complete']:
                for name, value in done['output_sha256'].items():
                    if digest(out/name) != value:
                        raise RuntimeError('Saved evidence changed')
                print('RETREAT_READ_ONLY', done['summary'])
                return
    else:
        atomic(mp, manifest)
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    welds = {w['id']: w for w in scene['welds']}
    rows, jobs = {}, []
    for index, row in enumerate(parent['rows']):
        p = out/f'point_{index:02d}.json'
        if args.weld_id and row['weld_id'] not in args.weld_id:
            previous = next(r for r in prior['rows'] if r['weld_id']==row['weld_id'])
            if previous['initial'] != row['initial']:
                raise RuntimeError('Reused initial pose differs')
            rows[index] = dict(previous, reused_negative=True,
                               reused_report_sha256=baseline[str((reuse/'report.json').relative_to(ROOT))])
            atomic(p, rows[index])
        elif args.resume and p.exists():
            rows[index] = json.loads(p.read_text())
        else:
            jobs.append((index, row))
    started = time.perf_counter()
    load = 0.
    if jobs:
        t = time.perf_counter()
        _ENGINE = VariableToolWorld(scene)
        load = time.perf_counter()-t
        print('RETREAT_MODEL_LOADED', round(load, 2), flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
            futures = [pool.submit(point_job, i, r, welds[r['weld_id']], parent_dir, out,
                                   args.distance_mm, args.arm_candidates) for i, r in jobs]
            for f in concurrent.futures.as_completed(futures):
                i, rows[i] = f.result()
                atomic(out/'progress.json', dict(finished=len(rows), total=len(parent['rows']),
                      passed=sum(r['passed'] for r in rows.values()), elapsed_s=time.perf_counter()-started))
    for name, value in {**baseline, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Inputs changed during experiment')
    output = {str(p.relative_to(out)): digest(p) for p in list(out.glob('point_*.json'))+list((out/'paths').glob('*.json'))}
    report = dict(manifest=manifest, summary=dict(complete=len(rows)==len(parent['rows']),
                  points=len(rows), newly_computed_points=len(jobs), reused_negative_points=len(rows)-len(jobs), passed=sum(r['passed'] for r in rows.values()),
                  required_straight_distance_mm=args.distance_mm, model_load_s=load,
                  wall_s=time.perf_counter()-started), rows=[rows[i] for i in sorted(rows)],
                  output_sha256=output,
                  scope='Original initial poses and no QP; fixed-orientation straight retreat. Finite direction/IK search; negatives do not prove infeasibility. Full-system collision is sampled.')
    atomic(out/'report.json', report)
    print('RETREAT_SUMMARY', report['summary'], flush=True)


if __name__ == '__main__':
    main()
