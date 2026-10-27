"""Existing-catalog joint alternatives at the frozen route angles and assignment.

Generation requires Linux fork and the authoritative source geometry backend.
Finite search failures are recorded and never interpreted as infeasibility.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).absolute().parents[2]
ENGINE = None
OUT = None
SETTINGS = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def path_initial(path):
    return dict(path['candidate'], robot_id=path['robot_id'], q_rad=path['q_start_rad'],
                other_robot_id=path['other_robot_id'], other_q_rad=path['other_q_rad'],
                positioner_deg=path['positioner_deg'])


def rank_existing(rows, baseline, previous=None, following=None):
    """Prefer the baseline chart, then existing alternative catalog charts."""
    unique = {}
    for row in rows:
        if row['robot_id'] != baseline['robot_id']:
            continue
        q = tuple(float(x) for x in row['q_rad'])
        if len(q) != 6 or not all(math.isfinite(x) for x in q):
            continue
        if max(abs(a-b) for a, b in zip(q, baseline['q_rad'])) < 1e-4:
            continue
        unique.setdefault(q, row)
    neighbors = [x for x in (previous, following, baseline['q_rad']) if x is not None]
    return sorted(unique.values(), key=lambda row: (
        (row['roll_deg'], row['axis_sign']) != (baseline['roll_deg'], baseline['axis_sign']),
        sum(sum((a-b)**2 for a, b in zip(row['q_rad'], q)) for q in neighbors),
        tuple(row['q_rad'])))


def assert_polyline(samples, line, tolerance=1e-10):
    """Require every resampled replay state on a saved action segment."""
    import numpy as np
    from scipy.spatial import cKDTree
    samples, line = np.asarray(samples), np.asarray(line)
    if not len(samples) or not len(line):
        raise ValueError('Missing baseline action phase')
    if len(line) == 1:
        error = np.max(abs(samples-line[0]))
    else:
        tree = cKDTree(line)
        _, near = tree.query(samples, k=min(4, len(line)))
        near = np.asarray(near).reshape(len(samples), -1)
        error = 0.
        for sample, indices in zip(samples, near):
            best = float('inf')
            for j in set(max(0, min(len(line)-2, int(i)+delta)) for i in indices for delta in (-1, 0)):
                a, b = line[j:j+2]; d = b-a
                fraction = np.clip((sample-a)@d/(d@d), 0., 1.) if d@d else 0.
                best = min(best, float(np.max(abs(sample-(a+fraction*d)))))
            error = max(error, best)
    if error > tolerance:
        raise ValueError(f'Baseline action differs from receipt: {error}')
    return float(error)


def verify_baseline_actions(tasks, data):
    import numpy as np
    from export_latest_route_v160 import action_sections
    proofs = []
    for task in tasks:
        arm = task['arm']; cols = list(range(6*arm, 6*arm+6))+[12, 13+arm]
        sections = dict(zip(('ingress', 'close', 'open', 'retreat'), action_sections(task['path'], task['states'])))
        proof = {}
        for phase, section in sections.items():
            mask = (data['tasks'][:, arm] == task['id']) & (data['phases'][:, arm] == phase)
            samples = data['states'][mask]
            line = np.asarray(section)
            if task['id'] <= 7:
                samples, line = samples[:, cols], line[:, cols]
            proof[phase] = dict(samples=len(samples), maximum_error=assert_polyline(samples, line))
        proofs.append(dict(task=task['id'], comparison='full15' if task['id'] >= 8 else 'active_arm_driver_angle', phases=proof))
    return proofs


def evaluate_task(task, rows, previous, following):
    import numpy as np
    from integrated_retreat_v117 import lift_path, check_path_contract
    from joint_search_v153 import try_configuration
    from run_direct import geometry_prefix
    baseline = path_initial(task['path']); rid = baseline['robot_id']
    ENGINE.set_angle(task['angle'])
    weld = next(w for w in ENGINE.scene['welds'] if w['id'] == task['weld_id'])
    started = time.perf_counter(); deadline = started + SETTINGS['point_seconds']
    actions, attempts = [], []
    ordered = rank_existing(rows, baseline, previous, following)
    for state in ordered[:SETTINGS['max_candidates']]:
        if time.perf_counter() >= deadline or len(actions) >= SETTINGS['extra_actions']:
            break
        if np.max(abs(np.asarray(state['q_rad'])-baseline['q_rad'])) < 1e-4:
            continue
        candidate = {k: state[k] for k in ('robot_id', 'roll_deg', 'axis_sign')}
        model = ENGINE.world.models[rid]; q = np.asarray(state['q_rad'])
        moved, target = ENGINE.target(weld, candidate); start = model.tcp(q)
        record = dict(catalog_state=state, passed=False)
        attempts.append(record)
        if not model.limits_valid(q) or np.max(abs(start-target)) > 1e-6:
            record['reason'] = 'invalid_source_weld_pose'; continue
        selected = dict(baseline, **candidate, q_rad=state['q_rad'],
                        target_tcp_matrix=target.ravel().tolist(), actual_tcp_matrix=start.ravel().tolist())
        ray = dict(direction_world=task['path']['direction_world'], direction_frame='current_world', length_mm=1000.)
        driver = task['path']['driver_mm']
        geometry = geometry_prefix(ENGINE, rid, start, moved, driver, np.asarray(ray['direction_world']), 1000.)
        path = lift_path(ENGINE, weld, candidate, selected, ray, driver_mm=driver) if geometry['passed'] else dict(passed=False, reason='geometry_prefix_failed')
        if path['passed']:
            path.update(weld_id=task['weld_id'], geometry_certificate=geometry)
        elif time.perf_counter() < deadline:
            parks = {baseline['other_robot_id']: [baseline['other_q_rad']]}
            settings = dict(angle_deg=task['angle'], deadline=deadline, directions=16,
                            amplitudes=list(dict.fromkeys([abs(driver), 10., 30., 60., 80., 120., 140., 180.])))
            result = try_configuration(ENGINE, weld, candidate, q, parks, settings, {})
            path = result.get('path') or path
            if result.get('passed'): selected = result['selected']
        record.update(passed=path['passed'], reason=path.get('reason'))
        if not path['passed']: continue
        if selected['q_rad'] != state['q_rad']: raise ValueError('Catalog coordinates changed')
        check_path_contract(path, selected, 1000.)
        name = f"paths/task_{task['id']:02d}_catalog_{len(actions):02d}.json"
        atomic(OUT/name, path)
        actions.append(dict(id=f"t{task['id']:02d}-catalog-{len(actions):02d}", source='catalog',
                            path_file=str((OUT/name).resolve()), path_sha256=digest(OUT/name),
                            catalog_state=state, initial=selected))
    row = dict(task=task['id'], actions=actions, attempts=attempts, elapsed_s=time.perf_counter()-started,
               status='requested_actions_found' if len(actions) == SETTINGS['extra_actions'] else 'finite_search_exhausted')
    atomic(OUT/f"task_{task['id']:02d}.json", row)
    print('CANDIDATE_POOL', task['id'], len(actions), row['status'], flush=True)
    return row


def build_pool(route_file, baseline_path, replacement_report, catalog, out, workers=12,
               point_seconds=180., max_candidates=10, extra_actions=3):
    global ENGINE, OUT, SETTINGS
    import numpy as np
    sys.path[:0] = [str(ROOT), str(ROOT/'experimental/welding_retreat_transfer_v1')]
    from experimental.dual_arm_parallel_v1.run_parallel import load_tasks, resolve_source
    from experimental.welding_angle_batches_v1.actions import load_candidates
    from replay_full21_core_v105 import local_states
    from integrated_retreat_v117 import check_path_contract
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    if workers < 1 or not math.isfinite(point_seconds) or point_seconds <= 0 or not 1 <= max_candidates <= 12 or not 0 <= extra_actions <= 3:
        raise ValueError('Invalid finite pool search settings')
    if 'fork' not in multiprocessing.get_all_start_methods(): raise RuntimeError('Linux fork required')
    OUT = Path(out).resolve()
    if OUT.exists(): raise FileExistsError('Fresh pool directory required')
    SETTINGS = dict(point_seconds=point_seconds, max_candidates=max_candidates, extra_actions=extra_actions)
    route, tasks, _, sources = load_tasks(Path(route_file))
    if [task['arm'] for task in tasks] != [1]*6+[0]*15:
        raise ValueError('Frozen current robot assignment changed')
    replacement_report = Path(replacement_report).resolve()
    replacement = json.loads(replacement_report.read_text())
    rows = {row['weld_id']: row for row in replacement['rows']}
    sources[str(replacement_report)] = digest(replacement_report)
    for task in tasks[2:4]:
        row = rows[task['weld_id']]
        file = replacement_report.parent/row['best']['path_file']
        if digest(file) != row['best']['path_sha256']: raise ValueError('Replacement action hash mismatch')
        path = json.loads(file.read_text()); check_path_contract(path, row['initial'], 1000.)
        if path['online_validation'] != row['online_acceptance']['validation']: raise ValueError('Replacement receipt mismatch')
        task.update(path=path, path_file=str(file.resolve()), states=np.asarray(local_states(path, route['robot_ids'])), angle=path['positioner_deg'])
        sources[str(file.resolve())] = digest(file)
    data = np.load(baseline_path, allow_pickle=False)
    proofs = verify_baseline_actions(tasks, data)
    layers = []
    for task in tasks:
        p = task['path']; initial = path_initial(p)
        check_path_contract(p, initial, 1000.)
        layers.append(dict(task=task['id'], weld_id=task['weld_id'], angle=task['angle'], arm=task['arm'],
                           robot_id=p['robot_id'], actions=[dict(id=f"t{task['id']:02d}-baseline", source='baseline',
                           path_file=task['path_file'], path_sha256=digest(task['path_file']), initial=initial)]))
    scene_file = ROOT/'build/scene_placement_corrected_v1.json'
    scene = json.loads(scene_file.read_text()); initialize_worker(); ENGINE = load_raw_world(scene, SIGNS)
    for rid, model in list(ENGINE.world.models.items()): ENGINE.world.models[rid] = getattr(model, 'source', model)
    catalog = Path(catalog).resolve()
    by_angle = {angle: load_candidates(catalog, angle) for angle in {t['angle'] for t in tasks[7:]}}
    sources.update({str(Path(route_file).resolve()): digest(route_file), str(Path(baseline_path).resolve()): digest(baseline_path),
                    str(catalog): digest(catalog), str(scene_file): digest(scene_file)})
    dependencies = [Path(__file__), ROOT/'experimental/welding_angle_batches_v1/actions.py']
    for namespace in ('welding_retreat_transfer_v1', 'welding_tool_exit_v1',
                      'welding_variable_open_v1', 'welding_direct_rrt_v1'):
        dependencies.extend((ROOT/'experimental'/namespace).glob('*.py'))
    codes = {str(p.resolve()): digest(p) for p in dependencies}
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(SETTINGS, workers=workers),
                    candidate_policy='fixed_angle_fixed_assignment_exact_existing_catalog_q', baseline_action_proofs=proofs)
    OUT.mkdir(parents=True); atomic(OUT/'manifest.json', manifest)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as executor:
        futures = [executor.submit(evaluate_task, task, by_angle[task['angle']].get(task['weld_id'], []),
                    tasks[i-1]['path']['q_start_rad'] if i > 7 else None,
                    tasks[i+1]['path']['q_start_rad'] if i+1 < len(tasks) else None) for i, task in enumerate(tasks) if i >= 7]
        for future in concurrent.futures.as_completed(futures):
            row = future.result(); results.append(row); layers[row['task']-1]['actions'].extend(row['actions'])
    for name, checksum in {**sources, **codes}.items():
        if digest(name) != checksum: raise ValueError('Input changed during generation: '+name)
    tail_arm = tasks[7]['arm']
    ingress_indices = np.flatnonzero((data['tasks'][:, tail_arm] == 8) & (data['phases'][:, tail_arm] == 'ingress'))
    if not len(ingress_indices) or ingress_indices[0] == 0: raise ValueError('Missing frozen prefix boundary')
    report = dict(layers=layers, robot_ids=route['robot_ids'], baseline_path=str(Path(baseline_path).resolve()),
                  baseline_sha256=digest(baseline_path), prefix_end_index=int(ingress_indices[0])-1, manifest=manifest,
                  rows=sorted(results, key=lambda r: r['task']), complete=True,
                  scope='Sampled source online local actions; finite failures do not prove infeasibility.')
    atomic(OUT/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('route', 'baseline', 'replacement-report', 'catalog', 'out'): parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--point-seconds', type=float, default=180.)
    parser.add_argument('--max-candidates', type=int, default=10)
    parser.add_argument('--extra-actions', type=int, default=3)
    args = parser.parse_args()
    build_pool(args.route, args.baseline, args.replacement_report, args.catalog, args.out,
               args.workers, args.point_seconds, args.max_candidates, args.extra_actions)


if __name__ == '__main__': main()
