"""Fresh 1000 mm source-validated actions from exact analytic catalog states.

Run on Linux: python -m experimental.welding_angle_batches_v1.actions
--catalog /path/catalog.sqlite --angle 90 --out build/new_actions --workers 12.
This stage certifies local actions only; transfers remain a separate stage.
"""
import argparse
import concurrent.futures
from contextlib import closing
import hashlib
import json
import math
import multiprocessing
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = HERE.parent/'welding_retreat_transfer_v1'
ENGINE = None
PARKS = None
SETTINGS = None
OUT = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_candidates(path, angle):
    rows = {}
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        for weld, robot, roll, sign, branch, payload in db.execute(
                'SELECT weld,robot,roll,sign,branch,payload FROM candidates WHERE angle=? ORDER BY weld,robot,roll,sign,branch', (angle,)):
            state = json.loads(payload)
            if state.get('origin') not in ('analytic', 'analytic_and_numeric'):
                continue
            rows.setdefault(weld, []).append(dict(state, robot_id=robot, roll_deg=roll,
                                                 axis_sign=sign, catalog_branch=branch))
    return rows


def required_weld_ids(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        return sorted({weld for weld,payload in db.execute('SELECT weld,payload FROM candidates')
                       if json.loads(payload).get('origin') in ('analytic','analytic_and_numeric')})


def rank_candidates(rows, previous=None, bounds=None):
    """Round-robin pose charts; rank windings by raw actuator distance/margin."""
    previous = previous or {}
    def score(row):
        same = row['robot_id'] == previous.get('robot_id')
        qdistance = (sum((a-b)**2 for a, b in zip(row['q_rad'], previous['q_rad']))
                     if same and previous.get('q_rad') else 0.)
        delta = abs(row['roll_deg']-previous.get('roll_deg', 0.)) % 360.
        roll = min(delta, 360.-delta)
        margin = 0.
        if bounds and row['robot_id'] in bounds:
            low, high = bounds[row['robot_id']]
            margin = min(min(q-lo, hi-q) for q, lo, hi in zip(row['q_rad'], low, high))
        return (not same if previous else False, roll, qdistance, -margin,
                row['robot_id'], row['axis_sign'], row.get('catalog_branch', 0))
    groups = {}
    for row in sorted(rows, key=score):
        groups.setdefault((row['robot_id'], row['roll_deg'], row['axis_sign']), []).append(row)
    result = []
    while groups:
        for key in list(groups):
            result.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return result


def action_summary(rows, requested, angle):
    passed = sum(bool(row.get('passed')) for row in rows)
    return dict(complete=len(rows)==requested, points=len(rows), requested_points=requested,
                passed=passed, all21_actions_passed=requested==21 and passed==21,
                all_requested_actions_passed=passed==requested, angle_deg=angle,
                required_straight_distance_mm=1000., transfers_planned=False,
                independent_cold_performed=False)


def amplitude_order(amplitudes, previous_driver=None):
    amplitude = abs(float(previous_driver)) if previous_driver is not None else None
    if amplitude is None or not math.isfinite(amplitude) or not 0. < amplitude <= 185.:
        return list(amplitudes)
    return [amplitude]+[a for a in amplitudes if a != amplitude]


def evaluate_point(index, weld, candidates, previous):
    from joint_search_v153 import try_configuration
    from integrated_retreat_v117 import check_path_contract
    begun = time.perf_counter()
    deadline = begun+SETTINGS['point_seconds']
    angle = SETTINGS['angle_deg']
    ENGINE.set_angle(angle)
    row = dict(weld_id=weld['id'], passed=False, initial=None, best=None,
               candidate_tries=0, static_analytic_candidates=len(candidates),
               required_straight_distance_mm=1000., attempts=[])
    layout = ENGINE.environment.layout_collisions(angle)
    bounds = {rid:[list(part) for part in model.bounds] for rid,model in ENGINE.world.models.items()}
    settings = dict(SETTINGS, deadline=deadline,
                    amplitudes=amplitude_order(SETTINGS['amplitudes'], (previous or {}).get('driver_mm')))
    if layout['blocking_pairs']:
        row['status'] = 'source_layout_blocked'
    else:
        ordered = rank_candidates(candidates, previous, bounds)
        for state in ordered:
            if time.perf_counter() >= deadline:
                break
            candidate = {key:state[key] for key in ('robot_id','roll_deg','axis_sign')}
            result = try_configuration(ENGINE, weld, candidate, state['q_rad'], PARKS,
                                       settings, {})
            row['candidate_tries'] += 1
            attempt = dict(candidate=candidate, catalog_branch=state['catalog_branch'],
                           q_rad=state['q_rad'], passed=result['passed'],
                           closed_passed=result['closed_passed'], reason=result.get('reason'),
                           ray_trials=len(result['attempts']))
            row['attempts'].append(attempt)
            if result['passed']:
                path = result['path']
                selected = result['selected']
                if selected['q_rad'] != state['q_rad']:
                    raise ValueError('Catalog actuator coordinates changed')
                check_path_contract(path, selected, 1000.)
                name = f'paths/retreat_{index:02d}.json'
                atomic(OUT/name, path)
                checksum = digest(OUT/name)
                row.update(passed=True, initial=selected,
                           best=dict(path_file=name, path_sha256=checksum, length_mm=1000.,
                                     driver_mm=path['driver_mm'], amplitude_mm=abs(path['driver_mm'])),
                           online_acceptance=dict(passed=True, path_sha256=checksum,
                                                  validation=path['online_validation']))
                break
        row['status'] = ('complete_action_accepted' if row['passed'] else
                         'finite_budget_expired' if time.perf_counter()>=deadline else
                         'no_complete_action_in_catalog_search')
    row['elapsed_s'] = time.perf_counter()-begun
    atomic(OUT/f'point_{index:02d}.json', row)
    print('ANGLE_ACTION', angle, weld['id'], row['status'], row['candidate_tries'], round(row['elapsed_s'],2), flush=True)
    return index, row


def main():
    global ENGINE, PARKS, SETTINGS, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--angle', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--point-seconds', type=float, default=120.)
    parser.add_argument('--directions', type=int, default=32)
    parser.add_argument('--previous-retreat', type=Path)
    args = parser.parse_args()
    if args.workers<1 or args.directions<1 or args.directions>2522 or not math.isfinite(args.angle) or not math.isfinite(args.point_seconds) or args.point_seconds<=0:
        parser.error('Invalid worker/direction/time/angle settings')
    if 'fork' not in multiprocessing.get_all_start_methods():
        parser.error('Linux fork is required: initialize one parent world and inherit workers')
    OUT = args.out.resolve()
    if OUT.exists():
        raise FileExistsError('Fresh output directory required')
    begun = time.perf_counter()
    sys.path[:0] = [str(ROOT), str(LEGACY)]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from signed_wrist_model_v40 import branch_parks
    from integrated_retreat_v117 import online_retreat_ids
    catalog = args.catalog.resolve()
    with closing(sqlite3.connect(catalog.as_uri()+'?mode=ro', uri=True)) as db:
        identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    # Catalog source hashes describe source data; historical catalog code is
    # preserved as provenance and current action code is sealed separately.
    source_hashes = dict(identity.get('source_sha256', {}))
    for name, expected in source_hashes.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Catalog source changed: '+name)
    candidates = load_candidates(catalog, args.angle)
    scene_path = ROOT/'build/scene_placement_corrected_v1.json'
    scene = json.loads(scene_path.read_text())
    required = required_weld_ids(catalog)
    welds = [weld for weld in scene['welds'] if weld['id'] in required]
    if set(required) != {weld['id'] for weld in welds}:
        raise ValueError('Catalog required weld identity missing from scene')
    previous = {}
    if args.previous_retreat:
        report = json.loads((args.previous_retreat/'report.json').read_text())
        previous = {row['weld_id']:row['initial'] for row in report['rows'] if row.get('best')}
        for row in report['rows']:
            if row.get('best'):
                previous[row['weld_id']] = dict(previous[row['weld_id']], driver_mm=row['best']['driver_mm'])
    initialize_worker()
    ENGINE = load_raw_world(scene, SIGNS)
    for rid, model in list(ENGINE.world.models.items()):
        ENGINE.world.models[rid] = getattr(model, 'source', model)
    PARKS = {rid:[q] for rid,q in branch_parks(ENGINE.world.models, SIGNS).items()}
    SETTINGS = dict(angle_deg=args.angle, point_seconds=args.point_seconds,
                    amplitudes=[10.,30.,60.,80.,120.,140.,180.], directions=args.directions)
    dependencies = list(LEGACY.glob('*.py'))
    for directory in ('welding_tool_exit_v1','welding_variable_open_v1','welding_direct_rrt_v1'):
        dependencies.extend((HERE.parent/directory).glob('*.py'))
    dependencies += [Path(__file__), ROOT/'scripts/audit_contact_release_v17.py']
    codes = {str(path.relative_to(ROOT)):digest(path) for path in dependencies}
    source_hashes[str(scene_path.relative_to(ROOT))] = digest(scene_path)
    manifest = dict(source_sha256=source_hashes, code_sha256=codes,
                    catalog_sha256=digest(catalog), catalog_path=str(catalog),
                    catalog_identity=identity, settings=dict(SETTINGS, workers=args.workers),
                    physical_source_limits=True, wrist_sign_domain_restriction=False,
                    inactive_parks=PARKS, supplied_catalog_q_only=True,
                    required_weld_ids=required)
    if args.previous_retreat:
        previous_report = (args.previous_retreat/'report.json').resolve()
        manifest['previous_report'] = dict(path=str(previous_report), sha256=digest(previous_report))
    OUT.mkdir(parents=True)
    atomic(OUT/'manifest.json', manifest)
    rows = [None]*len(welds)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
             mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
        jobs = [pool.submit(evaluate_point, i, weld, candidates.get(weld['id'], []), previous.get(weld['id']))
                for i,weld in enumerate(welds)]
        for future in concurrent.futures.as_completed(jobs):
            index,row = future.result()
            rows[index] = row
            atomic(OUT/'progress.json', dict(finished=sum(r is not None for r in rows),
                                           passed=sum(bool(r and r['passed']) for r in rows)))
    for name, checksum in {**source_hashes, **codes}.items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Input changed during run: '+name)
    if digest(catalog) != manifest['catalog_sha256']:
        raise ValueError('Catalog changed during run')
    outputs = {str(path.relative_to(OUT)):digest(path) for path in OUT.rglob('*.json')
               if path.name not in ('report.json','manifest.json','progress.json')}
    summary = action_summary(rows, len(welds), args.angle)
    summary['wall_s'] = time.perf_counter()-begun
    report = dict(manifest=manifest, summary=summary, rows=rows, output_sha256=outputs,
                  scope='Local exact-catalog welding/open/1000mm retreat actions only. Ingress reverses accepted local paths. Transfers and full cycle are not yet planned. Finite failures do not prove infeasibility. Sampled source online acceptance, not independent cold certification.')
    online_retreat_ids(report, OUT)
    atomic(OUT/'report.json', report)
    print(json.dumps(summary), flush=True)
    if not summary['all_requested_actions_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
