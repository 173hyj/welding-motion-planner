"""Expand sealed 200cm posture/direction experiments to the remaining inventory.

Uses the unchanged per-point experiment from probe_positioner_retreat. Keeps
the original assigned robot, source 5deg geometry candidates and angle. Every
positive still requires a complete 2000mm trace. Cold audit is a separate stage.
"""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import time
from pathlib import Path

import probe_analytic_retreat as experiment
from run_geometry import atomic, digest, initialize_worker

ROOT = experiment.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError('One or two workers')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Preserve evidence: use a fresh output')
    experiment.retreat.assert_frozen_geometry_cache()
    prior_path = ROOT/'build/welding_retreat_200cm_full_directions_v7/report.json'
    original_path = ROOT/'build/welding_variable_open_v1/report.json'
    static_path = ROOT/'build/welding_positioner_static_v3/report.json'
    prior = json.loads(prior_path.read_text())
    original = json.loads(original_path.read_text())
    if not prior['summary']['complete'] or prior['summary']['required_straight_distance_mm'] != 2000.:
        raise RuntimeError('Completed literal 200cm baseline required')
    sources = {**prior['manifest']['source_sha256'], **prior['manifest']['code_sha256']}
    for p in (prior_path, original_path, static_path):
        sources[str(p.relative_to(ROOT))] = digest(p)
    for name, value in prior['output_sha256'].items():
        p = prior_path.parent/name
        if digest(p) != value:
            raise RuntimeError('Baseline output changed')
        sources[str(p.relative_to(ROOT))] = value
    codes = {str(p.relative_to(ROOT)): digest(p) for p in
             (Path(__file__), HERE/'probe_analytic_retreat.py', HERE/'run_retreat_v2.py',
              HERE/'analytic_ik.py', HERE/'analytic_trace_motion.py')}
    for name, value in {**sources, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Input changed: '+name)
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        distance_mm=2000., posture_roll_step_deg=5, pose_budget=10000,
        direction_budget=10000, workers=args.workers, original_assigned_robot=True,
        original_positioner_angle=True, source_analytic_ik=True, no_qp=True))
    atomic(out/'manifest.json', manifest)
    rows = copy.deepcopy(prior['rows'])
    for i, row in enumerate(rows):
        if row['best']:
            src = prior_path.parent/row['best']['path_file']
            name = f'paths/reused_retreat_{i:02d}.json'
            atomic(out/name, json.loads(src.read_text()))
            row['best'].update(path_file=name, path_sha256=digest(out/name))
    already_searched = set()
    jobs = [row for row in original['rows'] if row['weld_id'] not in already_searched
            and not next(r for r in rows if r['weld_id'] == row['weld_id'])['best']]
    started = time.perf_counter()
    experiment.ENGINE = experiment.retreat.VariableToolWorld(json.loads(
        (ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
        fs = [pool.submit(experiment.point_job, row['weld_id'], out, 10000, 10000,
                          row['initial']['positioner_deg']) for row in jobs]
        for f in concurrent.futures.as_completed(fs):
            result = f.result()
            records.append(result)
            if result['best']:
                rows = [result if r['weld_id'] == result['weld_id'] else r for r in rows]
            atomic(out/'progress.json', dict(finished=len(records), requested=len(jobs),
                  passed=sum(bool(r['best']) for r in rows), elapsed_s=time.perf_counter()-started))
    for name, value in {**sources, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Input changed during run: '+name)
    outputs = {str(p.relative_to(out)): digest(p) for p in out.glob('OP10L*.json')}
    outputs.update({str(p.relative_to(out)): digest(p) for p in (out/'paths').glob('*.json')})
    atomic(out/'report.json', dict(manifest=manifest, summary=dict(complete=True,
        points=len(rows), passed=sum(bool(r['best']) for r in rows),
        newly_searched_points=len(jobs), required_straight_distance_mm=2000.,
        wall_s=time.perf_counter()-started), rows=rows, reselection_records=records,
        output_sha256=outputs, scope='Finite complete original direction bank and source 5deg allowed roll inventory, assigned robot and original angle retained. Analytic source shoulder/elbow/wrist branches; singular families retain one reference representative. Failure is not an infeasibility proof. All positive complete strokes require independent cold audit.'))
    print('INVENTORY_200CM', len(jobs), sum(bool(r['best']) for r in rows), flush=True)


if __name__ == '__main__':
    main()
