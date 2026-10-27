"""Reselect process poses/rays for continuous positive-q6 retreats, without wrap."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import probe_cell_inventory_v20 as search
from positive_wrist_model_v31 import PositiveWristModel
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retreat-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.retreat_out.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    parent = json.loads((source/'report.json').read_text())
    if not parent['summary']['complete'] or parent['summary']['required_straight_distance_mm'] != 2000.:
        raise RuntimeError('Completed 2000mm inventory required')
    if len(parent['rows']) != 21 or len({r['weld_id'] for r in parent['rows']}) != 21:
        raise RuntimeError('Full 21-weld inventory required')
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
               str((source/'report.json').relative_to(ROOT)): digest(source/'report.json')}
    for name, sha in parent['output_sha256'].items():
        if digest(source/name) != sha:
            raise RuntimeError('Prior output changed')
        sources[str((source/name).relative_to(ROOT))] = sha
    codes = {str(p.relative_to(ROOT)): digest(p) for p in
             [Path(__file__), HERE/'positive_wrist_model_v31.py', HERE/'probe_cell_inventory_v20.py',
              HERE/'probe_neighbor_rays_v17.py', HERE/'analytic_ik_v2.py', HERE/'analytic_trace_motion_v2.py']}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Input seal changed '+name)
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        distance_mm=2000., posture_roll_step_deg=5., source_wrist_search_low_rad=1e-6,
        both_robot_assignment=True, retain_old_positive_on_finite_failure=True,
        preserve_successful_positioner_angle=True, full_direction_bank=True,
        no_midpath_wrap=True, cold_uses_full_source_limits=True))
    atomic(out/'manifest.json', manifest)
    rows = copy.deepcopy(parent['rows'])
    jobs = []
    for i, row in enumerate(rows):
        if not row['best']:
            continue
        old_path = source/row['best']['path_file']
        trajectory = json.loads(old_path.read_text())
        if min(f['q_rad'][5] for f in trajectory['frames']) < 0.:
            jobs.append((i, row['weld_id'], trajectory['positioner_deg']))
        name = f'paths/reused_{i:02d}.json'
        (out/name).parent.mkdir(parents=True, exist_ok=True)
        (out/name).write_bytes(old_path.read_bytes())
        row['best'].update(path_file=name, path_sha256=digest(out/name))
    initialize_worker()
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    search.retreat.assert_frozen_geometry_cache()
    search.ENGINE = search.retreat.VariableToolWorld(scene)
    for rid, model in list(search.ENGINE.world.models.items()):
        search.ENGINE.world.models[rid] = PositiveWristModel(model)
    started = time.perf_counter()
    results = []
    for index, wid, angle in jobs:
        print('POSITIVE_WRIST_POINT_START', wid, angle, flush=True)
        result = search.point_job(wid, out, 10000, 10000, angle)
        if result['best']:
            trajectory = json.loads((out/result['best']['path_file']).read_text())
            if not all(q['q_rad'][5] >= 1e-6 for q in trajectory['frames']):
                raise RuntimeError('New path left continuous positive wrist domain')
            result['wrist_search_domain_reselected'] = True
            rows[index] = result
        results.append(dict(weld_id=wid, positioner_deg=angle, new_positive_found=bool(result['best']),
                            report_file=f'{wid}_angle_{angle:g}.json', elapsed_s=result.get('elapsed_s')))
        atomic(out/'progress.json', dict(finished=len(results), requested=len(jobs),
               positive_wrist_reselected=sum(r['new_positive_found'] for r in results),
               retained_complete_retreats=sum(bool(r['best']) for r in rows), elapsed_s=time.perf_counter()-started))
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Input changed during search '+name)
    outputs = {str(p.relative_to(out)): digest(p) for p in out.glob('OP10L*.json')}
    outputs.update({str(p.relative_to(out)): digest(p) for p in (out/'paths').glob('*.json')})
    summary = dict(complete=True, points=21, passed=sum(bool(r['best']) for r in rows),
                   required_straight_distance_mm=2000., requested_points=len(jobs),
                   positive_wrist_reselected=sum(r['new_positive_found'] for r in results),
                   wall_s=time.perf_counter()-started)
    atomic(out/'report.json', dict(manifest=manifest, summary=summary, rows=rows, search_records=results,
           output_sha256=outputs, scope='Finite process-pose/ray re-selection in positive-q6 search domain; full source coupling/geometry/FK checks. '
           'Original cold-positive retained on failed finite reselection. New positive requires independent full-source cold and fresh endpoint transfers; no wrap.'))
    print('POSITIVE_WRIST_SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
