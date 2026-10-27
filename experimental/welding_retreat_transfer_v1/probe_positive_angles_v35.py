"""Expand external-axis angles to join continuous positive-wrist retreat branches."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import probe_cell_inventory_v20 as search
from positive_wrist_model_v31 import PositiveWristModel
from positive_angles_v35 import angle_sequence, search_until_positive, read_certified_failures, normalize_result
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retreat-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--negative-evidence', type=Path, action='append', default=[])
    parser.add_argument('--angles', type=float, nargs='+',
                        default=[0., 30., -30., 60., -60., 90., -90., 120., -120., 150., -150., 180.])
    args = parser.parse_args()
    source, out = args.retreat_out.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    parent = json.loads((source/'report.json').read_text())
    if (not parent['summary']['complete']
            or parent['summary']['required_straight_distance_mm'] != 2000.
            or len(parent['rows']) != 21
            or len({r['weld_id'] for r in parent['rows']}) != 21):
        raise ValueError('Completed full 21-weld 2000mm inventory required')
    angle_sequence(0., args.angles)
    required = {str(p.relative_to(ROOT)): digest(p) for p in [
        HERE/'positive_wrist_model_v31.py', HERE/'probe_cell_inventory_v20.py',
        HERE/'probe_neighbor_rays_v17.py', HERE/'analytic_ik_v2.py', HERE/'analytic_trace_motion_v2.py']}
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
               str((source/'report.json').relative_to(ROOT)): digest(source/'report.json')}
    sources.update({str((source/n).relative_to(ROOT)): s for n,s in parent['output_sha256'].items()})
    failed = {}
    for directory in args.negative_evidence:
        directory = directory.resolve()
        failed.update(read_certified_failures(directory, ROOT, required))
        evidence = json.loads((directory/'report.json').read_text())
        sources.update(evidence['manifest']['source_sha256'])
        sources.update(evidence['manifest']['code_sha256'])
        sources[str((directory/'report.json').relative_to(ROOT))] = digest(directory/'report.json')
        sources.update({str((directory/n).relative_to(ROOT)): s for n,s in evidence['output_sha256'].items()})
    codes = {**required, **{str(p.relative_to(ROOT)): digest(p) for p in
                           [Path(__file__), HERE/'positive_angles_v35.py']}}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT/name) != sha:
            raise ValueError('Changed source '+name)
    out.mkdir(parents=True)
    settings = dict(distance_mm=2000., posture_roll_step_deg=5., source_wrist_search_low_rad=1e-6,
            both_robot_assignment=True, retain_old_positive_on_finite_failure=True,
            preserve_successful_positioner_angle=False, full_direction_bank=True,
            no_midpath_wrap=True, cold_uses_full_source_limits=True, angles=args.angles,
            identical_finite_negative_reuse=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=settings)
    atomic(out/'manifest.json', manifest)
    rows = copy.deepcopy(parent['rows'])
    jobs = []
    for i,row in enumerate(rows):
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
    search.retreat.assert_frozen_geometry_cache()
    search.ENGINE = search.retreat.VariableToolWorld(
        json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    for rid, model in list(search.ENGINE.world.models.items()):
        search.ENGINE.world.models[rid] = PositiveWristModel(model)
    started = time.perf_counter()
    records = []
    replaced = 0
    for ji, (index, wid, angle) in enumerate(jobs):
        sequence = angle_sequence(angle, args.angles)
        print('POSITIVE_ANGLE_POINT_START', wid, sequence, flush=True)
        def solve(external_angle):
            print('POSITIVE_ANGLE_ATTEMPT', wid, external_angle, flush=True)
            result = normalize_result(search.point_job(wid, out, 10000, 10000, external_angle),
                                      wid, external_angle)
            atomic(out/f'{wid}_angle_{external_angle:g}.json', result)
            if result['best']:
                trajectory = json.loads((out/result['best']['path_file']).read_text())
                if not all(f['q_rad'][5] >= 1e-6 for f in trajectory['frames']):
                    raise ValueError('New trajectory left positive wrist domain')
            return result
        best, history = search_until_positive(wid, sequence, failed, solve)
        records.extend(history)
        if best:
            best['wrist_search_domain_reselected'] = True
            rows[index] = best
            replaced += 1
        atomic(out/'progress.json', dict(finished=ji+1, requested=len(jobs),
              positive_wrist_reselected=replaced, retained_complete_retreats=sum(bool(r['best']) for r in rows),
              elapsed_s=time.perf_counter()-started))
    for name,sha in {**sources, **codes}.items():
        if digest(ROOT/name) != sha:
            raise ValueError('Source changed during search '+name)
    outputs = {str(p.relative_to(out)): digest(p) for p in out.glob('OP10L*.json')}
    outputs.update({str(p.relative_to(out)): digest(p) for p in (out/'paths').glob('*.json')})
    summary = dict(complete=True, points=21, passed=sum(bool(r['best']) for r in rows),
                  required_straight_distance_mm=2000., requested_points=len(jobs),
                  positive_wrist_reselected=replaced,
                  skipped_identical_finite_searches=sum('skipped_certified_negative' in r for r in records),
                  wall_s=time.perf_counter()-started)
    atomic(out/'report.json', dict(manifest=manifest, summary=summary, rows=rows,
          search_records=records, output_sha256=outputs,
          scope='Finite expanded external-axis/positive-q6 search; identical sealed failed point-angle '
          'searches skipped, not impossibility proofs. Old positive retained on finite failure. '
          'New trajectories require full-source independent cold and fresh 15-axis endpoint transfers.'))
    print('POSITIVE_ANGLE_SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
