"""Source-matched microbenchmark; no planner or safety-certificate replacement."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v40 as search
from signed_wrist_model_v40 import SignedWristModel
from rejection_gates_v63 import filter_ray
from probe_neighbor_rays_v17 import kinematic_rejection
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError('Use a small controlled microbenchmark')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    baseline = ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v61/report.json'
    parent = json.loads(baseline.read_text())
    recordfile = ROOT / 'build/welding_retreat_200cm_mixed_wrist_v44/OP10L-SUB3-POINT-GW-11_angle_120.json'
    record = json.loads(recordfile.read_text())
    cases, seen = [], set()
    for pose in record['alternative_pose_trials']:
        rid = pose['candidate']['robot_id']
        for start in pose['trials']:
            for trial in start['directions']:
                if trial.get('reason') != 'finite_continuation_prefilter_rejected':
                    continue
                key = (rid, *start['q_rad'], *trial['direction_world'])
                if key in seen:
                    continue
                seen.add(key)
                cases.append(dict(robot_id=rid, q_rad=start['q_rad'],
                    amplitude_mm=trial['amplitude_mm'], direction_world=trial['direction_world']))
                if len(cases) == args.count:
                    break
            if len(cases) == args.count:
                break
        if len(cases) == args.count:
            break
    if len(cases) != args.count:
        raise ValueError('Not enough distinct completed reject cases')
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
        str(baseline.relative_to(ROOT)): digest(baseline),
        str(recordfile.relative_to(ROOT)): digest(recordfile)}
    codes = {str((HERE / name).relative_to(ROOT)): digest(HERE / name) for name in
        [Path(__file__).name, 'rejection_gates_v63.py', 'branch_point_job_v40.py',
         'signed_wrist_model_v40.py', 'probe_neighbor_rays_v17.py', 'analytic_ik_v2.py']}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed source: ' + name)
    out.mkdir(parents=True)
    settings = dict(distance_mm=2000., positioner_deg=120., count=args.count,
        selected_negative_cases_only=True, coarse_gate_excluded_from_timing=True,
        no_whole_pipeline_speedup_claim=True, no_positive_path_or_certificate=True)
    atomic(out / 'manifest.json', dict(source_sha256=sources, code_sha256=codes, settings=settings))
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    load_start = time.perf_counter()
    engine = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    models = engine.world.models
    ids = sorted(models)
    signs = {ids[0]: -1, ids[1]: 1}
    for rid, model in list(models.items()):
        models[rid] = SignedWristModel(model, signs[rid])
    weld = next(w for w in engine.scene['welds'] if w['id'] == record['weld_id'])
    delta = engine.set_angle(120.)
    moved, _ = search.transform_weld(engine.scene, weld, delta)
    model_load_s = time.perf_counter() - load_start
    results = []
    for index, case in enumerate(cases):
        model = models[case['robot_id']]
        q = np.asarray(case['q_rad'])
        actual = model.tcp(q)
        direction = np.asarray(case['direction_world'])
        driver = -case['amplitude_mm']
        def kin():
            return kinematic_rejection(model, actual, q, direction)
        def geom():
            return search.retreat.geometry_prefix(engine, case['robot_id'], actual, moved, driver, direction, 2000.)
        row = dict(case=case, execution_order=['old', 'new'] if index % 2 == 0 else ['new', 'old'])
        for mode in row['execution_order']:
            start = time.perf_counter()
            if mode == 'old':
                geometry = geom()
                kinematic = kin() if geometry['passed'] else None
                decision = bool(geometry['passed'] and kinematic['passed'])
                row[mode] = dict(passed=decision, geometry_calls=1,
                    geometry_passed=geometry['passed'], kinematic_prefilter=kinematic)
            else:
                result = filter_ray({}, 'isolated-case', kin, geom)
                row[mode] = dict(passed=result['passed'], geometry_calls=int('geometry' in result),
                    reason=result['reason'], kinematic_prefilter=result['kinematic_prefilter'])
            row[mode]['wall_s'] = time.perf_counter() - start
        if row['old']['passed'] != row['new']['passed'] or row['new']['passed']:
            raise ValueError('Ordered gate acceptance differs from controlled negative evidence')
        if not row['old']['geometry_passed'] or row['new']['kinematic_prefilter']['passed']:
            raise ValueError('Case no longer matches original completed rejection')
        results.append(row)
        atomic(out / 'progress.json', dict(finished=len(results), requested=len(cases)))
        print('REJECTION_ORDER_CASE', index, row['old']['wall_s'], row['new']['wall_s'], flush=True)
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source changed during benchmark: ' + name)
    summary = dict(complete=True, cases=len(results), same_acceptance=True,
        model_load_s=model_load_s,
        old_gate_wall_s=sum(r['old']['wall_s'] for r in results),
        new_gate_wall_s=sum(r['new']['wall_s'] for r in results),
        old_geometry_calls=sum(r['old']['geometry_calls'] for r in results),
        new_geometry_calls=sum(r['new']['geometry_calls'] for r in results))
    atomic(out / 'report.json', dict(manifest=json.loads((out / 'manifest.json').read_text()),
        summary=summary, rows=results,
        scope='Selected coarse-passed/kinematic-rejected source rays only. Alternating execution order; '
              'model loading and prior coarse checks excluded. Gate microbenchmark, not full search speedup, '
              'not collision certification and not a proof of infeasibility. Existing live planner unmodified.'))
    print('REJECTION_ORDER_SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
