"""Same finite GW4 witness with reordered reject-only gates; no live mutation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from signed_wrist_model_v40 import SignedWristModel, branch_parks, path_compatibility
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh witness output required')
    baseline = ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v61/report.json'
    parent = json.loads(baseline.read_text())
    old_dir = ROOT / 'build/welding_retreat_200cm_mixed_wrist_v44'
    old_file = old_dir / 'OP10L-SUB3-POINT-GW-4_angle_30.json'
    old = json.loads(old_file.read_text())
    if not old['best'] or len(old['alternative_pose_trials']) != 30:
        raise ValueError('Exact completed first-30-pose GW4 witness required')
    old_path_file = old_dir / old['best']['path_file']
    old_path = json.loads(old_path_file.read_text())
    if digest(old_path_file) != old['best']['path_sha256']:
        raise ValueError('Changed reference witness')
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256']}
    for p in [baseline, old_file, old_path_file]:
        sources[str(p.relative_to(ROOT))] = digest(p)
    codes = {str((HERE / name).relative_to(ROOT)): digest(HERE / name) for name in
        [Path(__file__).name, 'branch_point_job_v64.py', 'rejection_gates_v63.py',
         'signed_wrist_model_v40.py', 'analytic_ik_v2.py', 'analytic_trace_motion_v2.py',
         'probe_neighbor_rays_v17.py']}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed input: ' + name)
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        weld_id=old['weld_id'], distance_mm=2000., positioner_deg=30.,
        pose_budget=30, direction_budget=10000, reject_order='coarse_then_kinematic_then_fine_geometry',
        same_5_degree_pose_grid=True, full_source_trace_unchanged=True,
        no_live_planner_mutation=True, fresh_cold_required=True))
    atomic(out / 'manifest.json', manifest)
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    start = time.perf_counter()
    search.ENGINE = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    models = search.ENGINE.world.models
    ids = sorted(models)
    signs = {ids[0]: -1, ids[1]: 1}
    parks = branch_parks(models, signs)
    for rid, model in list(models.items()):
        models[rid] = SignedWristModel(model, signs[rid])
    model_load_s = time.perf_counter() - start
    result = search.point_job(old['weld_id'], out, 30, 10000, 30., parks)
    same_states = False
    if result['best']:
        new_path = json.loads((out / result['best']['path_file']).read_text())
        if path_compatibility(new_path, signs, parks) != (True, True):
            raise ValueError('Witness changed signed domain or inactive park')
        identity = all(new_path[k] == old_path[k] for k in
            ['weld_id', 'robot_id', 'other_robot_id', 'other_q_rad', 'positioner_deg',
             'driver_mm', 'length_mm', 'fixed_distance_mm', 'direction_world'])
        if identity and len(new_path['frames']) == len(old_path['frames']):
            same_states = all(np.allclose(a['q_rad'], b['q_rad'], atol=1e-12, rtol=0)
                and a['driver_mm'] == b['driver_mm'] and a['phase'] == b['phase']
                for a, b in zip(new_path['frames'], old_path['frames']))
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source changed during witness: ' + name)
    summary = dict(complete=True, passed=bool(result['best']),
        same_saved_states_as_previous_witness=same_states,
        model_load_s=model_load_s, old_search_wall_s=old['elapsed_s'],
        new_search_wall_s=result['elapsed_s'], fresh_cold_required=True,
        full21_goal_complete=False)
    outputs = {str(p.relative_to(out)): digest(p) for p in
        [*out.glob('OP10L*.json'), *(out / 'paths').glob('*.json')]}
    atomic(out / 'witness_report.json', dict(manifest=manifest, summary=summary,
        result=result, output_sha256=outputs,
        scope='Controlled single-point first-30-pose search using unchanged full trace. '
              'Comparison excludes model loading; no universal speedup claim. '
              'Successful candidate still needs a fresh cold binding and does not complete the full21 goal.'))
    print('ORDERED_WITNESS_SUMMARY', summary, flush=True)
    if not result['best'] or not same_states:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
