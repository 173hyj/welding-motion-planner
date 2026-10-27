"""Read-only source probes near three archived failed GW8 traces."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from signed_wrist_model_v40 import SignedWristModel
from analytic_ik_v2 import analytical_continuous_ik
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh diagnostic output required')
    baseline = ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v66/report.json'
    parent = json.loads(baseline.read_text())
    p = ROOT / 'build/welding_retreat_200cm_mixed_wrist_v44/OP10L-SUB1-POINT-GW-8_angle_60.json'
    archived = json.loads(p.read_text())
    cases = []
    for pose in archived['alternative_pose_trials']:
        for trial in pose['trials']:
            for ray in trial['directions']:
                if ray.get('reason') == 'full_retreat_blocked':
                    cases.append(dict(robot_id=pose['candidate']['robot_id'],
                        q_start_rad=trial['q_rad'], direction_world=ray['direction_world'],
                        driver_mm=-ray['amplitude_mm'],
                        archived_last_accepted_mm=ray['last_accepted_travel_mm']))
    if len(cases) != 3:
        raise ValueError('Expected the three completed archived traces')
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
        str(baseline.relative_to(ROOT)): digest(baseline), str(p.relative_to(ROOT)): digest(p)}
    codes = {str((HERE / n).relative_to(ROOT)): digest(HERE / n) for n in
        [Path(__file__).name, 'branch_point_job_v64.py', 'signed_wrist_model_v40.py', 'analytic_ik_v2.py']}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed diagnostic input: ' + name)
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(
        source_positioner_deg=60., actual_source_full_pair_checks=True,
        requested_distance_mm=2000., diagnostic_only=True,
        sampled_failed_states_not_full_trajectory_certificates=True))
    atomic(out / 'manifest.json', manifest)
    started = time.perf_counter()
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    engine = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    models = engine.world.models
    signs = parent['manifest']['settings']['per_robot_wrist_sign']
    parks = parent['manifest']['settings']['per_robot_inactive_park']
    for rid, model in list(models.items()):
        models[rid] = SignedWristModel(model, signs[rid])
    weld = next(w for w in engine.scene['welds'] if w['id'] == archived['weld_id'])
    delta = engine.set_angle(60.)
    moved, _ = search.transform_weld(engine.scene, weld, delta)
    rows = []
    for i, case in enumerate(cases):
        rid = case['robot_id']
        other = next(k for k in models if k != rid)
        model = models[rid]
        q = np.asarray(case['q_start_rad'])
        start = model.tcp(q)
        direction = np.asarray(case['direction_world'])
        probes = []
        final = 2. * np.ceil((case['archived_last_accepted_mm'] + 12.) / 2.)
        for travel in np.arange(2., final + 1e-6, 2.):
            target = start.copy()
            target[:3, 3] += travel * direction
            fit = analytical_continuous_ik(model, target, q)
            if not fit['passed']:
                raise ValueError('Diagnostic continuation could not reach selected probe')
            q = np.asarray(fit['q_rad'])
            if travel < case['archived_last_accepted_mm'] - 4.:
                continue
            state = search.full_probe(engine, rid, q, other, np.asarray(parks[other]),
                case['driver_mm'], moved, 'separated')
            compact = {k: v for k, v in state.items() if k in
                ['passed', 'reason', 'remaining_pairs', 'tool_blocking_pairs', 'source_joint_limits_valid']}
            probes.append(dict(travel_mm=float(travel), q_rad=q.tolist(), state=compact))
        row = dict(case=case, probes=probes)
        rows.append(row)
        atomic(out / ('case_' + str(i) + '.json'), row)
        first = next((p for p in probes if not p['state']['passed']), None)
        print('GW8_BLOCK_PROBE', i, json.dumps(first, ensure_ascii=False), flush=True)
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source changed during diagnostic: ' + name)
    atomic(out / 'report.json', dict(manifest=manifest,
        summary=dict(complete=True, diagnostic_cases=len(rows), wall_s=time.perf_counter()-started),
        rows=rows, output_sha256={p.name: digest(p) for p in out.glob('case_*.json')},
        scope='Actual full-source probes near finite previously failed rays, using source-signed analytic '
              'continuation. Not an exact replay of every archived interpolation frame, not a complete '
              'retreat, not a new collision-free trajectory, and not a proof of infeasibility.'))


if __name__ == '__main__':
    main()
