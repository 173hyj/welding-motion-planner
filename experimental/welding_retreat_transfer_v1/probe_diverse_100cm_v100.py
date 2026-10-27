"""Fresh 1000mm search over additional archived poses; preserve certified candidates."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import run_retreat_100cm_v97 as exp
from signed_wrist_model_v40 import SignedWristModel
from run_geometry import atomic, digest, initialize_worker

ROOT = exp.ROOT
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); out = args.out.resolve()
    if out.exists(): raise FileExistsError('Fresh output required')
    baseline = ROOT / 'build/welding_retreat_100cm_v97'
    parent = json.loads((baseline / 'report.json').read_text())
    cold_path = ROOT / 'build/welding_retreat_100cm_v97_cold/cold_verification.json'
    cold = json.loads(cold_path.read_text())
    if (not parent['summary']['complete'] or parent['summary']['required_straight_distance_mm'] != 1000. or
            len(parent['rows']) != 21 or cold['retreat_report_sha256'] != digest(baseline / 'report.json') or
            not cold['summary']['all_selected_passed']):
        raise ValueError('Full identity inventory and SHA-bound positive cold evidence required')
    bank_path = ROOT / 'build/welding_retreat_100cm_seedbank_v100/seeds.json'
    bank = json.loads(bank_path.read_text())
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
               **bank['source_sha256']}
    for path in (baseline / 'report.json', cold_path, bank_path):
        sources[str(path.relative_to(ROOT))] = digest(path)
    for name, value in parent['output_sha256'].items():
        sources[str((baseline / name).relative_to(ROOT))] = value
    codes = {str(path.relative_to(ROOT)): digest(path) for path in (Path(__file__), HERE / 'run_retreat_100cm_v97.py')}
    for name, value in {**sources, **codes}.items():
        if digest(ROOT / name) != value: raise ValueError('Changed source ' + name)
    old = json.loads((ROOT / 'build/welding_retreat_200cm_large_new_seeds_v93/report.json').read_text())
    settings = old['manifest']['settings']; seeds = bank['seeds']
    # Diversify external angles and robots before neighbouring rolls at one angle.
    unique = []; deferred = []; used = set()
    for seed in seeds:
        key = (seed['positioner_deg'], seed['candidate']['robot_id'])
        if key in used: deferred.append(seed)
        else: unique.append(seed); used.add(key)
    seeds = unique + deferred
    manifest = dict(source_sha256=sources, code_sha256=codes, settings=dict(distance_mm=1000.,
        direction_budget=256, posture_roll_step_deg=5, sphere_step_deg=5, seeds=seeds,
        per_robot_wrist_sign=settings['per_robot_wrist_sign'],
        per_robot_inactive_park=settings['per_robot_inactive_park'], fresh_negative_rechecks=True,
        retained_cold_candidates=20, fresh_cold_required=True, archived_negatives_only_rank_seeds=True))
    out.mkdir(parents=True); atomic(out / 'manifest.json', manifest)
    started = time.perf_counter(); initialize_worker(); exp.search.retreat.assert_frozen_geometry_cache()
    exp.ENGINE = exp.search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    for rid, model in list(exp.ENGINE.world.models.items()):
        exp.ENGINE.world.models[rid] = SignedWristModel(model, settings['per_robot_wrist_sign'][rid])
    rows = copy.deepcopy(parent['rows'])
    passed_cold = {r['weld_id']: r for r in cold['retreat_rows'] if r['passed']}
    for i, row in enumerate(rows):
        if row['best']:
            b = row['best']; check = passed_cold.get(row['weld_id'])
            if not check or check['planned_path_sha256'] != b['path_sha256']:
                raise ValueError('Retained candidate has no matching cold evidence')
            source = baseline / b['path_file']; data = json.loads(source.read_text())
            if digest(source) != b['path_sha256'] or data['length_mm'] != 1000. or data['fixed_distance_mm'] != 1000.:
                raise ValueError('Retained identity/distance mismatch')
            name = f'paths/retained_{i:02d}.json'; (out / name).parent.mkdir(exist_ok=True)
            os.link(source, out / name); b['path_file'] = name
        else:
            if row['weld_id'] != bank['weld_id']: raise ValueError('Seed bank identity mismatch')
            _, rows[i] = exp.point_job(i, row, baseline, out, seeds, settings['per_robot_inactive_park'], 256)
        atomic(out / 'progress.json', dict(finished=i+1, requested=21, passed=sum(bool(r['best']) for r in rows),
                                          elapsed_s=time.perf_counter()-started))
    for name, value in {**sources, **codes}.items():
        if digest(ROOT / name) != value: raise ValueError('Source changed during search ' + name)
    outputs = {str(path.relative_to(out)): digest(path) for path in out.rglob('*.json')
               if path.name not in ('report.json', 'manifest.json', 'progress.json')}
    summary = dict(complete=True, points=21, requested_points=21, passed=sum(bool(r['best']) for r in rows),
        required_straight_distance_mm=1000., historical_chain_intact=False, wall_s=time.perf_counter()-started)
    atomic(out / 'report.json', dict(manifest=manifest, summary=summary, rows=rows, output_sha256=outputs,
        scope='Fresh finite 1000mm directions at a diverse archived source-exact pose bank; no old negatives accepted. '
              '20 SHA-bound cold positives retained immutable; new report and new positives still require fresh cold. '
              'Finite negative is not infeasibility proof; full 21 identity inventory preserved.'))
    print('DIVERSE_ONE_METER_SUMMARY', summary, flush=True)


if __name__ == '__main__': main()
