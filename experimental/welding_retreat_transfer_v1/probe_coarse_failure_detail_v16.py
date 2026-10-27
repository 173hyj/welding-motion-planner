"""Diagnostic only: locate archived coarse tool-geometry rejections."""
import argparse
import collections
import json
import time
from pathlib import Path
import numpy as np
import probe_analytic_retreat_v2 as exp
from run_geometry import atomic, digest

ROOT = exp.ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-point', type=int, default=24)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Use fresh evidence directory')
    if args.per_point < 1:
        raise ValueError('Positive sample count')
    exp.retreat.assert_frozen_geometry_cache()
    baseline = ROOT/'build/welding_retreat_200cm_inventory_analytic_v10'
    report = json.loads((baseline/'report.json').read_text())
    sources = {**report['manifest']['source_sha256'], **report['manifest']['code_sha256']}
    sources[str((baseline/'report.json').relative_to(ROOT))] = digest(baseline/'report.json')
    evidence_paths = [baseline/n for n in report['output_sha256'] if n.startswith('OP10L') and not n.endswith('_progress.json')]
    for p in evidence_paths:
        if digest(p) != report['output_sha256'][str(p.relative_to(baseline))]:
            raise RuntimeError('Archived evidence changed')
        sources[str(p.relative_to(ROOT))] = digest(p)
    code = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    for name, sha in sources.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Source changed: '+name)
    out.mkdir(parents=True)
    manifest = dict(source_sha256=sources, code_sha256=code,
                    settings=dict(per_point=args.per_point, distance_mm=2000., diagnostic_only=True, rng_seed=20261009))
    atomic(out/'manifest.json', manifest)
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    engine = exp.retreat.VariableToolWorld(scene)
    started = time.perf_counter()
    rows = []
    for p in sorted(evidence_paths):
        evidence = json.loads(p.read_text())
        selected = evidence['initial']
        rid = selected['robot_id']
        weld = next(w for w in scene['welds'] if w['id'] == evidence['weld_id'])
        engine.set_angle(selected['positioner_deg'])
        moved, _ = engine.target(weld, selected)
        model = engine.world.models[rid]
        bank = []
        for pose in evidence.get('alternative_pose_trials', []):
            for initial in pose['trials']:
                for trial in initial['directions']:
                    if trial.get('reason') == 'coarse_geometry_rejected':
                        bank.append((pose['candidate'], initial['q_rad'], trial))
        rng = np.random.default_rng(20261009)
        indices = rng.choice(len(bank), min(len(bank),args.per_point), replace=False)
        records = []
        for index in indices:
            candidate, q, trial = bank[index]
            start = model.tcp(np.asarray(q))
            direction = np.asarray(trial['direction_world'])
            driver = -trial['amplitude_mm']
            first = None
            for travel in np.arange(5., 2000.+1e-7, 10.):
                target = start.copy(); target[:3,3] += travel*direction
                state = engine.state(rid, target, driver, moved, engine.world.static, 'separated')
                if state['blocking_pairs']:
                    first = dict(travel_mm=float(travel), blocking_pairs=state['blocking_pairs'])
                    break
            if first is None:
                raise RuntimeError('Archived coarse rejection did not reproduce')
            record = dict(candidate=candidate, q_rad=q, amplitude_mm=-driver,
                          direction_world=direction.tolist(), first_collision=first)
            if first['travel_mm'] <= 5.:
                opened = engine.state(rid, start, driver, moved, engine.world.static, 'opening')
                context = engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
                target = start.copy(); target[:3,3] += 5.*direction
                release = engine.state(rid,target,driver,moved,engine.world.static,'release',context)
                record['release_phase_at_5mm'] = release
                if not release['blocking_pairs']:
                    record['complete_geometry_certificate'] = exp.retreat.geometry_prefix(
                        engine,rid,start,moved,driver,direction,2000.)
            records.append(record)
        counts = collections.Counter(str(r['first_collision']['travel_mm']) for r in records)
        row = dict(weld_id=evidence['weld_id'], total_archived_coarse_rejections=len(bank),
                   sampled=len(records), first_collision_travel_counts=dict(counts), records=records)
        rows.append(row)
        atomic(out/f'{evidence["weld_id"]}.json', row)
        print('COARSE_DIAG', evidence['weld_id'], dict(counts), flush=True)
    for name, sha in {**sources, **code}.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Evidence changed during diagnosis')
    atomic(out/'report.json',dict(manifest=manifest, rows=rows,
        summary=dict(points=len(rows), sampled=sum(r['sampled'] for r in rows), wall_s=time.perf_counter()-started),
        output_sha256={str(p.relative_to(out)):digest(p) for p in out.glob('OP10L*.json')},
        scope='Reproduced finite archived geometry rejections only. No new qualified robot trajectory.'))


if __name__ == '__main__':
    main()
