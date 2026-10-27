"""Four isolated FCL processes, lazy branch checks and append-only angle events."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(ROOT/'experimental/welding_positioner_v2'),
               str(ROOT/'experimental/welding_endpoints_v1'), str(ROOT)]
import numpy as np
from search import Search
from schedule import search_angles
from positioner import transform_weld
from refinement import roll_target
from world import sha, OUT as GEOMETRY_OUT
from fast_ik import iter_solutions, nearest_seeds

OUT = ROOT/'build/welding_positioner_static_v3'
_ENGINE = None


def job_indices(rows, stage):
    if stage == 'stationary':
        return [i for i, r in enumerate(rows) if r['relative_gun_candidates'] and not r['stationary_done']]
    if stage == 'positioner':
        if any(r['relative_gun_candidates'] and not r['stationary_done'] for r in rows):
            raise ValueError('All stationary searches must finish before changing the positioner')
        return [i for i, r in enumerate(rows) if r['relative_gun_candidates'] and not r['passed'] and not r['positioner_done']]
    raise ValueError(stage)


def summary_counts(rows):
    return dict(source_welds=29, geometry_screened=len(rows),
        geometry_candidates=sum(bool(r['relative_gun_candidates']) for r in rows),
        geometry_discarded=sum(not r['relative_gun_candidates'] for r in rows),
        stationary_completed=sum(r['stationary_done'] for r in rows),
        positioner_resolved=sum(r['passed'] or r['positioner_done'] for r in rows),
        pending_candidates=sum(bool(r['relative_gun_candidates']) and not(r['passed'] or r['positioner_done']) for r in rows),
        stationary_passed=sum(r['stationary_passed'] for r in rows),
        additional_positioner_passed=sum(r['passed'] and not r['stationary_passed'] for r in rows),
        total_passed=sum(r['passed'] for r in rows))


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    temporary.replace(path)


class FastSearch(Search):
    def search_angle(self, row, weld, maprow, deg, previous, save):
        angle_start = time.perf_counter()
        delta = self.environment.set_angle(deg)
        moved, base = transform_weld(self.scene, weld, delta)
        layout = self.environment.layout_collisions(deg)
        trial = dict(positioner_deg=deg, **layout, gun_clear_candidates=0, ik_states=0, ik_seed_starts=0,
                     fixed_gun_rejections=[], completed=False, ik_s=0., full_check_s=0., gun_screen_s=0.,
                     ik_nfev=0, warm_seed_uses=0)
        row['angle_trials'].append(trial)
        if layout['blocking_pairs'] or not row['allowed']:
            trial.update(completed=True, elapsed_s=time.perf_counter()-angle_start)
            return
        if deg == 0 and previous.get('selected'):
            old = previous['selected']; rid = old['robot_id']; q = np.asarray(old['q_rad'])
            actual = self.world.models[rid].tcp(q)
            sign = 1 if actual[:3, 2]@base[:3, 2] > 0 else -1
            start = roll_target(base, 0, sign); rotation = start[:3, :3].T@actual[:3, :3]
            roll = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 360)
            grid_roll = round(roll/5)*5 % 360
            candidate = dict(robot_id=rid, roll_deg=grid_roll, axis_sign=sign)
            if abs(roll/5-round(roll/5)) < .001 and candidate in row['allowed']:
                target = roll_target(base, grid_roll, sign)
                if not self.gun_pairs(rid, target, moved, self.environment.fixed_env):
                    check_start = time.perf_counter()
                    passed = self.full_check(row, weld, rid, q, deg, grid_roll, sign, target, delta)
                    trial['full_check_s'] += time.perf_counter()-check_start
                    if passed:
                        trial.update(completed=True, reused_seed_reverified=True, elapsed_s=time.perf_counter()-angle_start)
                        return
        cache = row.setdefault('warm_start_states', {})
        for candidate in row['allowed']:
            rid, roll, sign = (candidate[k] for k in ('robot_id', 'roll_deg', 'axis_sign'))
            target = roll_target(base, roll, sign)
            screen_start = time.perf_counter()
            pairs = self.gun_pairs(rid, target, moved, self.environment.fixed_env)
            trial['gun_screen_s'] += time.perf_counter()-screen_start
            if pairs:
                trial['fixed_gun_rejections'].append(dict(**candidate, pairs=pairs))
                continue
            trial['gun_clear_candidates'] += 1
            key = f'{rid}:{sign}:{roll}'
            warm = nearest_seeds(cache.get(key, []), deg)
            trial['warm_seed_uses'] += len(warm)
            sources = [np.asarray(c['q_rad']) for c in maprow['candidates']
                       if c['robot_id'] == rid and c['fk_candidate_passed']]
            near = [np.asarray(r['q_rad']) for r in row['full_checks'][-12:] if r['robot_id'] == rid]
            stats = {}
            generator = iter_solutions(self.world.models[rid], target, warm+sources+near, stats=stats)
            try:
                for solution in generator:
                    trial['ik_states'] += 1
                    q = np.asarray(solution['q_rad'])
                    # Save legal FK solutions even if their complete collision
                    # check fails: they are seeds, never feasibility evidence.
                    cache.setdefault(key, []).append((deg, q.tolist()))
                    check_start = time.perf_counter()
                    passed = self.full_check(row, weld, rid, q, deg, roll, sign, target, delta)
                    trial['full_check_s'] += time.perf_counter()-check_start
                    if passed:
                        break
            finally:
                generator.close()
            trial['ik_seed_starts'] += stats['starts']
            trial['ik_nfev'] += stats['nfev']; trial['ik_s'] += stats['solve_s']
            if row['passed']:
                break
        trial.update(completed=True, elapsed_s=time.perf_counter()-angle_start)


def initialize_worker():
    # FCL states and caches are private after fork; meshes are initially shared
    # read-only by the OS. Limit BLAS threads to avoid 4x32 thread contention.
    global _THREAD_LIMIT
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMIT = threadpool_limits(limits=1)
    except ImportError:
        pass


def run_point(index, row, weld, maprow, previous, stage):
    search = _ENGINE
    begin = time.perf_counter()
    row = copy.deepcopy(row)
    event_path = OUT/f'point_{index:02d}.jsonl'
    for deg in search_angles(stage):
        if row['passed']:
            break
        if any(t['positioner_deg'] == deg and t['completed'] for t in row['angle_trials']):
            continue
        search.search_angle(row, weld, maprow, deg, previous, lambda: None)
        trial = row['angle_trials'][-1]
        with event_path.open('a') as stream:
            stream.write(json.dumps(dict(stage=stage, trial=trial, passed=row['passed']), ensure_ascii=False, separators=(',', ':'))+'\n')
        atomic_json(OUT/f'point_{index:02d}_progress.json', dict(weld_id=weld['id'], name=weld['name'], stage=stage,
            angle_deg=deg, passed=row['passed'], elapsed_s=time.perf_counter()-begin,
            ik_seed_starts=sum(t['ik_seed_starts'] for t in row['angle_trials'])))
    if stage == 'stationary':
        row.update(stationary_done=True, status='stationary_verified' if row['passed'] else 'stationary_not_found')
    else:
        row.update(positioner_done=True, status='positioner_verified' if row['passed'] else 'not_found_in_5deg_finite_search')
    row.setdefault('stage_worker_s', {})[stage] = time.perf_counter()-begin
    atomic_json(OUT/f'point_{index:02d}_{stage}.json', row)
    print('POINT', index+1, weld['name'], stage, 'PASS', row['passed'],
          'angle', row['selected']['positioner_deg'] if row['passed'] else None,
          'seconds', round(time.perf_counter()-begin, 2), flush=True)
    return index, row


def main():
    global _ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.out.resolve() != OUT.resolve():
        raise ValueError('This audited release uses its dedicated output directory')
    if args.workers < 1 or args.workers > 4:
        raise ValueError('Use 1 to 4 processes to fit the source FCL meshes in memory')
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists() and not args.resume:
        raise FileExistsError('Preserve previous evidence; use --resume')
    begin = time.perf_counter(); timings = {}
    source = ROOT/'build/scene_placement_corrected_v1.json'
    scene = json.loads(source.read_text()); source_hash = sha(source)
    code_files = [Path(__file__), HERE/'fast_ik.py', ROOT/'experimental/welding_positioner_v2/search.py',
                  ROOT/'experimental/welding_positioner_v2/world.py', ROOT/'experimental/welding_endpoints_v1/refinement.py']
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_files}
    inventory = json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
    old = json.loads((ROOT/'build/welding_endpoints_fixed_normal_v2/report.json').read_text())
    maps = {r['weld_id']: r for r in inventory['rows']}; previous = {r['weld_id']: r for r in old['rows']}
    _ENGINE = FastSearch(scene)
    timings['model_load_s'] = time.perf_counter()-begin
    # Recompute relative candidates. Never reuse the v2 final solution set.
    geometry_begin = time.perf_counter()
    rows = [_ENGINE.relative_screen(weld) for weld in scene['welds']]
    for r in rows:
        if not r['relative_gun_candidates']:
            r.update(stationary_done=True, positioner_done=True, status='all_5deg_rolls_blocked_by_co_rotating_geometry')
    timings['relative_geometry_s'] = time.perf_counter()-geometry_begin
    print('GEOMETRY', summary_counts(rows), flush=True)
    manifest_path = OUT/'run_manifest.json'
    manifest = dict(scene_sha256=source_hash, code_sha256=code_hashes, geometry_audit_sha256=sha(GEOMETRY_OUT/'geometry_audit.json'), workers=args.workers)
    if args.resume and manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
        for i in range(len(rows)):
            for stage in ['stationary', 'positioner']:
                p = OUT/f'point_{i:02d}_{stage}.json'
                if p.exists():
                    rows[i] = json.loads(p.read_text())
    atomic_json(manifest_path, manifest)
    # One pool remains alive across a global stationary barrier.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
        for stage in ['stationary', 'positioner']:
            started = time.perf_counter()
            jobs = [pool.submit(run_point, i, rows[i], scene['welds'][i], maps[rows[i]['weld_id']],
                                previous[rows[i]['weld_id']], stage) for i in job_indices(rows, stage)]
            for future in concurrent.futures.as_completed(jobs):
                i, rows[i] = future.result()
                atomic_json(OUT/'progress.json', dict(phase=stage, elapsed_s=time.perf_counter()-begin,
                                                     summary=summary_counts(rows)))
            timings[stage+'_wall_s'] = time.perf_counter()-started
            print('STAGE', stage, 'SUMMARY', summary_counts(rows), flush=True)
    source_unchanged = sha(source) == source_hash
    assert source_unchanged
    for p in code_files:
        assert sha(p) == code_hashes[str(p.relative_to(ROOT))]
    search_wall = timings['relative_geometry_s']+timings['stationary_wall_s']+timings['positioner_wall_s']
    trials = [t for r in rows for t in r['angle_trials']]
    profiling = dict(timings, search_wall_s=search_wall,
        model_load_and_search_wall_s=time.perf_counter()-begin,
        ik_worker_sum_s=sum(t['ik_s'] for t in trials), full_check_worker_sum_s=sum(t['full_check_s'] for t in trials),
        gun_screen_worker_sum_s=sum(t['gun_screen_s'] for t in trials),
        ik_seed_starts=sum(t['ik_seed_starts'] for t in trials), ik_nfev=sum(t['ik_nfev'] for t in trials),
        ik_solutions=sum(t['ik_states'] for t in trials), gun_clear_pose_calls=sum(t['gun_clear_candidates'] for t in trials),
        full_checks=sum(len(r['full_checks']) for r in rows), warm_seed_uses=sum(t['warm_seed_uses'] for t in trials))
    for row in rows:
        row.pop('warm_start_states', None)
    report = dict(manifest, schema='source_positioner_static_search_v3_fast', phase='complete', workers=args.workers,
        roll_step_deg=5, angle_step_deg=5, positioner_angles_deg=[0]+search_angles('positioner'),
        positioner=_ENGINE.environment.positioner.metadata, geometry_audit_path=str(GEOMETRY_OUT/'geometry_audit.json'),
        profiling=profiling, summary=dict(summary_counts(rows), completed=True, elapsed_s=search_wall), rows=rows,
        scope='Closed driver 0 mm; conditional static results under identical explicit assembly-component exemptions as v2. No rotation path or robot motion certification.',
        ik_search='Lazy full-budget 220-evaluation seed fits, all 16 deterministic legal Sobol starts retained on failure; nearest same-weld/robot/axis-sign/roll external-angle solutions prioritized; accept only after unchanged full contact/collision check.',
        baseline_seed_report_sha256=sha(ROOT/'build/welding_endpoints_fixed_normal_v2/report.json'))
    write_begin = time.perf_counter()
    atomic_json(OUT/'report.json', report)
    profiling['final_report_write_s'] = time.perf_counter()-write_begin
    profiling['end_to_end_wall_s'] = time.perf_counter()-begin
    atomic_json(OUT/'timing.json', profiling)
    print('SUMMARY', json.dumps(report['summary']), 'TIMINGS', json.dumps(profiling), flush=True)


if __name__ == '__main__': main()
