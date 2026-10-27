"""All retained 5-degree weld orientations, tool-first source-mesh extraction."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path
import numpy as np
from tool_world import ToolWorld, ROOT, HERE

OUT = ROOT/'build/welding_tool_exit_v1'
_ENGINE = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    files = ['build/scene_placement_corrected_v1.json', 'build/welding_positioner_static_v3/report.json',
             'build/welding_positioner_static_v3/timing.json', 'build/welding_positioner_static_v2/report.json',
             'build/welding_positioner_static_v2/geometry_audit.json', 'build/gunR1_detail.json',
             'build/gunR2_detail.json', 'build/positionerL_detail.json', 'gun_kinematics.py']
    return {name:digest(ROOT/name) for name in files}


def point_job(index, row, weld, out, exits_per_candidate, limit):
    engine = _ENGINE; started = time.perf_counter()
    engine.set_angle(0)
    results = []; counts = dict(closed_contact_not_qualified=0, opening_blocked=0,
                               no_straight_exit_found=0, tool_exit_passed=0)
    candidates = row['allowed'] if not limit else row['allowed'][:limit]
    for i, candidate in enumerate(candidates):
        rid = candidate['robot_id']; moved, target = engine.target(weld, candidate)
        candidate_started = time.perf_counter()
        opening = engine.opening(rid, target, moved)
        opening_s = time.perf_counter()-candidate_started
        record = dict(candidate=candidate, target_tcp_matrix=target.ravel().tolist(), opening=opening,
                      tool_exit_passed=False)
        if not opening['passed']:
            counts[opening['reason']] += 1
        else:
            search = engine.search_exits(rid, target, moved, opening, count=exits_per_candidate)
            record.update(search)
            record['tool_exit_passed'] = bool(search['exits'])
            counts['tool_exit_passed' if search['exits'] else 'no_straight_exit_found'] += 1
        results.append(record)
        record['profiling'] = dict(opening_s=opening_s,exit_search_s=time.perf_counter()-candidate_started-opening_s)
        with (out/f'point_{index:02d}_candidates.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(weld_id=weld['id'],candidate_index=i,record=record),
                                    ensure_ascii=False,separators=(',',':'))+'\n')
        if i % 10 == 0 or i+1 == len(candidates):
            atomic(out/f'point_{index:02d}_progress.json',dict(weld_id=weld['id'], checked=i+1,
                        total=len(candidates), counts=counts, elapsed_s=time.perf_counter()-started))
            print('GEOMETRY', index+1, weld['id'], i+1, '/', len(candidates), counts, flush=True)
    result = dict(weld_id=weld['id'],name=weld['name'],source_candidate_count=len(row['allowed']),
        checked_candidates=len(results), complete=len(results)==len(row['allowed']),
        passed=bool(counts['tool_exit_passed']), counts=counts, candidates=results,
        elapsed_s=time.perf_counter()-started)
    atomic(out/f'point_{index:02d}.json', result)
    return index, result


def initialize_worker():
    global _LIMIT
    try:
        from threadpoolctl import threadpool_limits
        _LIMIT = threadpool_limits(limits=1)
    except ImportError:
        # CLI also sets OMP/OPENBLAS limits before Python starts.
        _LIMIT = None


def main():
    global _ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--point',type=int,help='1-based original weld index for a diagnostic')
    parser.add_argument('--limit-candidates',type=int,default=0)
    parser.add_argument('--exits-per-candidate',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not 1<=args.workers<=4: raise ValueError('Use 1 to 4 isolated FCL processes')
    if args.limit_candidates and args.point is None: raise ValueError('Limits are diagnostic-only')
    out = args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    baseline = source_hashes()
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    previous = json.loads((ROOT/'build/welding_positioner_static_v3/report.json').read_text())
    if previous['scene_sha256'] != baseline['build/scene_placement_corrected_v1.json']:
        raise RuntimeError('Static inventory differs from current source scene')
    settings = dict(positioner_deg=0,opening_step_mm=1.,roll_step_deg=5,
        min_exit_length_mm=100.,max_exit_length_mm=500.,final_clearance_mm=5.,
        release_window_mm=5.,release_step_mm=.1,translation_resolution_mm=.01,
        exits_per_candidate=args.exits_per_candidate,diagnostic_candidate_limit=args.limit_candidates)
    manifest = dict(baseline_sha256=baseline,settings=settings,
        code_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'tool_world.py',HERE/'contracts.py']})
    path = out/'manifest.json'
    if path.exists():
        if not args.resume: raise FileExistsError('Evidence exists; use --resume or a new --out directory')
        if json.loads(path.read_text()) != manifest: raise RuntimeError('Source/code/settings changed; use a new output directory')
    atomic(path,manifest)
    indices = [args.point-1] if args.point else list(range(len(scene['welds'])))
    if any(i<0 or i>=len(scene['welds']) for i in indices): raise ValueError('Invalid weld index')
    rows = {}; jobs = []
    for i in indices:
        row = previous['rows'][i]
        if row['weld_id'] != scene['welds'][i]['id']: raise RuntimeError('Weld identity mismatch')
        saved = out/f'point_{i:02d}.json'
        if saved.exists() and args.resume:
            rows[i] = json.loads(saved.read_text()); continue
        if not row['allowed']:
            rows[i] = dict(weld_id=row['weld_id'],name=row['name'],source_candidate_count=0,
                checked_candidates=0,complete=True,passed=False,status='excluded_before_extraction',candidates=[])
            atomic(saved,rows[i]); continue
        jobs.append(i)
    started = time.perf_counter(); load_s = 0.
    if jobs:
        load_start = time.perf_counter(); _ENGINE = ToolWorld(scene); load_s = time.perf_counter()-load_start
        print('MODEL_LOADED',round(load_s,3),'s',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            futures = [pool.submit(point_job,i,previous['rows'][i],scene['welds'][i],out,
                                   args.exits_per_candidate,args.limit_candidates) for i in jobs]
            for future in concurrent.futures.as_completed(futures):
                i, rows[i] = future.result()
                atomic(out/'progress.json',dict(finished_points=len(rows),total_points=len(indices),
                     passed_points=sum(r['passed'] for r in rows.values()),elapsed_s=time.perf_counter()-started))
    if source_hashes() != baseline: raise RuntimeError('Historical/source evidence changed during search')
    totals = {key:sum(r.get('counts',{}).get(key,0) for r in rows.values()) for key in
              ('closed_contact_not_qualified','opening_blocked','no_straight_exit_found','tool_exit_passed')}
    report = dict(schema='source_tool_first_straight_exit_v1',manifest=manifest,
        summary=dict(source_welds=29,reported_points=len(rows),retained_geometry_points=21,
            checked_candidates=sum(r['checked_candidates'] for r in rows.values()),
            passed_points=sum(r['passed'] for r in rows.values()),counts=totals,
            complete=len(rows)==29 and all(r['complete'] for r in rows.values()),
            model_load_s=load_s,total_wall_s=time.perf_counter()-started),
        rows=[rows[i] for i in sorted(rows)],
        scope='Positioner 0 deg, whole articulated source gun versus products and moving fixture only. '
              'Opening and local release are sampled; strictly separated straight translation uses distance lower bounds. '
              'No arm/fixed-environment/positioner-rotation certification in this report.',
        source_and_historical_evidence_unchanged=True)
    atomic(out/'geometry_report.json',report)
    print('SUMMARY',json.dumps(report['summary'],ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
