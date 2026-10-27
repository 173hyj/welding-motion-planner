"""Retry failed angle actions in fresh evidence, preserving accepted receipts.

Example: python -m experimental.welding_angle_batches_v1.retry_actions
--resume-from build/angle90a --out build/angle90b --workers 8
--point-seconds 600 --directions 64
"""
import argparse
import concurrent.futures
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=HERE.parent/'welding_retreat_transfer_v1'
ACTION=None
BASE_RANK=None
PRIOR={}


def state_key(row):
    candidate=row.get('candidate',row)
    return (candidate['robot_id'],candidate['roll_deg'],candidate['axis_sign'],tuple(row['q_rad']))


def retry_order(rows, attempts):
    """Untried exact actuator states, last attempted state, then old failures."""
    tried={state_key(attempt) for attempt in attempts}
    last=state_key(attempts[-1]) if attempts else None
    fresh=[row for row in rows if state_key(row) not in tried]
    interrupted=[row for row in rows if state_key(row)==last]
    fallback=[row for row in rows if state_key(row) in tried and state_key(row)!=last]
    return fresh+interrupted+fallback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_accepted(row, source, target, index):
    copied=deepcopy(row)
    path=Path(source)/row['best']['path_file']
    checksum=row['best']['path_sha256']
    if digest(path)!=checksum or row['online_acceptance']['path_sha256']!=checksum:
        raise ValueError('Accepted source path changed')
    name=f'paths/retreat_{index:02d}.json'
    destination=Path(target)/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,destination)
    if digest(destination)!=checksum:
        raise ValueError('Copied accepted path changed')
    copied['best']['path_file']=name
    return copied


def evaluate_retry(index,weld,candidates,previous):
    attempts=PRIOR[weld['id']].get('attempts',[])
    ACTION.rank_candidates=lambda rows,old=None,bounds=None:retry_order(BASE_RANK(rows,old,bounds),attempts)
    try:
        return ACTION.evaluate_point(index,weld,candidates,previous)
    finally:
        ACTION.rank_candidates=BASE_RANK


def main():
    global ACTION,BASE_RANK,PRIOR
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--catalog',type=Path)
    parser.add_argument('--angle',type=float)
    parser.add_argument('--previous-retreat',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--point-seconds',type=float,default=600.)
    parser.add_argument('--directions',type=int,default=64)
    args=parser.parse_args()
    import math
    if args.workers<1 or not 1<=args.directions<=2522 or not math.isfinite(args.point_seconds) or args.point_seconds<=0:
        parser.error('Invalid finite retry budgets')
    if 'fork' not in multiprocessing.get_all_start_methods():
        parser.error('Linux fork is required')
    sys.path[:0]=[str(HERE),str(ROOT),str(LEGACY)]
    import actions as action
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from signed_wrist_model_v40 import branch_parks
    from integrated_retreat_v117 import online_retreat_ids
    ACTION=action
    BASE_RANK=action.rank_candidates
    begun=time.perf_counter()
    source=args.resume_from.resolve()
    report_path=source/'report.json' if source.is_dir() else source
    source=report_path.parent
    prior=json.loads(report_path.read_text())
    online_retreat_ids(prior,source)
    manifest=deepcopy(prior['manifest'])
    for section in ('source_sha256','code_sha256'):
        for name,expected in manifest.get(section,{}).items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Prior bound input changed: '+name)
    catalog=(args.catalog or Path(manifest['catalog_path'])).resolve()
    if digest(catalog)!=manifest['catalog_sha256']:
        raise ValueError('Retry catalog differs from accepted source catalog')
    angle=manifest['settings']['angle_deg'] if args.angle is None else args.angle
    if angle!=manifest['settings']['angle_deg']:
        raise ValueError('Retry must preserve the original angle')
    out=args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh retry evidence directory required')
    PRIOR={row['weld_id']:row for row in prior['rows']}
    required=manifest['required_weld_ids']
    if set(required)!=set(PRIOR):
        raise ValueError('Prior report does not cover all required weld identities')
    previous={}
    previous_path=None
    if args.previous_retreat:
        previous_path=args.previous_retreat/'report.json'
    elif manifest.get('previous_report'):
        previous_path=Path(manifest['previous_report']['path'])
        if digest(previous_path)!=manifest['previous_report']['sha256']:
            raise ValueError('Previous preference report changed')
    if previous_path:
        preference=json.loads(previous_path.read_text())
        previous={row['weld_id']:dict(row['initial'],driver_mm=row['best']['driver_mm'])
                  for row in preference['rows'] if row.get('best')}
        manifest['previous_report']=dict(path=str(previous_path.resolve()),sha256=digest(previous_path))
    settings=dict(manifest['settings'],angle_deg=angle,point_seconds=args.point_seconds,
                  directions=args.directions,workers=args.workers)
    manifest['settings']=settings
    manifest['retry']=dict(resume_report_path=str(report_path),resume_report_sha256=digest(report_path),
                           retained_accepted_actions=sum(bool(row.get('passed')) for row in prior['rows']),
                           failed_candidate_order='unattempted exact actuator states, last attempted, prior failures')
    manifest['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    out.mkdir(parents=True)
    action.atomic(out/'manifest.json',manifest)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    by_weld={weld['id']:weld for weld in scene['welds']}
    rows=[None]*len(prior['rows'])
    failed=[]
    for index,row in enumerate(prior['rows']):
        if row.get('passed'):
            rows[index]=copy_accepted(row,source,out,index)
            action.atomic(out/f'point_{index:02d}.json',rows[index])
        else:
            failed.append((index,by_weld[row['weld_id']]))
    if failed:
        initialize_worker()
        action.ENGINE=load_raw_world(scene,SIGNS)
        for rid,model in list(action.ENGINE.world.models.items()):
            action.ENGINE.world.models[rid]=getattr(model,'source',model)
        action.PARKS=manifest.get('inactive_parks') or {rid:[q] for rid,q in branch_parks(action.ENGINE.world.models,SIGNS).items()}
        action.SETTINGS=settings
        action.OUT=out
        candidates=action.load_candidates(catalog,angle)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            futures=[pool.submit(evaluate_retry,index,weld,candidates.get(weld['id'],[]),previous.get(weld['id']))
                     for index,weld in failed]
            for future in concurrent.futures.as_completed(futures):
                index,row=future.result()
                rows[index]=row
                action.atomic(out/'progress.json',dict(finished=sum(row is not None for row in rows),
                                                      passed=sum(bool(row and row.get('passed')) for row in rows)))
    for section in ('source_sha256','code_sha256'):
        for name,expected in manifest.get(section,{}).items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Bound input changed during retry: '+name)
    if digest(report_path)!=manifest['retry']['resume_report_sha256'] or digest(catalog)!=manifest['catalog_sha256']:
        raise ValueError('Resume report/catalog changed during retry')
    summary=action.action_summary(rows,len(required),angle)
    summary.update(wall_s=time.perf_counter()-begun,reused_actions=len(rows)-len(failed),retried_welds=len(failed))
    outputs={str(path.relative_to(out)):digest(path) for path in out.rglob('*.json')
             if path.name not in ('report.json','manifest.json','progress.json')}
    report=dict(manifest=manifest,summary=summary,rows=rows,output_sha256=outputs,
                scope='Accepted actions copied byte-for-byte from sealed resume evidence; failed actions freshly regenerated at same angle from exact catalog states. Local actions only, transfers/full cycle not claimed. Sampled source acceptance, no independent cold certification.')
    online_retreat_ids(report,out)
    action.atomic(out/'report.json',report)
    print(json.dumps(summary),flush=True)
    if not summary['all_requested_actions_passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
