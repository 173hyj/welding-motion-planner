"""Resumable complete-grid geometry and multi-start numerical IK catalogue."""
import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from core import angle_grid,enumerate_angle
from store import CatalogStore

_SOURCE=None


def execute_angle(source,weld,angle,poses,stage,fixed_rows=None):
    started=time.monotonic();layout=source.set_angle(angle)
    if stage=='fixed':
        rows=[]
        for pose in poses:
            geometry=(dict(passed=False,reason='moving_assembly_vs_fixed_environment',pairs=layout['blocking_pairs'])
                      if layout.get('blocking_pairs') else source.fixed(weld,pose))
            rows.append(dict(**pose,geometry=geometry,status='fixed_clear' if geometry['passed'] else 'geometry_rejected',accepted=[]))
    elif stage=='ik':
        by_pose={(r['robot_id'],r['roll_deg'],r['axis_sign']):r for r in fixed_rows}
        records=enumerate_angle(weld['id'],angle,poses,
            lambda p:by_pose[p['robot_id'],p['roll_deg'],p['axis_sign']]['geometry'],
            lambda p:source.solve(weld,p),lambda p,s:source.accept(weld,p,s))
        rows=[dict(**r['pose'],**{k:v for k,v in r.items() if k!='pose'}) for r in records]
    else:raise ValueError('Unknown stage')
    return rows,dict(layout=layout,elapsed_s=time.monotonic()-started)


def _task(item):
    weld,angle,poses,stage,fixed=item
    # Warm starts are local to an angle task: task scheduling and resume order
    # must not alter the finite numerical seed bank.
    _SOURCE.nearby={rid:[] for rid in _SOURCE.robot_ids}
    rows,metadata=execute_angle(_SOURCE,weld,angle,poses,stage,fixed)
    return weld['id'],angle,rows,metadata


def source_identity(source,seed_count):
    # Seal original source FK/contact rules, numerical solver, the new runner,
    # scene/detail inputs and all adapted collision meshes before forking.
    paths=set(ROOT.glob('*.py'))
    for folder in ('experimental/welding_positioner_v1','experimental/welding_positioner_v2',
                   'experimental/welding_endpoints_v1','experimental/welding_candidate_catalog_v1',
                   'build/eight_hour_v17/experimental_release/multi_point_map'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.add(ROOT/'build/scene_placement_corrected_v1.json')
    audit_path=ROOT/'build/welding_positioner_static_v2/geometry_audit.json'
    paths.add(audit_path);audit=json.loads(audit_path.read_text())
    for section in ('source_sha256','mesh_sha256'):
        paths.update(ROOT/p for p in audit[section])
    paths.add(source.seed_source_path)
    paths.update(ROOT/p for p in ('build/positionerL_detail.json','source/psz/EngData/engData_ver2.xml',
                                 'build/welding_endpoints_v1_corrected_static.json'))
    for asset in source.scene['assets']:
        if 'url' in asset:paths.add(ROOT/'build'/asset['url'])
        for section in ('robot','gun'):
            part=asset.get(section,{})
            if 'source_detail' in part:paths.add(ROOT/part['source_detail'])
            for model in part.get('link_models',{}).values():paths.add(ROOT/'build'/model['url'])
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return dict(schema='exhaustive_static_candidates_v1',root=str(ROOT),source_sha256=hashes,
        weld_ids=[w['id'] for w in source.welds],robot_ids=source.robot_ids,positioner_angles_deg=angle_grid(),
        roll_angles_deg=list(range(0,360,5)),axis_signs=[1,-1],ik_seed_count=seed_count,
        numerical_ik_completeness_proven=False,opposite_arm_compatibility_checked=False,
        positioner_physical_limits_qualified=False)


def make_jobs(store,welds,stage,angles,max_tasks=0):
    jobs=[]
    for weld in welds:
        poses=store.relative(weld['id'])['allowed']
        for angle in angles:
            if store.has_angle(weld['id'],angle,stage):continue
            if stage=='ik' and not store.has_angle(weld['id'],angle,'fixed'):
                if max_tasks:continue
                raise ValueError('Complete fixed-environment screening before IK')
            fixed=store.angle_rows(weld['id'],angle,'fixed') if stage=='ik' else None
            jobs.append((weld,angle,poses,stage,fixed))
    return jobs[:max_tasks] if max_tasks else jobs


def main():
    global _SOURCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--seeds',type=int,default=48)
    parser.add_argument('--stage',choices=['all','relative','fixed','ik'],default='all')
    parser.add_argument('--max-tasks',type=int,default=0,help='Smoke-test task limit; partial results never marked complete')
    args=parser.parse_args()
    if args.workers<1 or args.seeds<1 or args.max_tasks<0:parser.error('Invalid worker/seed/task count')
    from source import SourceCatalog
    _SOURCE=SourceCatalog(seed_count=args.seeds)
    identity=source_identity(_SOURCE,args.seeds)
    args.out.mkdir(parents=True,exist_ok=True)
    store=CatalogStore(args.out/'catalog.sqlite',identity)
    (args.out/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
    for weld in _SOURCE.welds:
        if store.relative(weld['id']) is None:
            row=_SOURCE.relative(weld);store.save_relative(weld['id'],row)
            print('RELATIVE',weld['name'],len(row['allowed']),flush=True)
    print('RELATIVE_COMPLETE',len(_SOURCE.welds),'TRIALS',len(_SOURCE.welds)*len(identity['robot_ids'])*144,flush=True)
    stages=['fixed','ik'] if args.stage=='all' else ([] if args.stage=='relative' else [args.stage])
    # Single-threaded numerical libraries are set in the launcher before imports.
    # Each fork receives independent mutable FCL transforms and warm IK seeds.
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for stage in stages:
            jobs=make_jobs(store,_SOURCE.welds,stage,angle_grid(),args.max_tasks)
            print('STAGE_START',stage,'TASKS',len(jobs),'WORKERS',args.workers,flush=True)
            for done,(wid,angle,rows,metadata) in enumerate(pool.imap_unordered(_task,jobs,chunksize=1),1):
                store.save_angle(wid,angle,rows,stage,metadata)
                if done%10==0 or done==len(jobs):
                    summary=store.summary();summary.update(stage=stage,current_batch_done=done,current_batch_total=len(jobs))
                    (args.out/'progress.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
                    print('PROGRESS',json.dumps(summary,separators=(',',':')),flush=True)
    from verify import verify
    result=verify(args.out/'catalog.sqlite')
    (args.out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    summary=store.summary();summary.update(verification=result)
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');store.close()
    print('RUN_FINISHED',json.dumps(summary,separators=(',',':')),flush=True)


if __name__=='__main__':main()
