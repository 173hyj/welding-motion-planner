"""Source-validated analytic FANUC retry with a lossless numerical union."""
import argparse
import copy
import hashlib
import json
import multiprocessing
import sqlite3
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experimental.welding_candidate_catalog_v1.store import CatalogStore
from experimental.welding_candidate_catalog_v1.verify import verify

_SOURCE=None
_OLD_PATH=None


def copy_geometry(old,new):
    """Copy sealed relative/fixed records only, leaving old IK untouched."""
    with new.db:
        for table in ('relative','tasks','trials'):
            query='SELECT * FROM '+table
            if table!='relative':query+=" WHERE stage='fixed'"
            for row in old.execute(query):
                new.db.execute('INSERT OR IGNORE INTO '+table+' VALUES('+','.join('?' for _ in row)+')',row)


def _task(job):
    from experimental.welding_candidate_catalog_v1.run import execute_angle
    from experimental.welding_candidate_catalog_analytic_v1.union import merge_trials
    weld,angle,poses,fixed=job
    analytic,meta=execute_angle(_SOURCE,weld,angle,poses,'ik',fixed_rows=fixed)
    db=sqlite3.connect(f'file:{_OLD_PATH}?mode=ro',uri=True)
    numeric=[json.loads(r[0]) for r in db.execute("SELECT payload FROM trials WHERE stage='ik' AND weld=? AND angle=?",(weld['id'],angle))]
    db.close()
    merged,comparison=merge_trials(analytic,numeric)
    meta['comparison']=comparison
    meta['analytic_ik_elapsed_s']=sum(r.get('ik_stats',{}).get('elapsed_s',0.) for r in analytic)
    return weld['id'],angle,merged,meta


def main():
    global _SOURCE,_OLD_PATH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--max-tasks',type=int,default=0)
    args=parser.parse_args()
    baseline_check=verify(args.baseline)
    if not baseline_check['complete']:raise ValueError('Baseline is incomplete or its source identity changed')
    _OLD_PATH=args.baseline
    old=sqlite3.connect(f'file:{args.baseline}?mode=ro',uri=True)
    identity=copy.deepcopy(json.loads(old.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0]))
    identity.update(schema='analytic_fanuc_candidate_union_v1',ik_solver='source-geometric analytic equivalent of supplied FANUC fragment',
        baseline_database_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        numerical_ik_completeness_proven=False,analytic_global_completeness_proven=False)
    for p in sorted(HERE.iterdir()):
        if p.suffix in ('.py','.cpp'):identity['source_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    from experimental.welding_candidate_catalog_analytic_v1.source import AnalyticSourceCatalog
    _SOURCE=AnalyticSourceCatalog()
    identity['analytic_model_mapping']={rid:dict(getattr(solver,'metadata',{}),mapping_proof=solver.mapping_proof) for rid,solver in _SOURCE.analytic.items()}
    store=CatalogStore(args.out/'catalog.sqlite',identity)
    copy_geometry(old,store)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
    jobs=[]
    for weld in _SOURCE.welds:
        poses=store.relative(weld['id'])['allowed']
        for angle in identity['positioner_angles_deg']:
            if not store.has_angle(weld['id'],angle,'ik'):
                jobs.append((weld,angle,poses,store.angle_rows(weld['id'],angle,'fixed')))
    if args.max_tasks:jobs=jobs[:args.max_tasks]
    old.close();started=time.monotonic()
    print('ANALYTIC_START',len(jobs),'WORKERS',args.workers,flush=True)
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for done,(wid,angle,rows,metadata) in enumerate(pool.imap_unordered(_task,jobs,chunksize=1),1):
            store.save_angle(wid,angle,rows,'ik',metadata)
            if done%20==0 or done==len(jobs):
                progress=store.summary();progress.update(batch_done=done,batch_total=len(jobs),elapsed_s=time.monotonic()-started)
                (args.out/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2))
                print('ANALYTIC_PROGRESS',json.dumps(progress,separators=(',',':')),flush=True)
    result=verify(args.out/'catalog.sqlite')
    (args.out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    summary=store.summary();summary.update(verification=result,elapsed_s=time.monotonic()-started)
    totals={}
    for row in store.db.execute("SELECT payload FROM tasks WHERE stage='ik'"):
        meta=json.loads(row[0])
        for key,value in meta.get('comparison',{}).items():
            if isinstance(value,(int,float)):totals[key]=totals.get(key,0)+value
    summary['comparison']=totals
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');store.close()
    print('ANALYTIC_FINISHED',json.dumps(summary,separators=(',',':')),flush=True)


if __name__=='__main__':main()
