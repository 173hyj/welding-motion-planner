"""Independent SQL coverage checks; sampled coverage is not numerical IK completeness."""
import argparse
import hashlib
import itertools
import json
import sqlite3
from pathlib import Path


def audit_coverage(db, welds, robot_ids, angles):
    expected_poses=set(itertools.product(robot_ids,range(0,360,5),(1,-1)))
    expected_tasks=set(itertools.product(welds,angles))
    relative={r[0]:json.loads(r[1]) for r in db.execute('SELECT weld,payload FROM relative')}
    relative_errors=[];trial_errors=[];missing={}
    for weld in welds:
        row=relative.get(weld,{})
        entries=row.get('allowed',[])+row.get('relative_rejections',[])
        keys=[(p['robot_id'],p['roll_deg'],p['axis_sign']) for p in entries]
        if len(keys)!=len(expected_poses) or set(keys)!=expected_poses:
            relative_errors.append(weld)
    for stage in ('fixed','ik'):
        actual=set(db.execute('SELECT weld,angle FROM tasks WHERE stage=?',(stage,)))
        missing[stage]=len(expected_tasks-actual)
        if actual-expected_tasks:trial_errors.append(dict(stage=stage,unexpected_tasks=len(actual-expected_tasks)))
        for weld,angle in actual & expected_tasks:
            allowed={(p['robot_id'],p['roll_deg'],p['axis_sign']) for p in relative.get(weld,{}).get('allowed',[])}
            found=list(db.execute('SELECT robot,roll,sign FROM trials WHERE stage=? AND weld=? AND angle=?',(stage,weld,angle)))
            if len(found)!=len(allowed) or set(found)!=allowed:
                trial_errors.append(dict(stage=stage,weld=weld,angle=angle,expected=len(allowed),actual=len(found)))
    n=int(db.execute('SELECT count(*) FROM candidates').fetchone()[0])
    complete=not(relative_errors or trial_errors or missing['fixed'] or missing['ik'])
    return dict(complete=complete,sampled_grid_complete=complete,all_mathematical_ik_solutions_proven=False,
                expected_welds=len(welds),expected_angle_tasks=len(expected_tasks),missing_fixed_tasks=missing['fixed'],
                missing_ik_tasks=missing['ik'],relative_grid_errors=relative_errors,trial_coverage_errors=trial_errors,
                accepted_configurations=n,opposite_arm_compatibility_checked=False,continuous_motion_checked=False)


def verify(path):
    path=Path(path)
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    identity=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    changed=[]
    root=Path(identity['root'])
    for name,digest in identity['source_sha256'].items():
        p=root/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(name)
    result=audit_coverage(db,identity['weld_ids'],identity['robot_ids'],identity['positioner_angles_deg'])
    result['source_identity_checked']=not changed;result['changed_sources']=changed
    result['complete']=result['complete'] and not changed
    db.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('database',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();result=verify(args.database)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.out:args.out.write_text(text)
    print(text)
