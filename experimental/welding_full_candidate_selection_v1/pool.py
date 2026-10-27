"""Register every fixed-condition catalog row and checkpoint source action attempts.

No candidate count, pose chart, origin, winding, or proximity filter is applied.
Finite action-search failures remain unresolved; they are not infeasibility proofs.
"""
import argparse
import concurrent.futures
from contextlib import closing
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).absolute().parents[2]
ENGINE=OUT=SETTINGS=None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def register_catalog(catalog,layers):
    result=[]
    with closing(sqlite3.connect(Path(catalog).absolute().as_uri()+'?mode=ro',uri=True)) as db:
        for layer in layers:
            records=db.execute('SELECT rowid,angle,weld,robot,roll,sign,branch,payload FROM candidates WHERE angle=? AND weld=? AND robot=? ORDER BY rowid',
                               (layer['angle'],layer['weld_id'],layer['robot_id']))
            for rowid,angle,weld,robot,roll,sign,branch,payload in records:
                state=dict(json.loads(payload),robot_id=robot,roll_deg=roll,axis_sign=sign,catalog_branch=branch)
                result.append(dict(id=f'catalog-row-{rowid}',catalog_rowid=rowid,
                                   catalog_key=dict(angle=angle,weld=weld,robot=robot,roll=roll,sign=sign,branch=branch),
                                   payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                                   task=layer['task'],weld_id=weld,angle=angle,robot_id=robot,
                                   catalog_state=state,status='pending',attempt_count=0))
    if len({row['id'] for row in result})!=len(result):raise ValueError('Catalog row registered in multiple task layers')
    return result


def coverage(rows):
    counts={status:sum(row['status']==status for row in rows) for status in ('pending','certified','search_unresolved','proven_invalid')}
    if sum(counts.values())!=len(rows):raise ValueError('Unknown checkpoint status')
    return dict(registered=len(rows),**counts,attempted=len(rows)-counts['pending'],
                complete=counts['pending']==0,all_candidates_registered=True,
                exhaustive_feasibility_search=False)


def pending_rows(rows,retry_unresolved=False):
    return [row for row in rows if row['status']=='pending' or (retry_unresolved and row['status']=='search_unresolved')]


def checkpoint_path(row):
    return OUT/'checkpoints'/(row['id']+'.json')


def same_catalog_action(row,action):
    state=row['catalog_state'];initial=action['initial']
    if any(initial.get(key)!=state[key] for key in ('robot_id','roll_deg','axis_sign','q_rad')):
        return False
    previous=action.get('catalog_state')
    return initial['positioner_deg']==row['angle'] and (previous is None or previous.get('catalog_branch')==state['catalog_branch'])


def evaluate_row(row,baseline_action):
    import numpy as np
    from integrated_retreat_v117 import lift_path,check_path_contract
    from run_direct import geometry_prefix
    from joint_search_v153 import try_configuration
    row=dict(row);row['attempt_count']+=1
    start_time=time.perf_counter()
    state=row['catalog_state'];rid=state['robot_id']
    try:
        reference=json.loads(Path(baseline_action['path_file']).read_text())
        ENGINE.set_angle(row['angle'])
        weld=next(w for w in ENGINE.scene['welds'] if w['id']==row['weld_id'])
        candidate={key:state[key] for key in ('robot_id','roll_deg','axis_sign')}
        q=np.asarray(state['q_rad'],float);model=ENGINE.world.models[rid]
        moved,target=ENGINE.target(weld,candidate)
        if q.shape!=(6,) or not np.isfinite(q).all() or not model.limits_valid(q):
            row.update(status='proven_invalid',reason='catalog_q_violates_raw_source_limits_or_finiteness')
        else:
            start=model.tcp(q);error=float(np.max(abs(start-target)))
            row['source_target_max_matrix_error']=error
            if error>1e-6:
                row.update(status='proven_invalid',reason='catalog_q_does_not_reach_fixed_catalog_pose')
            else:
                selected=dict(candidate,q_rad=state['q_rad'],other_robot_id=reference['other_robot_id'],
                              other_q_rad=reference['other_q_rad'],positioner_deg=row['angle'],
                              target_tcp_matrix=target.ravel().tolist(),actual_tcp_matrix=start.ravel().tolist())
                driver=reference['driver_mm']
                ray=dict(direction_world=reference['direction_world'],direction_frame='current_world',length_mm=1000.)
                geometry=geometry_prefix(ENGINE,rid,start,moved,driver,np.asarray(ray['direction_world']),1000.)
                path=lift_path(ENGINE,weld,candidate,selected,ray,driver_mm=driver) if geometry['passed'] else dict(passed=False,reason='original_ray_geometry_failed')
                row['original_ray_attempt']=dict(passed=path['passed'],reason=path.get('reason'))
                if path['passed']:
                    path.update(weld_id=row['weld_id'],geometry_certificate=geometry)
                else:
                    deadline=time.perf_counter()+SETTINGS['row_seconds']
                    settings=dict(angle_deg=row['angle'],deadline=deadline,directions=16,
                                  amplitudes=list(dict.fromkeys([abs(driver),10.,30.,60.,80.,120.,140.,180.])))
                    result=try_configuration(ENGINE,weld,candidate,q,{selected['other_robot_id']:[selected['other_q_rad']]},settings,{})
                    row['fallback_attempt']=dict(passed=result['passed'],reason=result.get('reason'),ray_trials=len(result.get('attempts',[])))
                    if result['passed']:path=result['path'];selected=result['selected']
                if path['passed']:
                    if selected['q_rad']!=state['q_rad']:raise ValueError('Exact catalog actuator coordinates changed')
                    check_path_contract(path,selected,1000.)
                    file=OUT/'paths'/(row['id']+'.json');atomic(file,path)
                    action=dict(id=row['id'],source='catalog',catalog_rowid=row['catalog_rowid'],catalog_key=row['catalog_key'],
                                catalog_state=state,initial=selected,path_file=str(file.resolve()),path_sha256=digest(file))
                    row.update(status='certified',action=action,reason='complete_action_source_certified')
                else:row.update(status='search_unresolved',reason=path.get('reason','finite_action_search_did_not_certify'))
    except Exception as exc:
        row.update(status='search_unresolved',reason='evaluation_exception',error=type(exc).__name__+': '+str(exc))
    row['elapsed_s']=time.perf_counter()-start_time
    row['search_exhaustive']=False
    atomic(checkpoint_path(row),row)
    print('FULL_CANDIDATE',row['id'],row['task'],row['status'],round(row['elapsed_s'],2),flush=True)
    return row


def seal(bindings):
    for name,checksum in bindings.items():
        if digest(name)!=checksum:raise ValueError('Sealed input changed: '+name)


def build_pool(catalog,previous_pool,out,workers=12,row_seconds=20.,retry_unresolved=False):
    global ENGINE,OUT,SETTINGS
    if workers<1 or not math.isfinite(row_seconds) or row_seconds<=0:raise ValueError('Invalid row evaluation settings')
    if 'fork' not in multiprocessing.get_all_start_methods():raise RuntimeError('Linux fork required')
    sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_retreat_transfer_v1')]
    from integrated_retreat_v117 import check_path_contract
    previous_file=Path(previous_pool).resolve()
    if previous_file.is_dir():previous_file=previous_file/'report.json'
    previous=json.loads(previous_file.read_text());catalog=Path(catalog).resolve();OUT=Path(out).resolve()
    SETTINGS=dict(row_seconds=row_seconds,directions=16,
                  row_seconds_scope='Fallback cooperative ray search only; original-ray authoritative source gate runs to completion')
    layers=[dict(layer,actions=[layer['actions'][0]]) for layer in previous['layers']]
    if len(layers)!=21:raise ValueError('All21 original task layers required')
    rows=register_catalog(catalog,layers)
    bindings={**previous['manifest']['source_sha256'],**previous['manifest']['code_sha256'],
              str(previous_file):digest(previous_file),str(catalog):digest(catalog)}
    for layer in previous['layers']:
        for action in layer['actions']:bindings[action['path_file']]=action['path_sha256']
    dependencies=[Path(__file__)]
    for directory in ('welding_retreat_transfer_v1','welding_tool_exit_v1','welding_variable_open_v1','welding_direct_rrt_v1'):
        dependencies.extend((ROOT/'experimental'/directory).glob('*.py'))
    bindings.update({str(file.resolve()):digest(file) for file in dependencies})
    seal(bindings)
    manifest=dict(bindings=bindings,source_sha256={**previous['manifest']['source_sha256'],str(catalog):digest(catalog)},
                  code_sha256={str(file.resolve()):digest(file) for file in dependencies},settings=SETTINGS,
                  fixed_conditions=[{k:layer[k] for k in ('task','weld_id','angle','robot_id','arm')} for layer in layers],
                  candidate_policy='every_sqlite_row_at_current_weld_angle_assigned_robot_no_q_or_origin_filter',
                  previous_pool_path=str(previous_file),previous_pool_sha256=digest(previous_file),catalog_path=str(catalog),catalog_sha256=digest(catalog))
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_file=OUT/'manifest.json'
    if manifest_file.exists():
        if json.loads(manifest_file.read_text())!=manifest:raise ValueError('Resume seals/settings differ; use fresh output')
    else:atomic(manifest_file,manifest)
    for index,row in enumerate(rows):
        file=checkpoint_path(row)
        if file.exists():
            saved=json.loads(file.read_text())
            for key in ('id','catalog_rowid','catalog_key','payload_sha256','task','catalog_state'):
                if saved[key]!=row[key]:raise ValueError('Checkpoint catalog identity mismatch')
            if saved['status']=='certified':
                action=saved['action']
                if not same_catalog_action(row,action):raise ValueError('Certified checkpoint does not match exact catalog row')
                if digest(action['path_file'])!=action['path_sha256']:raise ValueError('Certified checkpoint path changed')
                check_path_contract(json.loads(Path(action['path_file']).read_text()),action['initial'],1000.)
            rows[index]=saved;continue
        for action in previous['layers'][row['task']-1]['actions']:
            if not same_catalog_action(row,action):continue
            path=json.loads(Path(action['path_file']).read_text());check_path_contract(path,action['initial'],1000.)
            copied=dict(action,id=row['id'],source='catalog',catalog_rowid=row['catalog_rowid'],catalog_key=row['catalog_key'],catalog_state=row['catalog_state'])
            row.update(status='certified',action=copied,reused_from_action_id=action['id'],reason='exact_row_existing_source_receipt',search_exhaustive=False)
            atomic(file,row);break
    def write_report():
        current=[dict(layer,actions=list(layer['actions'])) for layer in layers]
        for row in rows:
            if row['status']=='certified':current[row['task']-1]['actions'].append(row['action'])
        counts=coverage(rows)
        report=dict(layers=current,registry=rows,coverage_all_rows=counts,complete=counts['complete'],all_candidates_registered=True,
                    robot_ids=previous['robot_ids'],baseline_path=previous['baseline_path'],baseline_sha256=previous['baseline_sha256'],
                    manifest=manifest,first_seven_selectable=True,scope='All catalog rows registered and queued; finite ray/transfer searches do not prove exhaustive feasibility or optimality.')
        atomic(OUT/'report.json',report);return report
    write_report();queue=pending_rows(rows,retry_unresolved)
    if queue:
        from run_integrated_retreat_v118 import load_raw_world
        from run_geometry import initialize_worker
        from mixed_bridge_v95 import SIGNS
        scene_file=ROOT/'build/scene_placement_corrected_v1.json'
        if str(scene_file) not in bindings:raise ValueError('Raw source scene not sealed')
        initialize_worker();ENGINE=load_raw_world(json.loads(scene_file.read_text()),SIGNS)
        for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
        lookup={row['id']:i for i,row in enumerate(rows)}
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as executor:
            futures=[executor.submit(evaluate_row,row,layers[row['task']-1]['actions'][0]) for row in queue]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();rows[lookup[row['id']]]=row;write_report()
    seal(bindings)
    return write_report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--previous-pool',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--row-seconds',type=float,default=20.);parser.add_argument('--retry-unresolved',action='store_true')
    args=parser.parse_args()
    report=build_pool(args.catalog,args.previous_pool,args.out,args.workers,args.row_seconds,args.retry_unresolved)
    print(json.dumps(report['coverage_all_rows']),flush=True)


if __name__=='__main__':main()
