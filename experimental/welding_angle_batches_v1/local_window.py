"""Replace only arm 1 tasks 3/4, preserving the fastest baseline exterior.

All proposed transfer positives are checked at original SourceBridge fine
resolution. Candidate routes are saved even when duration/rotation gates fail.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=HERE.parent/'welding_retreat_transfer_v1'
WELDS={3:'OP10L-SUB2-POINT-GW-4',4:'OP10L-SUB3-POINT-GW-6'}
ENGINE=None
OUT=None
SECONDS=600.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def window_indices(tasks,phases):
    tasks,phases=np.asarray(tasks),np.asarray(phases)
    if tasks.ndim!=2 or tasks.shape[1]!=2 or phases.shape!=tasks.shape:
        raise ValueError('Baseline tasks/phases must both be N by 2')
    left=np.flatnonzero((tasks[:,1]==2)&(phases[:,1]=='retreat'))
    right=np.flatnonzero((tasks[:,1]==5)&(phases[:,1]=='ingress'))
    if not len(left) or not len(right) or left[-1]>=right[0]:
        raise ValueError('Task2 retreat/task5 ingress window missing or reversed')
    return int(left[-1]),int(right[0])


def splice_states_times(old,times,left,right,middle,middle_times):
    old,middle=np.asarray(old),np.asarray(middle)
    times,middle_times=np.asarray(times),np.asarray(middle_times)
    if not np.allclose(old[left],middle[0],atol=1e-12,rtol=0) or not np.allclose(old[right],middle[-1],atol=1e-12,rtol=0):
        raise ValueError('Replacement endpoints differ from baseline')
    states=np.concatenate([old[:left+1],middle[1:],old[right+1:]])
    end=times[left]+middle_times[-1]
    clock=np.r_[times[:left+1],times[left]+middle_times[1:],times[right+1:]-times[right]+end]
    return states,clock


def accept_metrics(duration,episodes,travel):
    return bool(duration<463.720785 and episodes<7 and travel<=540.+1e-8)


def transfer_job(index,first,last):
    from world_bridge_v2 import SourceBridge
    from run_transfer_limited_v156 import solve_segment
    from endpoint_folded_gateway_v94 import endpoint_folded_gateways
    from shared_gateway import connect_shared
    from signed_wrist_model_v40 import branch_parks
    from mixed_bridge_v95 import SIGNS
    started=time.perf_counter();deadline=started+SECONDS
    class BudgetExpired(RuntimeError):pass
    class Proposal(SourceBridge):
        def valid_scaled(self,state):
            if time.perf_counter()>deadline:raise BudgetExpired()
            return super().valid_scaled(state)
    bridge=Proposal(ENGINE)
    fine=SourceBridge(ENGINE)
    bridge.resolutions[:12]=np.deg2rad(.5)
    a,b=np.asarray(first)/bridge.scale,np.asarray(last)/bridge.scale
    trials=[];path=None
    def fine_accept(candidate):
        return candidate is not None and all(fine.edge_scaled(x,y) for x,y in zip(candidate,candidate[1:]))
    try:
        if bridge.edge_scaled(a,b):
            if fine_accept([a,b]):path=[a,b]
            trials.append(dict(strategy='direct',passed=path is not None))
        if path is None:
            parks=branch_parks(ENGINE.world.models,SIGNS)
            # Preserve the actual first axes; F769's folded side is +3 rad.
            for arm,rid in enumerate(bridge.robot_ids):
                parks[rid][0]=float(first[6*arm])
                if rid.startswith('F769'):parks[rid][0]=3.
            gateways=endpoint_folded_gateways(first,last,bridge.robot_ids,parks)
            # Never introduce the historical both-first-axes=-3 detour.
            gateways=[g/bridge.scale for g in gateways if not (abs(g[0]+3.)<1e-9 and abs(g[6]+3.)<1e-9)]
            gateways.sort(key=lambda g:np.max(abs(g-a))+np.max(abs(g-b)))
            # Same expanded endpoint-folded bank and connect_shared stages as
            # connect_fresh_park_options, with the corrected park first axes.
            for budget,bank in ((0,gateways),(250,gateways[:8])):
                for gateway in bank:
                    for order in ((0,1),(1,0)):
                        candidate,stages=connect_shared(bridge,a,b,gateway,order,index*10000+len(trials),budget)
                        fine_passed=fine_accept(candidate)
                        trials.append(dict(strategy='endpoint_folded_gateway',rrt_stage_budget=budget,
                                           gateway_state=(gateway*bridge.scale).tolist(),arm_order=order,
                                           coarse_passed=candidate is not None,passed=fine_passed,stages=stages))
                        if fine_passed:path=candidate
                        if path is not None:break
                    if path is not None:break
                if path is not None:break
            if path is None and abs(first[12]-last[12])<=1e-12:
                free=list(np.flatnonzero(abs(a-b)>1e-12))
                free=[axis for axis in free if axis!=12]
                candidate,stats=solve_segment(bridge,a,b,free,33000+index,1500)
                if fine_accept(candidate):path=candidate
                trials.append(dict(strategy='full_rrt_fixed_positioner',stats=stats,passed=path is not None))
    except BudgetExpired:
        trials.append(dict(strategy='cooperative_proposal_budget_expired'))
    except ValueError as exc:
        trials.append(dict(strategy='gateway_contract_rejected',reason=str(exc)))
    passed=path is not None
    result=dict(passed=bool(passed),full_source_accepted=bool(passed),trials=trials,
                fine_checks=fine.checks,source_resolution=fine.resolutions.tolist(),
                elapsed_s=time.perf_counter()-started,independent_cold_performed=False)
    if passed:
        frames=fine.dense_frames(path)
        name=f'paths/edge_{index:02d}.json'
        atomic(OUT/name,dict(passed=True,full_source_accepted=True,frames=frames,
                            waypoints_scaled=[p.tolist() for p in path],
                            source_resolution=fine.resolutions.tolist(),independent_cold_performed=False))
        result.update(path_file=name,path_sha256=digest(OUT/name))
    else:result['failure']=fine.last_failure or bridge.last_failure
    atomic(OUT/f'edge_{index:02d}.json',result)
    print('LOCAL_WINDOW_EDGE',index,passed,round(result['elapsed_s'],2),flush=True)
    return index,result


def main():
    global ENGINE,OUT,SECONDS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--actions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--edge-seconds',type=float,default=600.)
    args=parser.parse_args()
    if args.workers<1 or not np.isfinite(args.edge_seconds) or args.edge_seconds<=0:
        parser.error('Positive workers and finite edge seconds required')
    if 'fork' not in multiprocessing.get_all_start_methods():parser.error('Linux fork required')
    OUT=args.out.resolve();SECONDS=args.edge_seconds
    if OUT.exists():raise FileExistsError('Fresh trial output required')
    sys.path[:0]=[str(HERE),str(ROOT),str(LEGACY)]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from integrated_retreat_v117 import online_retreat_ids,check_path_contract
    from mixed_bridge_v95 import SIGNS
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    from welding_kinematics import RobotKinematics
    from replay import append_section,route_times,rotation_metrics
    baseline=args.baseline.resolve()
    if baseline.is_dir():baseline=baseline/'route.npz'
    baseline_sha256=digest(baseline)
    with np.load(baseline,allow_pickle=False) as archive:
        old={key:archive[key].copy() for key in archive.files}
    clock_key='times' if 'times' in old else 'time'
    if clock_key not in old:raise ValueError('Baseline has no time/times array')
    left,right=window_indices(old['tasks'],old['phases'])
    if np.any((old['phases'][left+1:right,0]=='weld')):
        raise ValueError('Replacement would discard an arm 0 welding action')
    source=args.actions.resolve();report_path=source/'report.json'
    report_sha256=digest(report_path)
    report=json.loads(report_path.read_text())
    accepted=online_retreat_ids(report,source)
    if not set(WELDS.values())<=accepted:raise ValueError('Required new tasks3/4 not source accepted')
    bound={section:dict(report['manifest'].get(section,{})) for section in ('source_sha256','code_sha256')}
    bound['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    bound['code_sha256'][str((HERE/'replay.py').relative_to(ROOT))]=digest(HERE/'replay.py')
    for section in bound:
        for name,expected in bound[section].items():
            if digest(ROOT/name)!=expected:raise ValueError('Action source/code changed: '+name)
    scene_path=ROOT/'build/scene_placement_corrected_v1.json'
    scene=json.loads(scene_path.read_text())
    initialize_worker();ENGINE=load_raw_world(scene,SIGNS)
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    rids=sorted(ENGINE.world.models)
    rows={row['weld_id']:row for row in report['rows']}
    paths={};locals_={}
    path_hashes={}
    for task,wid in WELDS.items():
        row=rows[wid];path_file=source/row['best']['path_file']
        path=json.loads(path_file.read_text());check_path_contract(path,row['initial'],1000.)
        if path['robot_id']!=rids[1] or abs(path['positioner_deg']-120.)>1e-9:
            raise ValueError('New action must use arm 1 at exactly 120 degrees')
        paths[task]=path;locals_[task]=local_states(path,rids)
        path_hashes[str(path_file)]=digest(path_file)
    if paths[3]['other_robot_id']!=paths[4]['other_robot_id'] or paths[3]['other_q_rad']!=paths[4]['other_q_rad']:
        raise ValueError('New tasks must share the same inactive park')
    sequence=[old['states'][left],locals_[3][-1],locals_[4][-1],old['states'][right]]
    OUT.mkdir(parents=True)
    atomic(OUT/'proposal.json',dict(left=left,right=right,replaced_tasks=WELDS,
                                    sequence=[state.tolist() for state in sequence],
                                    baseline_sha256=baseline_sha256,actions_report_sha256=report_sha256))
    results=[None]*3
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers,3),
             mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        futures=[pool.submit(transfer_job,i,a,b) for i,(a,b) in enumerate(zip(sequence,sequence[1:]))]
        for future in concurrent.futures.as_completed(futures):
            index,result=future.result();results[index]=result
            atomic(OUT/'progress.json',dict(finished=sum(r is not None for r in results),passed=sum(bool(r and r['passed']) for r in results)))
    audit=dict(manifest=bound,baseline_sha256=baseline_sha256,actions_report_sha256=report_sha256,
               action_path_sha256=path_hashes,edges=results,left=left,right=right,
               accepted=False,full_source_accepted=all(row['passed'] for row in results),
               independent_cold_performed=False,sampled_source_validation=True,
               required_thresholds=dict(duration_less_than_s=463.720785,rotation_episodes_less_than=7,travel_at_most_deg=540.))
    if not audit['full_source_accepted']:
        audit['rejection_reason']='one_or_more_new_transfers_not_source_accepted'
        atomic(OUT/'audit.json',audit);raise SystemExit(2)
    middle=[];phases=[];tasks=[];stages=[]
    def add(section,phase,task):
        start=len(middle);append_section(middle,section);count=len(middle)-start
        phases.extend([['idle',phase]]*count);tasks.extend([[0,task]]*count)
        stages.extend(['local_window']*count)
    def edge_states(index):
        record=results[index];file=OUT/record['path_file']
        if digest(file)!=record['path_sha256']:raise ValueError('Transfer output changed')
        return [frame['state'] for frame in json.loads(file.read_text())['frames']]
    add(edge_states(0),'transfer',2)
    for task in (3,4):
        ingress,close,opening,retreat=action_sections(paths[task],locals_[task])
        add(ingress,'ingress',task);add(close,'close',task)
        add([middle[-1],middle[-1].copy()],'weld',task)
        add(opening,'open',task);add(retreat,'retreat',task)
        add(edge_states(task-2),'transfer',task)
    middle=np.asarray(middle)
    models=[RobotKinematics(scene,next(asset for asset in scene['assets'] if asset['id']==rid)) for rid in rids]
    tcp=np.empty((len(middle),2,3))
    for arm,model in enumerate(models):
        cache={}
        for i,state in enumerate(middle):
            q=state[6*arm:6*arm+6];key=tuple(q)
            if key not in cache:
                if not model.limits_valid(q):raise ValueError('Source robot limit violation')
                cache[key]=model.tcp(q)[:3,3]
            tcp[i,arm]=cache[key]
    if not np.allclose(tcp[0],old['tcp'][left],atol=.05,rtol=0) or not np.allclose(tcp[-1],old['tcp'][right],atol=.05,rtol=0):
        raise ValueError('Baseline boundary TCP disagrees with source FK')
    tcp[0]=old['tcp'][left];tcp[-1]=old['tcp'][right]
    # The final duplicated boundary belongs to the preserved task5 ingress.
    phases[-1]=old['phases'][right].tolist()
    tasks[-1]=old['tasks'][right].tolist()
    # Weld phases apply to arm1; all other timing uses both arms and TCPs.
    local_times=route_times(middle,tcp,[pair[1] for pair in phases])
    new_states,new_times=splice_states_times(old['states'],old[clock_key],left,right,middle,local_times)
    new=dict(old)
    new['states']=new_states
    for key in ('times','time'):
        if key in old:new[key]=new_times
    def splice_array(original,insert):
        return np.concatenate([original[:left+1],np.asarray(insert)[1:],original[right+1:]])
    new['tcp']=splice_array(old['tcp'],tcp)
    new['phases']=splice_array(old['phases'].astype('<U32'),np.asarray(phases,dtype='<U32'))
    new['tasks']=splice_array(old['tasks'],tasks)
    if 'stage' in old:
        original=old['stage']
        fill=np.empty((len(middle),)+original.shape[1:],dtype=original.dtype)
        fill[:]=original[left]
        fill[-1]=original[right]
        new['stage']=splice_array(original,fill)
    # Required exterior arrays and the original per-edge suffix timing survive.
    for key in ('states','tcp','phases','tasks')+ (('stage',) if 'stage' in old else ()):
        if not np.array_equal(new[key][:left+1],old[key][:left+1]) or not np.array_equal(new[key][-len(old[key][right:]):],old[key][right:]):
            raise ValueError('Baseline prefix/suffix changed: '+key)
    def coverage(data):
        return {(arm,int(task)) for arm in range(2)
                for task in np.unique(data['tasks'][data['phases'][:,arm]=='weld',arm]) if task>0}
    if coverage(new)!=coverage(old) or len(coverage(new))!=21:raise ValueError('All21 welding coverage changed')
    metrics=rotation_metrics(new_states)
    duration=float(new_times[-1]-new_times[0])
    audit.update(metrics,duration_s=duration,baseline_duration_s=float(old[clock_key][-1]-old[clock_key][0]),
                 replacement_duration_s=float(local_times[-1]),original_window_duration_s=float(old[clock_key][right]-old[clock_key][left]),
                 all21_coverage_unchanged=True,same_start_end=True,exact_prefix_suffix_preserved=True,
                 accepted=accept_metrics(duration,metrics['rotation_episodes'],metrics['total_positioner_travel_deg']),
                 timing_display_only=True,replaced_task_welds=WELDS)
    for section in bound:
        for name,expected in bound[section].items():
            if digest(ROOT/name)!=expected:raise ValueError('Bound input changed during planning: '+name)
    if digest(baseline)!=audit['baseline_sha256'] or digest(report_path)!=audit['actions_report_sha256']:
        raise ValueError('Baseline/actions report changed during planning')
    for name,expected in path_hashes.items():
        if digest(name)!=expected:raise ValueError('Accepted action path changed')
    np.savez_compressed(OUT/'trial_route.npz',**new)
    audit['trial_array_sha256']=digest(OUT/'trial_route.npz')
    if audit['accepted']:
        import shutil
        shutil.copyfile(OUT/'trial_route.npz',OUT/'route.npz')
        audit['array_sha256']=digest(OUT/'route.npz')
    else:audit['rejection_reason']='duration_rotation_or_travel_acceptance_gate_failed'
    atomic(OUT/'audit.json',audit)
    print(json.dumps({key:audit[key] for key in ('accepted','duration_s','rotation_episodes','total_positioner_travel_deg')}),flush=True)
    if not audit['accepted']:raise SystemExit(2)


if __name__=='__main__':main()
