"""Offline fixed-candidate dual-arm pilot, accepted only by source-mesh checks."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'experimental/welding_retreat_transfer_v1'),
              str(ROOT/'experimental/welding_variable_open_v1')]
from experimental.dual_arm_parallel_v1.motion import new_chain,append_piece,piece_at
from experimental.dual_arm_parallel_v1.schedule import align_batches,merge_samples


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def combine_state(a,b,angle):return np.r_[a[:6],b[:6],np.deg2rad(angle),a[6],b[6]]


def sample_chain(chain,t):
    states=np.asarray(chain['states']);times=np.asarray(chain['times'])
    state=np.array([np.interp(t,times,states[:,i]) for i in range(7)])
    return state,piece_at(chain,t)


def overlap_seconds(a,b,offsets,phase='weld'):
    return float(sum(max(0.,min(p['end']+offsets[0],q['end']+offsets[1])-
                         max(p['start']+offsets[0],q['start']+offsets[1]))
                     for p in a if p['phase']==phase for q in b if q['phase']==phase))


def release_history_passed(samples,initial,reverse=False):
    sys.path.insert(0,str(ROOT/'experimental/welding_tool_exit_v1'))
    from contracts import ContactTracker
    tracker=ContactTracker(initial)
    return all(tracker.advance(touching,travel) for travel,touching in (reversed(samples) if reverse else samples))


def check_baseline_timing(data):
    states=np.asarray(data['states']);tcp=np.asarray(data['tcp']);times=np.asarray(data['times'])
    if times[0]!=0. or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid baseline time origin/order')
    dt=np.maximum(np.max(abs(np.diff(states[:,:12],axis=0)),axis=1)/np.deg2rad(30.),
                  np.max(abs(np.diff(states[:,13:],axis=0)),axis=1)/80.)
    dt=np.maximum(dt,abs(np.diff(states[:,12]))/np.deg2rad(15.))
    dt=np.maximum(dt,np.max(np.linalg.norm(np.diff(tcp,axis=0),axis=2),axis=1)/200.)
    dt=np.maximum(dt,1e-5);dt[np.asarray(data['phases'])[1:]=='weld']=.6
    if not np.allclose(np.diff(times),dt,atol=1e-9,rtol=1e-9):raise ValueError('Baseline display timing assumptions differ')
    return float(times[-1]-times[0])


def load_baseline(path,route,initial,final,sources):
    audit_file=path.parent/'audit.json';audit=json.loads(audit_file.read_text())
    if not audit['full_route_online_accepted'] or not audit['timing_display_only'] or sha(path)!=audit['array_sha256']:
        raise ValueError('Unsealed baseline replay')
    if audit['order']!=route['order_weld_ids'] or audit['robot_ids']!=route['robot_ids']:raise ValueError('Different baseline tasks')
    for p,h in audit['source_sha256'].items():
        p=resolve_source(p)
        if sha(p)!=h:raise ValueError('Changed baseline source '+str(p))
        sources[str(p)]=h
    data=np.load(path,allow_pickle=False)
    if not np.allclose(data['states'][0],initial,atol=1e-12,rtol=0) or not np.allclose(data['states'][-1],final,atol=1e-12,rtol=0):
        raise ValueError('Different baseline endpoint states')
    adapter=json.loads(resolve_source(audit['transfer_report']).read_text())
    for key in ('order','robot_ids','nodes','selected_edges','final_retreat_out'):
        if adapter[key]!=route[key]:raise ValueError('Baseline route identity differs: '+key)
    sources[str(path.resolve())]=sha(path);sources[str(audit_file.resolve())]=sha(audit_file)
    return check_baseline_timing(data)


def resolve_source(name):
    path=Path(name)
    if path.is_absolute() and not path.exists() and 'build' in path.parts:
        path=ROOT.joinpath(*path.parts[path.parts.index('build'):])
    return path if path.is_absolute() else ROOT/path


def load_tasks(route_file):
    from experimental.trajopt_transfer_v1.export_replay import verify_route
    from replay_full21_core_v105 import local_states
    route,baseline,sources=verify_route(route_file)
    sources={str(resolve_source(p)):h for p,h in sources.items()}
    for group in ('source_sha256','code_sha256'):
        for p,h in baseline['manifest'][group].items():sources[str(resolve_source(p))]=h
    tasks=[];parks=[None,None]
    for number,index in enumerate(route['order'],1):
        node=route['nodes'][index];path_file=resolve_source(node['retreat_path_file'])
        if sha(path_file)!=node['retreat_path_sha256']:raise ValueError('Changed original local action')
        sources[str(path_file)]=sha(path_file);path=json.loads(path_file.read_text())
        arm=route['robot_ids'].index(path['robot_id']);other=1-arm
        park=np.r_[path['other_q_rad'],0.]
        if parks[other] is not None and not np.array_equal(parks[other],park):raise ValueError('Inconsistent existing park')
        parks[other]=park
        if path['length_mm']!=1000. or path['weld_id']!=node['weld_id']:raise ValueError('Changed fixed weld action')
        states=np.asarray(local_states(path,route['robot_ids']))
        tasks.append(dict(id=number,node=index,weld_id=node['weld_id'],arm=arm,angle=path['positioner_deg'],
                          path=path,states=states,path_file=str(path_file)))
    if len(tasks)!=21 or len(set(t['weld_id'] for t in tasks))!=21:raise ValueError('Exact full21 required')
    for p,h in sources.items():
        if sha(p)!=h:raise ValueError('Changed input '+p)
    return route,tasks,parks,sources


def prepare_context(engine,task):
    p=task['path'];rid=p['robot_id'];engine.set_angle(task['angle'])
    weld=next(w for w in engine.scene['welds'] if w['id']==task['weld_id'])
    moved,_=engine.target(weld,p['candidate']);target=engine.world.models[rid].tcp(p['q_start_rad'])
    opened=engine.state(rid,target,p['driver_mm'],moved,engine.world.static,'opening')
    if opened['blocking_pairs']:raise RuntimeError('Original opening no longer accepted')
    release=engine.release_context(rid,target,moved,dict(open_state=opened),driver_mm=p['driver_mm'])
    return dict(robot_id=rid,weld=moved,target=target,direction=np.asarray(p['direction_world']),release_context=release)


def action_context(engine,meta,piece,state7):
    if piece is None or piece['task'] is None:return None
    task=meta[piece['task']];phase=piece['phase'];context=task['context']
    if phase in ('close','weld'):contact='closed'
    elif phase=='open':contact='opening'
    elif phase in ('ingress','retreat'):
        tcp=engine.world.models[context['robot_id']].tcp(state7[:6])
        travel=float((tcp[:3,3]-context['target'][:3,3])@context['direction'])
        contact='release' if travel<=5.000001 else 'separated'
    else:return None
    return dict(robot_id=context['robot_id'],weld=context['weld'],phase=contact,
                release_context=context['release_context'] if contact=='release' else None)


class TimelineValidator:
    def __init__(self,engine,checker,meta,pool=None):
        self.engine=engine;self.checker=checker;self.meta=meta;self.pool=pool;self.cache={};self.cache_hits=0

    def validate(self,chains,angle,offsets,fine):
        duration=max(offsets[i]+chains[i]['times'][-1] for i in range(2))
        if fine:
            times,states=merge_samples(chains[0],chains[1],offsets=offsets,angle_deg=angle)
        else:
            boundaries=[p[k]+offsets[a] for a in range(2) for p in chains[a]['pieces'] for k in ('start','end')]
            times=np.unique(np.r_[np.arange(0.,duration,.5),duration,boundaries])
            states=np.array([combine_state(sample_chain(chains[0],t-offsets[0])[0],
                            sample_chain(chains[1],t-offsets[1])[0],angle) for t in times])
        phases=[];tasks=[];release_samples={};fresh_keys=set()
        if fine and self.pool is not None:
            pending={}
            for t,state in zip(times,states):
                pieces=[piece_at(chains[a],t-offsets[a]) for a in range(2)]
                local=[np.r_[state[6*a:6*a+6],state[13+a]] for a in range(2)]
                contexts=[action_context(self.engine,self.meta,pieces[a],local[a]) for a in range(2)]
                key=(state.tobytes(),tuple((pieces[a]['task'],contexts[a]['phase']) if contexts[a] else None for a in range(2)))
                if key not in self.cache:pending[key]=(state,contexts)
            print('TIMELINE_POOL_START',angle,offsets,len(pending),flush=True)
            results=self.pool.dual_states(pending.values())
            if len(results)!=len(pending):raise RuntimeError('Source pool lost timeline samples')
            self.cache.update(zip(pending,results));self.checker.checks+=len(pending);fresh_keys=set(pending)
            print('TIMELINE_POOL_END',angle,offsets,len(pending),flush=True)
        for j,(t,state) in enumerate(zip(times,states)):
            pieces=[piece_at(chains[a],t-offsets[a]) for a in range(2)]
            local=[np.r_[state[6*a:6*a+6],state[13+a]] for a in range(2)]
            contexts=[action_context(self.engine,self.meta,pieces[a],local[a]) for a in range(2)]
            key=(state.tobytes(),tuple((pieces[a]['task'],contexts[a]['phase']) if contexts[a] else None for a in range(2)))
            if key in self.cache:
                result=self.cache[key]
                if key not in fresh_keys:self.cache_hits+=1
                fresh_keys.discard(key)
            else:
                result=self.checker.probe(state,contexts)
                touching={a:result.get('tool_states',{}).get(self.checker.robot_ids[a],{}).get('touching',[]) for a in range(2)}
                # Retain verdicts and failure pairs, not bulky per-state tool certificates.
                result={k:result[k] for k in ('passed','reason','blocking_pairs') if k in result}
                result['touching']=touching
                self.cache[key]=result
            if not result['passed']:
                return dict(passed=False,samples=j+1,time=float(t),failure=result),None
            if fine:
                for a,piece in enumerate(pieces):
                    if piece is not None and piece['phase'] in ('ingress','retreat'):
                        context=self.meta[piece['task']]['context']
                        tcp=self.engine.world.models[context['robot_id']].tcp(local[a][:6])
                        travel=float((tcp[:3,3]-context['target'][:3,3])@context['direction'])
                        release_samples.setdefault((piece['task'],piece['phase']),[]).append((travel,result['touching'][a]))
            phases.append([p['phase'] if p else 'idle' for p in pieces])
            tasks.append([p['task'] or 0 if p else 0 for p in pieces])
            if fine and j%10000==0:print('TIMELINE_CHECK',angle,offsets,j,len(times),flush=True)
        for (task,phase),samples in release_samples.items():
            initial=self.meta[task]['context']['release_context']['initial_touching']
            if not release_history_passed(samples,initial,reverse=phase=='ingress'):
                return dict(passed=False,reason='source_release_contact_history_failed',task=task,phase=phase,samples=len(times)),None
        return dict(passed=True,samples=len(times),fine=fine),dict(times=times,states=states,phases=phases,tasks=tasks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-replay',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();started=time.perf_counter()
    if args.out.exists():raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    route,tasks,parks,sources=load_tasks(args.route)
    for p in HERE.glob('*.py'):sources[str(p.resolve())]=sha(p)
    from variable_world import VariableToolWorld
    from mixed_bridge_v95 import SignedSearchBridge
    from world_bridge_v2 import SourceBridge
    from state_space import interpolate
    from export_latest_route_v160 import action_sections
    from experimental.dual_arm_parallel_v1.checker import DualSourceChecker
    from experimental.dual_arm_parallel_v1.transfer import make_transfer
    from experimental.dual_arm_parallel_v1.source_pool import SourcePool,prefill_fine_cache
    from experimental.dual_arm_parallel_v1.offset_candidates import offset_candidates
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    engine=VariableToolWorld(scene);bridge=SignedSearchBridge(engine);checker=DualSourceChecker(engine)
    print('DUAL_WORLD_READY',round(time.perf_counter()-started,2),flush=True)
    meta={t['id']:t for t in tasks}
    for t in tasks:t['context']=prepare_context(engine,t)
    stages=align_batches([[t for t in tasks if t['arm']==a] for a in range(2)])
    pool=SourcePool(bridge,checker,args.workers)
    # Check real FCL/contact results from independent forked worlds against
    # the same sequential probes before accepting any accelerated verdicts.
    strict_samples=tasks[0]['states'][-5:]
    expected=[{k:r[k] for k in ('passed','reason','blocking_pairs')} for r in (bridge.probe(s) for s in strict_samples)]
    if pool.strict_states(strict_samples)!=expected:raise RuntimeError('Source pool strict equivalence failed')
    records=[]
    for state in tasks[0]['states'][:5]:
        arm=tasks[0]['arm'];contexts=[None,None]
        contexts[arm]=action_context(engine,meta,dict(task=tasks[0]['id'],phase='open'),np.r_[state[6*arm:6*arm+6],state[13+arm]])
        records.append((state,contexts))
    expected=[]
    for state,contexts in records:
        result=checker.probe(state,contexts)
        minimal={k:result[k] for k in ('passed','reason','blocking_pairs')}
        minimal['touching']={a:list(result.get('tool_states',{}).get(rid,{}).get('touching',[])) for a,rid in enumerate(checker.robot_ids)}
        expected.append(minimal)
    if pool.dual_states(records)!=expected:raise RuntimeError('Source pool contact equivalence failed')
    print('SOURCE_POOL_REAL_EQUIVALENCE_PASSED',len(strict_samples)+len(records),args.workers,flush=True)
    bridge.fine_prefill=lambda path:prefill_fine_cache(bridge,path,pool)
    validator=TimelineValidator(engine,checker,meta,pool)
    existing=[json.loads(Path(e['path_file']).read_text()) for e in route['selected_edges']]
    sources.update({e['path_file']:e['path_sha256'] for e in route['selected_edges']})
    initial=tasks[0]['states'][-1].copy();final=tasks[-1]['states'][-1].copy()
    baseline_duration=load_baseline(args.baseline_replay,route,initial,final,sources)
    all_times=[];all_states=[];all_phases=[];all_tasks=[];all_stage=[];stage_rows=[];transfers=[]
    absolute_time=0.;grouped_serial=0.;rotation_time=0.
    tcp_cache=[{},{}]
    def tcp(arm,states):
        result=[];model=engine.world.models[route['robot_ids'][arm]]
        for state in states:
            key=tuple(state[:6])
            if key not in tcp_cache[arm]:tcp_cache[arm][key]=model.tcp(state[:6])[:3,3]
            result.append(tcp_cache[arm][key])
        return np.array(result)
    def add_result(data,stage):
        nonlocal absolute_time
        start=1 if all_states else 0
        if all_states and not np.allclose(all_states[-1],data['states'][0],atol=1e-12,rtol=0):raise RuntimeError('Full timeline seam mismatch')
        all_times.extend(absolute_time+data['times'][start:]);all_states.extend(data['states'][start:])
        all_phases.extend(data['phases'][start:]);all_tasks.extend(data['tasks'][start:]);all_stage.extend([stage]*(len(data['times'])-start))
        absolute_time+=float(data['times'][-1])
    for si,stage in enumerate(stages):
        angle=stage['angle'];print('DUAL_STAGE_START',si,angle,[[t['id'] for t in arm] for arm in stage['tasks']],flush=True)
        if si:
            a=np.asarray(all_states[-1]);b=combine_state(parks[0],parks[1],angle)
            if not SourceBridge.edge_scaled(bridge,a/bridge.scale,b/bridge.scale):raise RuntimeError('Park rotation blocked '+str(bridge.last_failure))
            rotation=np.asarray(interpolate(a,b,bridge.resolutions));duration=abs(b[12]-a[12])/np.deg2rad(15.)
            add_result(dict(times=np.linspace(0.,duration,len(rotation)),states=rotation,
                            phases=[['rotate','rotate']]*len(rotation),tasks=[[0,0]]*len(rotation)),si)
            rotation_time+=duration;grouped_serial+=duration
        chains=[]
        for arm in range(2):
            start=np.r_[initial[6*arm:6*arm+6],initial[13+arm]] if si==0 else parks[arm]
            chain=new_chain(start)
            def transfer_to(goal):
                current=np.asarray(chain['states'][-1])
                if np.allclose(current,goal,atol=1e-12,rtol=0):return
                print('TRANSFER_START',si,arm,flush=True)
                path,stats=make_transfer(engine.world,bridge,arm,angle,current,goal,
                    existing_edges=existing,seed=2100+len(transfers),park7=parks[1-arm])
                if path is None:
                    save(args.out/'failure.json',dict(stage=si,arm=arm,transfer=stats))
                    raise RuntimeError('No source-certified rebuilt transfer')
                fine=np.asarray([s for k,(a,b) in enumerate(zip(path[:-1],path[1:]))
                    for s in (interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5]) if k==0 else interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5])[1:])])
                file=args.out/'transfers'/f'{len(transfers):02d}.json'
                save(file,dict(arm=arm,angle=angle,waypoints=path.tolist(),stats=stats))
                transfers.append(dict(arm=arm,angle=angle,file=str(file.resolve()),sha256=sha(file),stats=stats))
                append_piece(chain,fine,'transfer',tcp=tcp(arm,fine))
                print('TRANSFER_END',si,arm,len(fine),stats.get('method'),flush=True)
            for task in stage['tasks'][arm]:
                local=task['states'];own=local[:,np.r_[np.arange(arm*6,arm*6+6),13+arm]]
                transfer_to(own[-1])
                ingress,close,opening,retreat=action_sections(task['path'],own)
                for phase,section in [('ingress',ingress),('close',close),('weld',[close[-1],close[-1]]),('open',opening),('retreat',retreat)]:
                    section=np.asarray(section)
                    append_piece(chain,section,phase,task=task['id'],duration=.6 if phase=='weld' else None,tcp=tcp(arm,section))
            target=np.r_[final[6*arm:6*arm+6],final[13+arm]] if si==len(stages)-1 else parks[arm]
            transfer_to(target)
            chain['states']=np.asarray(chain['states']);chain['times']=np.asarray(chain['times']);chains.append(chain)
        durations=[float(c['times'][-1]) for c in chains]
        grouped_serial+=sum(durations)
        offsets=[(0.,0.)]
        for value in (.5,1.,2.,4.,8.,16.,32.,64.):
            if value<durations[1]:offsets.append((value,0.))
            if value<durations[0]:offsets.append((0.,value))
        offsets += [(0.,durations[0]),(durations[1],0.)]
        offsets=[r['offsets'] for r in offset_candidates(chains,base_offsets=offsets)]
        tries=[];accepted=None
        # First valid makespan in sorted candidates is the best among tested offsets.
        for offset in offsets:
            coarse,_=validator.validate(chains,angle,offset,False)
            row=dict(offsets=offset,coarse=coarse,makespan=max(offset[a]+durations[a] for a in range(2)))
            if coarse['passed']:
                verdict,data=validator.validate(chains,angle,offset,True);row['fine']=verdict
                if verdict['passed']:accepted=(offset,verdict,data)
            tries.append(row);print('OFFSET_CHECK',si,offset,coarse['passed'],accepted is not None,flush=True)
            if accepted:break
        if accepted is None:
            save(args.out/'failure.json',dict(stage=si,tries=tries))
            raise RuntimeError('No full source-certified schedule; no unsafe parallel fallback exported')
        offset,verdict,data=accepted;start_time=absolute_time
        add_result(data,si)
        row=dict(angle=angle,offsets=list(offset),durations=durations,pieces=[c['pieces'] for c in chains],
                 task_ids=[[t['id'] for t in arm] for arm in stage['tasks']],start_time=start_time,end_time=absolute_time,
                 simultaneous_weld_seconds=overlap_seconds(chains[0]['pieces'],chains[1]['pieces'],offset),
                 motion_overlap_seconds=max(0.,min(offset[0]+durations[0],offset[1]+durations[1])-max(offset)),
                 validation=verdict,offset_trials=tries)
        stage_rows.append(row);save(args.out/f'stage_{si:02d}.json',row)
        print('DUAL_STAGE_ACCEPTED',si,'SECONDS',data['times'][-1],'OFFSETS',offset,flush=True)
    states=np.asarray(all_states);times=np.asarray(all_times)
    if not np.allclose(states[0],initial,atol=1e-12,rtol=0) or not np.allclose(states[-1],final,atol=1e-12,rtol=0):raise RuntimeError('Original global endpoints changed')
    for p,h in sources.items():
        if sha(p)!=h:raise RuntimeError('Changed bound source after run '+p)
    np.savez_compressed(args.out/'trajectory.npz',states=states,time=times,phases=np.asarray(all_phases),tasks=np.asarray(all_tasks),stage=np.asarray(all_stage))
    report=dict(complete=True,candidate_policy='existing_selected_tasks_only',display_only=True,
        order_weld_ids=route['order_weld_ids'],robot_ids=route['robot_ids'],original_assignment=[t['arm'] for t in tasks],
        source_sha256=sources,trajectory_sha256=sha(args.out/'trajectory.npz'),stages=stage_rows,transfers=transfers,
        timing=dict(baseline_duration_s=baseline_duration,grouped_serial_duration_s=grouped_serial,parallel_duration_s=absolute_time,
                    rotation_duration_s=rotation_time,simultaneous_weld_seconds=sum(r['simultaneous_weld_seconds'] for r in stage_rows),
                    limits=dict(joint_deg_s=30.,positioner_deg_s=15.,driver_mm_s=80.,tcp_mm_s=200.,weld_seconds=.6),
                    timing_is_display_only=True,acceleration_jerk_or_controller_execution_not_certified=True),
        validation=dict(passed=True,source_checks=checker.checks,source_cache_hits=validator.cache_hits,transfer_source_checks=bridge.checks,
            source_process_equivalence_samples=len(strict_samples)+len(records),source_processes=args.workers,
            frames=len(states),source_resolution=dict(joint_deg=.05,positioner_deg=.25,driver_mm=.5),
            continuous_collision_certified=False,unchanged_all21_local_actions=True,unchanged_per_arm_task_order=True,
            max_global_endpoint_error=float(max(abs(states[0]-initial).max(),abs(states[-1]-final).max()))),
        elapsed_s=time.perf_counter()-started)
    save(args.out/'report.json',report)
    pool.close();pool.join()
    print('DUAL_COMPLETE',len(states),report['timing'],'ELAPSED',report['elapsed_s'],flush=True)


if __name__=='__main__':main()
