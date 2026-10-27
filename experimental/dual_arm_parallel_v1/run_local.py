"""Local dual-arm splice: retain baseline transfers, accept only a faster route."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_retreat_transfer_v1'),
              str(ROOT/'experimental/welding_variable_open_v1')]
from experimental.dual_arm_parallel_v1.run_parallel import (
    sha,save,load_tasks,load_baseline,prepare_context,TimelineValidator,overlap_seconds)
from experimental.dual_arm_parallel_v1.motion import new_chain,append_piece,section_times


def choose_improvement(candidates,baseline_duration):
    valid=[c for c in candidates if c.get('passed') and np.isfinite(c['duration'])
           and c['duration']<baseline_duration-1e-8]
    return min(valid,key=lambda c:c['duration']) if valid else None


def validate_action_coverage(data,assignment,initial,final):
    states=np.asarray(data['states']);times=np.asarray(data['time'])
    if not np.allclose(states[0],initial,atol=1e-12,rtol=0) or not np.allclose(states[-1],final,atol=1e-12,rtol=0):
        raise ValueError('Original global endpoint changed')
    if times[0]!=0 or not np.isfinite(states).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid physical states or timestamps')
    limits=np.r_[np.full(12,np.deg2rad(.05)),np.deg2rad(.25),.5,.5]
    groups=np.asarray(data.get('stage',np.zeros(len(states),int)))
    fresh=np.isin(groups[1:],[0,2])
    if np.any(abs(np.diff(states,axis=0))[fresh]>limits+2e-10):raise ValueError('Fine source axis steps exceeded')
    required=['ingress','close','weld','open','retreat']
    for arm in (0,1):
        seen=[]
        for task in (t for t in range(1,22) if assignment[t-1]==arm):
            mask=np.asarray(data['tasks'])[:,arm]==task
            labels=np.asarray(data['phases'])[mask,arm]
            runs=labels[np.r_[True,labels[1:]!=labels[:-1]]] if len(labels) else labels
            if list(runs)!=required:raise ValueError('Missing, duplicate or reordered original action phases')
        other=(np.asarray(data['tasks'])[:,arm]>0)
        if any(assignment[int(t)-1]!=arm for t in np.asarray(data['tasks'])[other,arm]):
            raise ValueError('Original assignment changed')
        weld=np.asarray(data['tasks'])[np.asarray(data['phases'])[:,arm]=='weld',arm]
        seen=list(weld[np.r_[True,weld[1:]!=weld[:-1]]]) if len(weld) else []
        if seen!=[t for t in range(1,22) if assignment[t-1]==arm]:raise ValueError('Original weld order changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route',type=Path,required=True)
    parser.add_argument('--baseline-replay',type=Path,required=True)
    parser.add_argument('--prior-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();tick=time.perf_counter()
    if args.out.exists():raise FileExistsError('Fresh output required')
    args.out.mkdir(parents=True)
    from experimental.dual_arm_parallel_v1.splice import arm_chain_from_baseline,serial_part,stitch_parts
    from experimental.dual_arm_parallel_v1.shortcut import shortcut
    from experimental.dual_arm_parallel_v1.source_pool import SourcePool,prefill_fine_cache
    from experimental.dual_arm_parallel_v1.checker import DualSourceChecker
    from experimental.dual_arm_parallel_v1.offset_candidates import offset_candidates
    from state_space import interpolate
    from variable_world import VariableToolWorld
    from mixed_bridge_v95 import SignedSearchBridge
    from world_bridge_v2 import SourceBridge
    from export_latest_route_v160 import action_sections
    route,tasks,parks,sources=load_tasks(args.route)
    for file in HERE.glob('*.py'):sources[str(file.resolve())]=sha(file)
    initial=tasks[0]['states'][-1];final=tasks[-1]['states'][-1]
    baseline_duration=load_baseline(args.baseline_replay,route,initial,final,sources)
    with np.load(args.baseline_replay,allow_pickle=False) as archive:
        baseline={k:archive[k].copy() for k in archive.files}
    assignment=[t['arm'] for t in tasks]
    if assignment!=[1]*6+[0]*15 or tasks[0]['angle']!=30 or tasks[1]['angle']!=30 or tasks[6]['angle']!=30:
        raise ValueError('Local pilot requires this exact selected route')
    prior_report=args.prior_run/'report.json';prior=json.loads(prior_report.read_text())
    if not prior['complete'] or not prior['validation']['passed']:raise ValueError('Accepted prior transfer seed required')
    sources[str(prior_report.resolve())]=sha(prior_report)
    scene_file=ROOT/'build/scene_placement_corrected_v1.json'
    engine=VariableToolWorld(json.loads(scene_file.read_text()));bridge=SignedSearchBridge(engine)
    checker=DualSourceChecker(engine);meta={t['id']:t for t in tasks}
    for task in tasks:task['context']=prepare_context(engine,task)
    pool=SourcePool(bridge,checker,args.workers)
    validator=TimelineValidator(engine,checker,meta,pool)
    cache=[{},{}]
    def tcp(arm,states):
        model=engine.world.models[route['robot_ids'][arm]];values=[]
        for state in states:
            key=tuple(state[:6])
            if key not in cache[arm]:cache[arm][key]=model.tcp(state[:6])[:3,3]
            values.append(cache[arm][key])
        return np.asarray(values)
    r1=new_chain(parks[0]);transfer_rows=[]
    for number in (0,1):
        item=prior['transfers'][number];file=Path(item['file'])
        if item['arm']!=0 or item['angle']!=30 or sha(file)!=item['sha256']:
            raise ValueError('Prior transfer identity differs')
        transfer=json.loads(file.read_text());path=np.asarray(transfer['waypoints'])
        if not transfer['stats']['fine_certified']:raise ValueError('Prior transfer was not fine accepted')
        sources[str(file.resolve())]=item['sha256'];transfer_rows.append(dict(file=str(file.resolve()),sha256=item['sha256']))
        dense=np.vstack([np.asarray(interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5]))[0 if k==0 else 1:]
                         for k,(a,b) in enumerate(zip(path[:-1],path[1:]))])
        append_piece(r1,dense,'transfer',tcp=tcp(0,dense))
        if number==0:
            own=tasks[6]['states'][:,np.r_[np.arange(6),13]]
            ingress,close,opening,retreat=action_sections(tasks[6]['path'],own)
            for phase,section in [('ingress',ingress),('close',close),('weld',[close[-1],close[-1]]),('open',opening),('retreat',retreat)]:
                section=np.asarray(section)
                append_piece(r1,section,phase,task=7,duration=.6 if phase=='weld' else None,tcp=tcp(0,section))
    r1['states']=np.asarray(r1['states']);r1['times']=np.asarray(r1['times'])
    r2=arm_chain_from_baseline(baseline,1,assignment,1,2);chains=[r1,r2]
    durations=[float(c['times'][-1]) for c in chains]
    print('LOCAL_CHAINS_READY',durations,'ELAPSED',time.perf_counter()-tick,flush=True)
    # Validate process equivalence on real source/contact contexts before pooling.
    records=[(initial,[None,None])]
    expected=checker.probe(initial,[None,None])
    actual=pool.dual_states(records)[0]
    if actual['passed']!=expected['passed'] or actual['blocking_pairs']!=expected.get('blocking_pairs',[]):
        raise RuntimeError('Fork source equivalence differs')
    # Remove action7 from its old position. Retain original two accepted
    # transfer polylines as a safe seed and propose only fine-checked shortcuts.
    edges=[json.loads(Path(route['selected_edges'][i]['path_file']).read_text()) for i in (5,6)]
    for i in (5,6):sources[route['selected_edges'][i]['path_file']]=route['selected_edges'][i]['path_sha256']
    first_path=np.asarray(edges[0]['waypoints']);second_path=np.asarray(edges[1]['waypoints'])
    if not np.array_equal(first_path[-1],second_path[0]):raise ValueError('Original transfer internal seam differs')
    waypoints=np.vstack([first_path,second_path[1:]])
    cost_cache={}
    def dense_path(path):
        return np.vstack([np.asarray(interpolate(a,b,bridge.resolutions))[0 if k==0 else 1:]
                          for k,(a,b) in enumerate(zip(path[:-1],path[1:]))])
    def edge_cost(a,b):
        key=(a.tobytes(),b.tobytes())
        if key not in cost_cache:
            s=np.asarray(interpolate(a,b,bridge.resolutions))
            dts=[]
            for arm in (0,1):
                own=s[:,np.r_[np.arange(6*arm,6*arm+6),13+arm]]
                dts.append(np.diff(section_times(own,tcp(arm,own))))
            dt=np.maximum(dts[0],dts[1]);dt=np.maximum(dt,abs(np.diff(s[:,12]))/np.deg2rad(15.))
            cost_cache[key]=float(sum(dt))
        return cost_cache[key]
    def cost(path):return sum(edge_cost(a,b) for a,b in zip(path[:-1],path[1:]))
    trial_number=0
    def certify(a,b):
        nonlocal trial_number
        trial_number+=1
        scaled=np.asarray([a,b])/bridge.scale
        if not bridge.edge_scaled(*scaled):
            print('LOCAL_SHORTCUT',trial_number,'COARSE_REJECT',flush=True);return False
        prefill_fine_cache(bridge,scaled,pool)
        passed=SourceBridge.edge_scaled(bridge,*scaled)
        print('LOCAL_SHORTCUT',trial_number,'FINE',passed,flush=True)
        return passed
    fast_path,shortcut_stats=shortcut(waypoints,certify,cost,max_trials=60)
    fast_states=dense_path(fast_path)
    fast_times=[0.]
    for arm in (0,1):
        own=fast_states[:,np.r_[np.arange(6*arm,6*arm+6),13+arm]]
        fast_times.append(np.diff(section_times(own,tcp(arm,own))))
    dt=np.maximum(fast_times[1],fast_times[2]);dt=np.maximum(dt,abs(np.diff(fast_states[:,12]))/np.deg2rad(15.))
    connector=dict(time=np.r_[0.,np.cumsum(dt)],states=fast_states,
                   phases=np.full((len(fast_states),2),'transfer'),tasks=np.zeros((len(fast_states),2),int))
    save(args.out/'connector.json',dict(waypoints=fast_path.tolist(),stats=shortcut_stats))
    transfer_rows.append(dict(file=str((args.out/'connector.json').resolve()),sha256=sha(args.out/'connector.json')))
    print('LOCAL_CONNECTOR_SECONDS',connector['time'][-1],shortcut_stats,flush=True)
    def indices(task,phase=None):
        mask=baseline['tasks']==task
        if phase is not None:mask&=baseline['phases']==phase
        else:mask&=baseline['phases']!='transfer'
        return np.flatnonzero(mask)
    end2=int(indices(2,'retreat')[-1]);end6=int(indices(6,'retreat')[-1])
    start8=int(indices(8,'ingress')[0])-1
    middle=serial_part(baseline,assignment,end2,end6)
    suffix=serial_part(baseline,assignment,start8,len(baseline['times'])-1)
    for part in (middle,suffix):part['time']=part.pop('times')
    fixed_duration=float(middle['time'][-1]+connector['time'][-1]+suffix['time'][-1])
    offsets=[(0.,0.)]+[(float(k),0.) for k in range(1,31)]+[(0.,float(k)) for k in range(1,31)]
    offsets=[tuple(r['offsets']) for r in offset_candidates(chains,base_offsets=offsets)]
    trials=[];accepted=[];best_weld=None
    for offset in offsets:
        duration=max(offset[a]+durations[a] for a in (0,1))+fixed_duration
        overlap=overlap_seconds(r1['pieces'],r2['pieces'],offset)
        # No collision work can rescue a candidate that already loses in time.
        if duration>=baseline_duration-1e-8:continue
        if accepted and duration>=accepted[0]['duration']-1e-8 and (overlap<.599999 or best_weld is not None):continue
        coarse,_=validator.validate(chains,30,offset,False)
        row=dict(offsets=list(offset),duration=duration,simultaneous_weld_seconds=overlap,coarse=coarse,passed=False)
        if coarse['passed']:
            verdict,data=validator.validate(chains,30,offset,True);row.update(passed=verdict['passed'],fine=verdict)
            if verdict['passed']:
                entry=dict(row,data=data)
                accepted.append(entry);accepted.sort(key=lambda r:r['duration'])
                if overlap>=.599999:best_weld=entry
        trials.append(row);print('LOCAL_OFFSET',offset,'TOTAL',duration,'WELD_OVERLAP',overlap,'PASSED',row['passed'],flush=True)
        if accepted and best_weld is not None:break
    save(args.out/'search.json',dict(baseline_duration_s=baseline_duration,shortcut=shortcut_stats,trials=trials))
    best=choose_improvement(accepted,baseline_duration)
    if best is None:
        pool.close();pool.join();raise RuntimeError('No faster source-accepted splice; original baseline retained')
    for name,entry in [('fastest',best),('simultaneous-weld',best_weld)]:
        if entry is None:continue
        prefix=dict(entry['data']);prefix['time']=prefix.pop('times')
        parts=[prefix,middle,connector,suffix];combined=stitch_parts(parts)
        validate_action_coverage(combined,assignment,initial,final)
        out=args.out/name;out.mkdir()
        np.savez_compressed(out/'trajectory.npz',**combined)
        # Generate contiguous report pieces, retaining exact baseline action
        # start/end times; transfer stages may move the common positioner.
        stages=[];absolute=0.
        for pi,part in enumerate(parts):
            times=part['time'];duration=float(times[-1])
            if pi==0:
                pieces=[c['pieces'] for c in chains];stage_durations=durations;stage_offsets=entry['offsets'];angle=30
            else:
                # Split serial mixed-angle part into stages at each run of
                # phase/task, so each original action retains its true angle.
                continue
            stages.append(dict(angle=angle,offsets=stage_offsets,durations=stage_durations,pieces=pieces,
                task_ids=[[p['task'] for p in pieces[a] if p['phase']=='weld'] for a in (0,1)],
                start_time=absolute,end_time=absolute+duration,simultaneous_weld_seconds=entry['simultaneous_weld_seconds'],validation=entry['fine']))
            absolute+=duration
        # Serial tail phases use the destination sample label, matching the
        # sealed baseline exporter. Each section includes its predecessor.
        for part in parts[1:]:
            cuts=np.r_[1,np.flatnonzero(np.any(part['phases'][2:]!=part['phases'][1:-1],axis=1)|
                           np.any(part['tasks'][2:]!=part['tasks'][1:-1],axis=1))+2,len(part['time'])]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                start=float(part['time'][lo-1]);end=float(part['time'][hi-1]);duration=end-start
                phases=part['phases'][lo];labels=part['tasks'][lo];active=next((a for a in (0,1) if int(labels[a])>0),0)
                pieces=[[],[]];phase=str(phases[active]);task=int(labels[active]) or None
                pieces[active]=[dict(start=0.,end=duration,phase=phase,task=task)]
                stage_durations=[0.,0.];stage_durations[active]=duration
                angle=tasks[task-1]['angle'] if task is not None else float(np.rad2deg(part['states'][lo,12]))
                stages.append(dict(angle=angle,offsets=[0.,0.],durations=stage_durations,
                    pieces=pieces,task_ids=[[task] if a==active and phase=='weld' else [] for a in (0,1)],
                    start_time=absolute,end_time=absolute+duration,simultaneous_weld_seconds=0.))
                absolute+=duration
        bound=dict(sources)
        report=dict(complete=True,candidate_policy='existing_selected_tasks_only',display_only=True,
            transfer_policy='baseline_local_splice',order_weld_ids=route['order_weld_ids'],robot_ids=route['robot_ids'],
            original_assignment=assignment,source_sha256=bound,trajectory_sha256=sha(out/'trajectory.npz'),stages=stages,
            transfers=transfer_rows,splice=dict(prefix_tasks=[[7],[1,2]],removed_old_task=7,
                retained_baseline_spans=[[end2,end6],[start8,len(baseline['times'])-1]],
                shortcut=shortcut_stats,offsets=entry['offsets']),
            timing=dict(baseline_duration_s=baseline_duration,parallel_duration_s=float(combined['time'][-1]),
                grouped_serial_duration_s=sum(durations)+fixed_duration,rotation_duration_s=0.,
                simultaneous_weld_seconds=entry['simultaneous_weld_seconds'],timing_is_display_only=True,
                limits=dict(joint_deg_s=30.,positioner_deg_s=15.,driver_mm_s=80.,tcp_mm_s=200.,weld_seconds=.6),
                acceleration_jerk_or_controller_execution_not_certified=True),
            validation=dict(passed=True,source_checks=checker.checks,transfer_source_checks=bridge.checks,
                frames=len(combined['time']),source_resolution=dict(joint_deg=.05,positioner_deg=.25,driver_mm=.5),
                fine_axis_steps_scope='new parallel prefix and rebuilt connector; retained baseline action knots remain unchanged',
                continuous_collision_certified=False,unchanged_all21_local_actions=True,unchanged_per_arm_task_order=True,
                retained_segments_acceptance='Exact sealed baseline state/time identity; inherited source acceptance',
                stage_index_semantics='splice parts: 0 prefix, 1 retained middle, 2 connector, 3 retained suffix; report stages are action intervals',
                max_global_endpoint_error=float(max(abs(combined['states'][0]-initial).max(),abs(combined['states'][-1]-final).max()))),
            elapsed_s=time.perf_counter()-tick)
        if report['timing']['parallel_duration_s']>=baseline_duration:raise RuntimeError('Slower output refused')
        for file,h in bound.items():
            if sha(file)!=h:raise RuntimeError('Changed bound source '+file)
        save(out/'report.json',report)
        print('LOCAL_COMPLETE',name,report['timing'],flush=True)
    pool.close();pool.join()

if __name__=='__main__':main()
