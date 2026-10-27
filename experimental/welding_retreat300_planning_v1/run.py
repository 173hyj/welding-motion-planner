"""Fresh 300 mm all-candidate joint order selection and certified batch scheduling."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).absolute().parents[2]
ENGINE=None
ROBOT_IDS=None
PARKS=None
STAGES=[(30,[[7],[1,2]]),(0,[[10,11],[]]),(60,[[8,9],[]]),(120,[[12,13,14,15,16,17,18,19,20],[3,4,5,6]]),(90,[[21],[]])]
CURRENT_SOLUTION=None


def acceptance(before,after):
    reasons=[]
    if after['duration_s']>before['duration_s']+1e-7:reasons.append('display_time_regression')
    if after['rotation_episodes']>before['rotation_episodes']:reasons.append('extra_positioner_rotation')
    return dict(accepted=not reasons,reasons=reasons)


def merge_bindings(*groups):
    result={}
    for group in groups:
        for path,sha in group.items():
            if path in result and result[path]!=sha:raise ValueError('Conflicting inherited binding: '+path)
            result[path]=sha
    return result


def local_metrics(states,times):
    states=np.asarray(states);delta=abs(np.diff(states,axis=0));duration=float(times[-1])
    joint=float(np.rad2deg(delta[:,:6]).sum());wrist=float(np.rad2deg(delta[:,3:6]).sum())
    return dict(cost=float(np.deg2rad(joint+.5*wrist)+.05*duration),duration_s=duration,joint_travel_deg=joint,wrist_travel_deg=wrist)


def endpoint_cost(a,b):
    d=abs(np.asarray(b)-np.asarray(a));seconds=max(float(d[:6].max()/np.deg2rad(30)),float(d[6]/80.))
    return float(d[:6].sum()+.5*d[3:6].sum()+.05*seconds)


def vector(row):
    return np.asarray([row['duration_s'],row['joint_travel_deg'],row['wrist_travel_deg']])


def lower_bound(a,b):
    d=abs(np.asarray(b)-np.asarray(a))
    return np.asarray([max(float(d[:6].max()/np.deg2rad(30)),float(d[6]/80.)),float(np.rad2deg(d[:6]).sum()),float(np.rad2deg(d[3:6]).sum())])


def build_graph(task_layers,nodes,records,arm_orders,starts,goals,stage):
    global CURRENT_SOLUTION
    from .graph import grouped_open_path
    edgekeys=set();ordered=[];indices=[];total=np.zeros(3)
    def transfer(arm,left,right,a,b):
        key=(stage,arm,left,right);edgekeys.add(key);row=records.get(key)
        return (vector(row) if row['passed'] else np.full(3,np.inf)) if row is not None else lower_bound(a,b)
    for arm,original in enumerate(arm_orders):
        if not original:continue
        tasks=sorted(original)
        nodecost=[np.asarray([vector(x) for x in nodes[t]]) for t in tasks]
        startcost=[np.asarray([transfer(arm,'start',x['id'],starts[arm],x['endpoint']) for x in nodes[t]]) for t in tasks]
        endcost=[np.asarray([transfer(arm,x['id'],'end',x['endpoint'],goals[arm]) for x in nodes[t]]) for t in tasks]
        edges={(a,b):np.asarray([[transfer(arm,x['id'],y['id'],x['endpoint'],y['endpoint']) for y in nodes[tb]] for x in nodes[ta]]) for a,ta in enumerate(tasks) for b,tb in enumerate(tasks) if a!=b}
        answer=grouped_open_path(nodecost,edges,startcost,endcost)
        if answer is None:
            CURRENT_SOLUTION=None
            return [],[],np.array([]),np.array([]),edgekeys
        order=[tasks[i] for i in answer['order']]
        arm_orders[arm][:]=order
        ordered.extend(order);indices.extend(answer['indices']);total+=answer['cost']
    CURRENT_SOLUTION=dict(indices=indices,cost=total.tolist())
    return ordered,[],np.array([]),np.array([]),edgekeys


def shortest_path_avoiding(*args,forbidden_paths=(),**kwargs):
    # Schedule rejection is unresolved, rather than wrongly removing a node/edge.
    return None if forbidden_paths else CURRENT_SOLUTION


def selected_transfers(ordered,indices,nodes,arm_orders,starts,goals,stage):
    chosen=dict(zip(ordered,indices));result=[]
    for arm,order in enumerate(arm_orders):
        previous='start';state=starts[arm]
        for task in order:
            node=nodes[task][chosen[task]]
            result.append(((stage,arm,previous,node['id']),state,node['endpoint']))
            previous=node['id'];state=node['endpoint']
        result.append(((stage,arm,previous,'end'),state,goals[arm]))
    return result


def transfer_worker(job):
    key,start,goal,angle,seconds,path=job
    from world_bridge_v2 import SourceBridge
    from state_space import interpolate
    from experimental.welding_candidate_selection_v1.pool import atomic,digest
    from experimental.dual_arm_parallel_v1.motion import section_times
    from .seeded_transfer import make_transfer
    previous=None
    if previous is not None:
        if digest(previous['file'])!=previous['sha256']:raise ValueError('Sealed previous edge changed')
        states=np.load(previous['file'],allow_pickle=False)['states']
        if np.array_equal(states[[0,-1]],[start,goal]):
            row=dict(previous,stats=dict(previous['stats'],reused_sealed_edge=True))
            atomic(Path(path).with_suffix('.json'),row)
            return key,row
    deadline=time.perf_counter()+seconds
    class Expired(RuntimeError):pass
    class ProposalBridge(SourceBridge):
        def edge_scaled(self,a,b):
            if time.perf_counter()>deadline:raise Expired()
            resolution=self.resolutions/self.scale
            resolution[:12]=np.deg2rad(.5)
            for sample in interpolate(a,b,resolution):
                if time.perf_counter()>deadline:raise Expired()
                if not self.valid_scaled(sample):return False
            return True
    bridge=ProposalBridge(ENGINE)
    arm=key[1]
    try:
        corners,stats=make_transfer(ENGINE.world,bridge,arm,angle,start,goal,
            existing_edges=(),seed=17003+sum(ord(c) for c in str(key)),park7=PARKS[1-arm],max_iterations=1800)
    except Expired:
        corners=None;stats=dict(fine_certified=False,reason='finite_coarse_proposal_budget_expired')
    row=dict(key=list(key),passed=corners is not None,stats=stats)
    if corners is not None:
        dense=[]
        for a,b in zip(corners,corners[1:]):
            samples=interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5])
            dense.extend(samples if not dense else samples[1:])
        states=np.asarray(dense) if dense else corners.copy();states[[0,-1]]=[start,goal]
        tcp=np.array([ENGINE.world.models[ROBOT_IDS[arm]].tcp(q[:6])[:3,3] for q in states])
        times=section_times(states,tcp)
        np.savez_compressed(path,states=states,times=times,tcp=tcp)
        row.update(file=str(path),sha256=digest(path),**local_metrics(states,times))
        stats.update(status='source_fine_certified',strategy=stats['method'],source_resolution_deg=.05,proposal_resolution_deg=.5)
    else:stats.update(status='unresolved',fine_certified=False)
    atomic(Path(path).with_suffix('.json'),row)
    return key,row


def optimize(pool_folder,out,incumbent_folder,workers=8,iterations=12,edge_seconds=60,reference_route=None,reference_sha256=None):
    global ENGINE,ROBOT_IDS,PARKS
    from experimental.welding_candidate_selection_v1.pool import atomic,digest
    from experimental.welding_candidate_selection_v1.run import verify_bindings
    from experimental.welding_candidate_selection_v1.trajectory import motion_metrics,combine
    if workers<1 or iterations<1 or edge_seconds<=0 or not np.isfinite(edge_seconds):raise ValueError('Invalid finite budget')
    pool_file=Path(pool_folder)/'report.json';pool=json.loads(pool_file.read_text());out=Path(out).resolve();incumbent_folder=Path(incumbent_folder)
    if out.exists():raise FileExistsError('Fresh output required')
    if not pool.get('accepted') or not pool.get('complete') or len(pool['layers'])!=21:raise ValueError('Complete all21 pool required')
    coverage=pool.get('coverage_all_rows',{})
    if coverage.get('registered')!=784 or coverage.get('attempted')!=784 or coverage.get('pending')!=0 or not pool.get('source_assets_frozen_before_and_after'):raise ValueError('Completed 784-row frozen-source 300mm library required')
    if any(a.get('length_mm')!=300. for l in pool['layers'] for a in l['actions']):raise ValueError('Only freshly generated 300mm library actions accepted')
    inc_file=incumbent_folder/'report.json';inc_report=json.loads(inc_file.read_text());inc_route=incumbent_folder/'route.npz';audit_file=incumbent_folder/'artifact-audit.json'
    if not inc_report['accepted'] or not inc_route.exists():raise ValueError('Accepted incumbent required')
    inherited=[inc_report['manifest']['bindings'],pool['manifest'].get('bindings',{})]
    for group in inherited:verify_bindings(group)
    bindings=merge_bindings(*inherited,{str(pool_file.resolve()):digest(pool_file),str(inc_file.resolve()):digest(inc_file),str(inc_route.resolve()):digest(inc_route)})
    if not audit_file.exists():raise ValueError('Sealed incumbent artifact audit required')
    audit=json.loads(audit_file.read_text());bindings[str(audit_file.resolve())]=digest(audit_file)
    if not audit.get('passed') or not audit.get('accepted'):raise ValueError('Incumbent artifact audit failed')
    if audit['route_sha256']!=digest(inc_route) or audit['result_report_sha256']!=digest(inc_file):raise ValueError('Incumbent differs from sealed artifact audit')
    for folder in (Path(__file__).parent,ROOT/'experimental/welding_full_candidate_selection_v1',ROOT/'experimental/dual_arm_parallel_v1'):
        for path in folder.glob('*.py'):bindings[str(path.resolve())]=digest(path)
    for layer in pool['layers']:
        for action in layer['actions']:bindings[action['path_file']]=action['path_sha256']
    verify_bindings(bindings)
    old=dict(np.load(inc_route,allow_pickle=False))
    from . import seeded_transfer
    seeded_transfer.OLD_STATES=old['states'];seeded_transfer.OLD_PHASES=old['phases']
    seeded_transfer.REFERENCES=[]
    if (reference_route is None)!=(reference_sha256 is None):raise ValueError('Reference route and expected SHA256 required together')
    if reference_route is not None:
        reference_route=Path(reference_route).resolve()
        if digest(reference_route)!=reference_sha256:raise ValueError('Additional reference SHA256 mismatch')
        bindings=merge_bindings(bindings,{str(reference_route):reference_sha256})
        reference=np.load(reference_route,allow_pickle=False)
        seeded_transfer.REFERENCES.append((reference['states'],reference['phases']))
    oldtime=old.get('time',old.get('times'));before=motion_metrics(old['states'],oldtime)
    sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_retreat_transfer_v1'),str(ROOT/'experimental/welding_variable_open_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    from world_bridge_v2 import SourceBridge
    from state_space import interpolate
    from experimental.dual_arm_parallel_v1.motion import new_chain,append_piece,section_times
    from experimental.dual_arm_parallel_v1.run_parallel import TimelineValidator,prepare_context,action_context
    from experimental.dual_arm_parallel_v1.source_pool import SourcePool
    from experimental.dual_arm_parallel_v1.offset_candidates import offset_candidates
    from experimental.welding_full_candidate_selection_v1.raw_checker import RawDualSourceChecker
    initialize_worker();ENGINE=load_raw_world(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()),SIGNS)
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    ROBOT_IDS=pool['robot_ids'];PARKS=[None,None];task_layers={x['task']:x for x in pool['layers']};nodes={}
    for layer in pool['layers']:
        group=[]
        for index,action in enumerate(layer['actions']):
            path=json.loads(Path(action['path_file']).read_text())
            from integrated_retreat_v117 import check_path_contract
            check_path_contract(path,action['initial'],300.)
            arm=layer['arm'];other=1-arm;park=np.r_[path['other_q_rad'],0.]
            if PARKS[other] is not None and not np.array_equal(PARKS[other],park):raise ValueError('Canonical passive park differs between actions')
            PARKS[other]=park
            states=np.asarray(local_states(path,ROBOT_IDS));own=states[:,np.r_[np.arange(arm*6,arm*6+6),13+arm]]
            sections=action_sections(path,own);action_chain=new_chain(own[-1]);pieces=[]
            for phase,section in zip(('ingress','close','weld','open','retreat'),(sections[0],sections[1],[sections[1][-1]]*2,sections[2],sections[3])):
                section=np.asarray(section);tcp=np.array([ENGINE.world.models[ROBOT_IDS[arm]].tcp(q[:6])[:3,3] for q in section])
                append_piece(action_chain,section,phase,task=layer['task'],duration=.6 if phase=='weld' else None,tcp=tcp);pieces.append((phase,section,tcp))
            group.append(dict(id=action['id'],action=action,path=path,pieces=pieces,endpoint=own[-1].copy(),**local_metrics(action_chain['states'],action_chain['times'])))
        nodes[layer['task']]=group
    if any(park is None for park in PARKS):raise ValueError('Missing canonical parks')
    if not np.allclose(old['states'][[0,-1],12],np.deg2rad([30.,90.]),atol=1e-10,rtol=0):raise ValueError('Expected original 30 degree initial and 90 degree final states')
    bridge=SourceBridge(ENGINE);checker=RawDualSourceChecker(ENGINE);records={};rounds=[];stage_rows=[];chosen_all={};participation=[];candidate_pieces=[];unresolved=[]
    out.mkdir();(out/'edges').mkdir();initial=old['states'][0];final=old['states'][-1];current=initial.copy()
    for si,(angle,orders) in enumerate(STAGES):
        starts=[np.r_[initial[6*a:6*a+6],initial[13+a]] if si==0 else PARKS[a] for a in range(2)]
        goals=[np.r_[final[6*a:6*a+6],final[13+a]] if si==len(STAGES)-1 else PARKS[a] for a in range(2)]
        ordered,_,_,_,edgekeys=build_graph(task_layers,nodes,{},orders,starts,goals,si)
        participation.append(dict(stage=si,angle=angle,layers=[dict(task=t,arm=task_layers[t]['arm'],count=len(nodes[t]),action_ids=[n['id'] for n in nodes[t]]) for t in ordered],potential_transfer_edges=len(edgekeys),all_nodes_in_initial_graph=True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as executor,SourcePool(bridge,checker,workers) as source_pool:
        strict=np.asarray([initial,final]);expected=[]
        for s in strict:
            r=bridge.probe(s);expected.append(dict(passed=bool(r['passed']),reason=r.get('reason',''),blocking_pairs=r.get('blocking_pairs',[])))
        if source_pool.strict_states(strict)!=expected:raise RuntimeError('Fork source strict equivalence failed')
        for si,(angle,orders) in enumerate(STAGES):
            print('FULL_STAGE_START',si,angle,orders,flush=True)
            if si:
                target=current.copy();target[12]=np.deg2rad(angle)
                if not bridge.edge_scaled(current/bridge.scale,target/bridge.scale):unresolved.append(dict(stage=si,reason='canonical_park_rotation_source_rejected'));break
                states=np.asarray(interpolate(current,target,bridge.resolutions));dt=abs(target[12]-current[12])/np.deg2rad(15.)
                candidate_pieces.append(dict(states=states,time=np.linspace(0.,dt,len(states)),phases=np.full((len(states),2),'rotate'),tasks=np.zeros((len(states),2),int),stage=np.full(len(states),si,int)))
                current=target
            starts=[np.r_[current[6*a:6*a+6],current[13+a]] for a in range(2)]
            goals=[np.r_[final[6*a:6*a+6],final[13+a]] if si==len(STAGES)-1 else PARKS[a] for a in range(2)]
            forbidden=[];accepted=None
            for iteration in range(iterations):
                idle_failed=[arm for arm,order in enumerate(orders) if not order and (si,arm,'start','end') in records and not records[si,arm,'start','end']['passed']]
                if idle_failed:
                    unresolved.append(dict(stage=si,reason='finite_idle_arm_transfer_unresolved',arms=idle_failed));break
                ordered,costs,sc,ec,edgekeys=build_graph(task_layers,nodes,records,orders,starts,goals,si)
                # Participation for all six complete graphs was recorded before bounded source search.
                solution=shortest_path_avoiding([nodes[t] for t in ordered],costs,sc,ec,forbidden_paths=forbidden)
                if solution is None:unresolved.append(dict(stage=si,reason='no_route_after_finite_search_failures_and_schedule_rejections'));break
                selected=selected_transfers(ordered,solution['indices'],nodes,orders,starts,goals,si)
                missing=[(key,a,b) for key,a,b in selected if key not in records]
                row=dict(stage=si,iteration=iteration,solution=solution,new_edges=[list(k) for k,a,b in missing]);rounds.append(row);atomic(out/'progress.json',dict(rounds=rounds,stages=stage_rows))
                if missing:
                    jobs=[(key,a,b,angle,edge_seconds,out/'edges'/('e'+str(len(records)+i).zfill(5)+'.npz')) for i,(key,a,b) in enumerate(missing)]
                    for key,record in executor.map(transfer_worker,jobs):records[key]=record;print('FULL_EDGE',key,record['passed'],record.get('duration_s'),flush=True)
                    continue
                if any(not records[key]['passed'] for key,a,b in selected):raise RuntimeError('Failed edge escaped graph')
                choices=dict(zip(ordered,solution['indices']));chains=[];meta={}
                for arm,order in enumerate(orders):
                    chain=new_chain(starts[arm]);previous='start'
                    for task in order:
                        node=nodes[task][choices[task]];transfer=records[si,arm,previous,node['id']];data=np.load(transfer['file']);append_piece(chain,data['states'],'transfer',tcp=data['tcp'])
                        for phase,section,tcp in node['pieces']:append_piece(chain,section,phase,task=task,duration=.6 if phase=='weld' else None,tcp=tcp)
                        meta[task]=dict(id=task,path=node['path'],angle=angle,arm=arm,weld_id=task_layers[task]['weld_id'])
                        meta[task]['context']=prepare_context(ENGINE,meta[task]);previous=node['id']
                    data=np.load(records[si,arm,previous,'end']['file']);append_piece(chain,data['states'],'transfer',tcp=data['tcp']);chain['states']=np.asarray(chain['states']);chain['times']=np.asarray(chain['times']);chains.append(chain)
                equivalence_records=[]
                for task in ordered[:2]:
                    arm=task_layers[task]['arm'];node=nodes[task][choices[task]];state=np.asarray(local_states(node['path'],ROBOT_IDS))[0];contexts=[None,None]
                    contexts[arm]=action_context(ENGINE,meta,dict(task=task,phase='open'),np.r_[state[6*arm:6*arm+6],state[13+arm]])
                    equivalence_records.append((state,contexts))
                expected_dual=[]
                for state,contexts in equivalence_records:
                    result=checker.probe(state,contexts);minimal={k:result[k] for k in ('passed','reason','blocking_pairs')};minimal['touching']={a:list(result.get('tool_states',{}).get(rid,{}).get('touching',[])) for a,rid in enumerate(ROBOT_IDS)};expected_dual.append(minimal)
                if source_pool.dual_states(equivalence_records)!=expected_dual:raise RuntimeError('Fork source contact equivalence failed')
                validator=TimelineValidator(ENGINE,checker,meta,source_pool);durations=[float(c['times'][-1]) for c in chains]
                base=[(0.,0.),(0.,durations[0]),(durations[1],0.)]
                for shift in (.5,1.,2.,4.,8.,16.,32.,64.):
                    if shift<durations[0]:base.append((0.,shift))
                    if shift<durations[1]:base.append((shift,0.))
                tries=[]
                for record in offset_candidates(chains,base_offsets=base):
                    offsets=record['offsets'];coarse,_=validator.validate(chains,angle,offsets,False);attempt=dict(offsets=offsets,coarse=coarse)
                    if coarse['passed']:
                        fine,data=validator.validate(chains,angle,offsets,True);attempt['fine']=fine
                        if fine['passed']:accepted=(choices,offsets,data,fine)
                    tries.append(attempt)
                    if accepted:break
                row['offset_trials']=tries
                if accepted:break
                forbidden.append(tuple(solution['indices']));row['status']='finite_offset_schedule_unresolved_combination_only';print('FULL_COMBINATION_REJECTED',si,solution['indices'],flush=True)
            if accepted is None:
                unresolved.append(dict(stage=si,reason='finite_stage_iteration_budget_or_unresolved_schedule',iterations=iterations));break
            choices,offsets,data,verdict=accepted
            candidate_pieces.append(dict(states=data['states'],time=data['times'],phases=np.asarray(data['phases']),tasks=np.asarray(data['tasks']),stage=np.full(len(data['times']),si,int)))
            stage_path=out/f'stage_{si:02d}.npz'
            stage_tcp=np.stack([[ENGINE.world.models[rid].tcp(q[:6])[:3,3] for q in data['states'][:,6*arm:6*arm+6]] for arm,rid in enumerate(ROBOT_IDS)],axis=1)
            np.savez_compressed(stage_path,states=data['states'],time=data['times'],times=data['times'],phases=np.asarray(data['phases']),tasks=np.asarray(data['tasks']),tcp=stage_tcp)
            bindings[str(stage_path.resolve())]=digest(stage_path)
            current=data['states'][-1];chosen_all.update(choices);stage_rows.append(dict(stage=si,angle=angle,offsets=list(offsets),choices=choices,validation=verdict,duration_s=float(data['times'][-1]),task_ids=orders,receipt_file=str(stage_path.resolve()),receipt_sha256=digest(stage_path)));atomic(out/f'stage_{si:02d}.json',stage_rows[-1]);print('FULL_STAGE_ACCEPTED',si,data['times'][-1],flush=True)
    complete=len(stage_rows)==len(STAGES)
    if complete:
        first=candidate_pieces[0];prefix={k:v[:1] for k,v in first.items()};trial=combine(prefix,candidate_pieces)
        if not np.allclose(trial['states'][[0,-1]],[initial,final],atol=1e-12,rtol=0):raise ValueError('Global endpoints changed')
        after=motion_metrics(trial['states'],trial['time']);verdict=acceptance(before,after)
        for task in range(1,22):
            arm=task_layers[task]['arm'];mask=(trial['tasks'][:,arm]==task)&(trial['phases'][:,arm]=='weld');runs=int(np.sum(mask&~np.r_[False,mask[:-1]]))
            if runs!=1:raise ValueError('Each all21 weld must occur once: '+str(task))
            node=nodes[task][chosen_all[task]]
            if not np.allclose(trial['states'][mask,6*arm:6*arm+6],node['path']['q_start_rad'],atol=1e-10,rtol=0):raise ValueError('Chosen weld raw joint binding changed')
            if not np.allclose(trial['states'][mask,12],np.deg2rad(task_layers[task]['angle']),atol=1e-12,rtol=0):raise ValueError('Fixed weld angle changed')
        tcp_arms=[]
        for arm,rid in enumerate(ROBOT_IDS):
            cache={};xyz=[]
            for q in trial['states'][:,6*arm:6*arm+6]:
                key=tuple(q)
                if key not in cache:cache[key]=ENGINE.world.models[rid].tcp(q)[:3,3]
                xyz.append(cache[key])
            tcp_arms.append(xyz)
        trial['tcp']=np.stack(tcp_arms,axis=1);trial['times']=trial['time'].copy()
        from .timing import audit_timing
        timing_receipt=audit_timing(trial,{t:task_layers[t]['arm'] for t in range(1,22)})
        if not timing_receipt['passed']:
            verdict['accepted']=False;verdict['reasons'].append('actual_fullroute_timing_invalid')
        atomic(out/'timing-audit.json',timing_receipt)
        bindings[str((out/'timing-audit.json').resolve())]=digest(out/'timing-audit.json')
        np.savez_compressed(out/'trial_route.npz',**trial)
        if verdict['accepted']:np.savez_compressed(out/'route.npz',**trial)
    else:timing_receipt=None;after=before.copy();verdict=dict(accepted=False,reasons=['incomplete_finite_source_search_incumbent_retained'])
    verify_bindings(bindings);verify_bindings({r['file']:r['sha256'] for r in records.values() if r['passed']})
    report=dict(**verdict,timing_audit=timing_receipt,adopted=verdict['accepted'],complete=complete,before=before,after=after,manifest=dict(bindings=bindings,settings=dict(workers=workers,iterations=iterations,edge_seconds=edge_seconds)),pool_sha256=digest(pool_file),incumbent_route=str(inc_route.resolve()),incumbent_sha256=digest(inc_route),baseline_retained=None if verdict['accepted'] else str(inc_route.resolve()),participation=participation,all_candidate_action_count=sum(len(v) for v in nodes.values()),all21_graph_nodes=[dict(task=t,action_ids=[n['id'] for n in nodes[t]]) for t in range(1,22)],selected_actions=[dict(task=t,arm=task_layers[t]['arm'],angle=task_layers[t]['angle'],weld_id=task_layers[t]['weld_id'],action=nodes[t][chosen_all[t]]['action']) for t in sorted(chosen_all)],stages=stage_rows,edges=list(records.values()),solver_rounds=rounds,unresolved=unresolved,canonical_park_barriers=True,fixed_angle_sequence=[s[0] for s in STAGES],no_hardware=True,timing_is_display_only=True,independent_cold_validation=False,scope='All certified 300mm library nodes participate. Exact per-arm joint order/candidate DP with duration-first lexicographic objective; additive arm duration proxy is not global concurrent makespan optimality. Single merged 120 degree batch and fixed 30,0,60,120,90 order give four rotations while preserving original full start/end states. Source fine-certified parallel timeline offsets tested. Finite lazy-edge failures remain unresolved, and schedule rejection stops this bounded attempt. Canonical parks only between angle batches.')
    atomic(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','out','incumbent'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--edge-seconds',type=float,default=60.)
    parser.add_argument('--reference-route',type=Path);parser.add_argument('--reference-sha256')
    args=parser.parse_args();optimize(args.pool,args.out,args.incumbent,args.workers,args.iterations,args.edge_seconds,args.reference_route,args.reference_sha256)

if __name__=='__main__':main()
