"""All-candidate fixed-stage offline optimization with source-certified lazy transfers."""
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
STAGES=[(30,[[7],[1,2]]),(120,[[],[3,4,5,6]]),(60,[[8,9],[]]),(0,[[10,11],[]]),(120,[[12,13,14,15,16,17,18,19,20],[]]),(90,[[21],[]])]


def local_metrics(states,times):
    states=np.asarray(states);delta=abs(np.diff(states,axis=0));duration=float(times[-1])
    joint=float(np.rad2deg(delta[:,:6]).sum());wrist=float(np.rad2deg(delta[:,3:6]).sum())
    return dict(cost=float(np.deg2rad(joint+.5*wrist)+.05*duration),duration_s=duration,joint_travel_deg=joint,wrist_travel_deg=wrist)


def endpoint_cost(a,b):
    d=abs(np.asarray(b)-np.asarray(a));seconds=max(float(d[:6].max()/np.deg2rad(30)),float(d[6]/80.))
    return float(d[:6].sum()+.5*d[3:6].sum()+.05*seconds)


def build_graph(task_layers,nodes,records,arm_orders,starts,goals,stage):
    """Every certified action remains a vertex; joining arms is only additive accounting."""
    ordered=[t for order in arm_orders for t in order]
    matrices=[];edgekeys=set()
    def transfer(arm,left,right,a,b):
        key=(stage,arm,left,right);edgekeys.add(key)
        row=records.get(key)
        return (row['cost'] if row['passed'] else np.inf) if row is not None else endpoint_cost(a,b)
    def incoming(task,j):
        arm=task_layers[task]['arm'];node=nodes[task][j]
        return transfer(arm,'start',node['id'],starts[arm],node['endpoint'])+node['cost']
    def outgoing(task,i):
        arm=task_layers[task]['arm'];node=nodes[task][i]
        return transfer(arm,node['id'],'end',node['endpoint'],goals[arm])
    sc=np.array([incoming(ordered[0],j) for j in range(len(nodes[ordered[0]]))])
    ec=np.array([outgoing(ordered[-1],i) for i in range(len(nodes[ordered[-1]]))])
    for left,right in zip(ordered,ordered[1:]):
        matrix=np.empty((len(nodes[left]),len(nodes[right])))
        for i,a in enumerate(nodes[left]):
            for j,b in enumerate(nodes[right]):
                if task_layers[left]['arm']!=task_layers[right]['arm']:matrix[i,j]=outgoing(left,i)+incoming(right,j)
                else:
                    arm=task_layers[left]['arm']
                    matrix[i,j]=transfer(arm,a['id'],b['id'],a['endpoint'],b['endpoint'])+b['cost']
        matrices.append(matrix)
    return ordered,matrices,sc,ec,edgekeys


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
    from state_space import interpolate,connect_rrt
    from experimental.dual_arm_parallel_v1.motion import section_times
    from experimental.welding_candidate_selection_v1.pool import atomic,digest
    arm=key[1];axes=np.r_[np.arange(arm*6,arm*6+6),13+arm]
    base=np.zeros(15);other=1-arm;base[6*other:6*other+6]=PARKS[other][:6];base[13+other]=PARKS[other][6];base[12]=np.deg2rad(angle)
    fine=SourceBridge(ENGINE);coarse=SourceBridge(ENGINE);coarse.resolutions[:12]=np.deg2rad(.5)
    deadline=time.perf_counter()+seconds
    def full(q):
        s=base.copy();s[axes]=q;return s/fine.scale
    def certify(candidate):
        return all(fine.edge_scaled(full(a),full(b)) for a,b in zip(candidate,candidate[1:])) if len(candidate)>1 else fine.valid_scaled(full(candidate[0]))
    corners=None;stats=dict(status='unresolved',fine_certified=False,source_resolution_deg=.05,proposal_resolution_deg=.5)
    if np.array_equal(start,goal):proposals=[('identity',np.asarray([start]))]
    else:
        closed_start=start.copy();closed_start[6]=0.;closed_goal=goal.copy();closed_goal[6]=0.
        proposals=[('direct',np.array([start,goal])),('close_move_open',np.array([start,closed_start,closed_goal,goal])),('canonical_park_gateway',np.array([start,closed_start,PARKS[arm],closed_goal,goal]))]
    for method,candidate in proposals:
        if time.perf_counter()>deadline:break
        if certify(candidate):corners=candidate;stats.update(strategy=method);break
    if corners is None and time.perf_counter()<deadline:
        scale=fine.scale[axes];low=fine.low[axes]/scale;high=fine.high[axes]/scale
        class Expired(RuntimeError):pass
        def valid(a,b):
            if time.perf_counter()>deadline:raise Expired()
            return coarse.edge_scaled(full(a*scale),full(b*scale))
        try:
            proposed,rrt=connect_rrt(start/scale,goal/scale,low,high,valid,seed=7103+sum(ord(c) for c in str(key)),max_iterations=1200,step=.04)
            stats['rrt']=rrt
            if proposed is not None:
                proposed=np.asarray(proposed)*scale;proposed[[0,-1]]=[start,goal]
                if certify(proposed):corners=proposed;stats['strategy']='rrt'
        except Expired:stats['reason']='finite_proposal_budget_expired'
    row=dict(key=list(key),passed=corners is not None,stats=stats)
    if corners is not None:
        dense=[]
        for a,b in zip(corners,corners[1:]):
            samples=interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5]);dense.extend(samples if not dense else samples[1:])
        states=np.asarray(dense) if dense else corners.copy();states[[0,-1]]=[start,goal]
        tcp=np.array([ENGINE.world.models[ROBOT_IDS[arm]].tcp(q[:6])[:3,3] for q in states]);times=section_times(states,tcp)
        np.savez_compressed(path,states=states,times=times,tcp=tcp)
        row.update(file=str(path),sha256=digest(path),**local_metrics(states,times));stats.update(status='source_fine_certified',fine_certified=True,checks=fine.checks)
    else:stats.setdefault('reason','finite_search_exhausted_not_infeasibility')
    atomic(Path(path).with_suffix('.json'),row);return key,row


def optimize(pool_folder,out,incumbent_folder,workers=8,iterations=12,edge_seconds=60):
    global ENGINE,ROBOT_IDS,PARKS
    from experimental.welding_candidate_selection_v1.pool import atomic,digest
    from experimental.welding_candidate_selection_v1.run import verify_bindings
    from experimental.welding_candidate_selection_v1.trajectory import motion_metrics,acceptance,combine
    from .graph import shortest_path_avoiding
    if workers<1 or iterations<1 or edge_seconds<=0 or not np.isfinite(edge_seconds):raise ValueError('Invalid finite budget')
    pool_file=Path(pool_folder)/'report.json';pool=json.loads(pool_file.read_text());out=Path(out).resolve();incumbent_folder=Path(incumbent_folder)
    if out.exists():raise FileExistsError('Fresh output required')
    if not pool.get('complete') or len(pool['layers'])!=21:raise ValueError('Complete all21 pool required')
    inc_file=incumbent_folder/'report.json';inc_report=json.loads(inc_file.read_text());inc_route=incumbent_folder/'route.npz';audit_file=incumbent_folder/'artifact-audit.json'
    if not inc_report['accepted'] or not inc_route.exists():raise ValueError('Accepted incumbent required')
    bindings={**inc_report['manifest']['bindings'],**pool['manifest'].get('source_sha256',{}),**pool['manifest'].get('code_sha256',{}),**pool['manifest'].get('bindings',{}),str(pool_file.resolve()):digest(pool_file),str(inc_file.resolve()):digest(inc_file),str(inc_route.resolve()):digest(inc_route)}
    if not audit_file.exists():raise ValueError('Sealed incumbent artifact audit required')
    audit=json.loads(audit_file.read_text());bindings[str(audit_file.resolve())]=digest(audit_file)
    if not audit.get('passed') or not audit.get('accepted'):raise ValueError('Incumbent artifact audit failed')
    if audit['route_sha256']!=digest(inc_route) or audit['result_report_sha256']!=digest(inc_file):raise ValueError('Incumbent differs from sealed artifact audit')
    for folder in (Path(__file__).parent,ROOT/'experimental/dual_arm_parallel_v1'):
        for path in folder.glob('*.py'):bindings[str(path.resolve())]=digest(path)
    for layer in pool['layers']:
        for action in layer['actions']:bindings[action['path_file']]=action['path_sha256']
    verify_bindings(bindings)
    old=dict(np.load(inc_route,allow_pickle=False));oldtime=old.get('time',old.get('times'));before=motion_metrics(old['states'],oldtime)
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
    from .raw_checker import RawDualSourceChecker
    initialize_worker();ENGINE=load_raw_world(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()),SIGNS)
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    ROBOT_IDS=pool['robot_ids'];PARKS=[None,None];task_layers={x['task']:x for x in pool['layers']};nodes={}
    for layer in pool['layers']:
        group=[]
        for index,action in enumerate(layer['actions']):
            path=json.loads(Path(action['path_file']).read_text());arm=layer['arm'];other=1-arm;park=np.r_[path['other_q_rad'],0.]
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
            current=data['states'][-1];chosen_all.update(choices);stage_rows.append(dict(stage=si,angle=angle,offsets=list(offsets),choices=choices,validation=verdict,duration_s=float(data['times'][-1]),task_ids=orders));atomic(out/f'stage_{si:02d}.json',stage_rows[-1]);print('FULL_STAGE_ACCEPTED',si,data['times'][-1],flush=True)
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
        np.savez_compressed(out/'trial_route.npz',**trial)
        if verdict['accepted']:np.savez_compressed(out/'route.npz',**trial)
    else:after=before.copy();verdict=dict(accepted=False,reasons=['incomplete_finite_source_search_incumbent_retained'])
    verify_bindings(bindings);verify_bindings({r['file']:r['sha256'] for r in records.values() if r['passed']})
    report=dict(**verdict,adopted=verdict['accepted'],complete=complete,before=before,after=after,manifest=dict(bindings=bindings,settings=dict(workers=workers,iterations=iterations,edge_seconds=edge_seconds)),pool_sha256=digest(pool_file),incumbent_route=str(inc_route.resolve()),incumbent_sha256=digest(inc_route),baseline_retained=None if verdict['accepted'] else str(inc_route.resolve()),participation=participation,all_candidate_action_count=sum(len(v) for v in nodes.values()),all21_graph_nodes=[dict(task=t,action_ids=[n['id'] for n in nodes[t]]) for t in range(1,22)],selected_actions=[dict(task=t,arm=task_layers[t]['arm'],angle=task_layers[t]['angle'],weld_id=task_layers[t]['weld_id'],action=nodes[t][chosen_all[t]]['action']) for t in sorted(chosen_all)],stages=stage_rows,edges=list(records.values()),solver_rounds=rounds,unresolved=unresolved,canonical_park_barriers=True,fixed_angle_sequence=[s[0] for s in STAGES],no_hardware=True,timing_is_display_only=True,independent_cold_validation=False,scope='Every passed pool action is a graph vertex. Finite lazy transfer/schedule search with explicit canonical-park stage barriers; no claim of global optimality. Finite failures remain unresolved.')
    atomic(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','out','incumbent'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--edge-seconds',type=float,default=60.)
    args=parser.parse_args();optimize(args.pool,args.out,args.incumbent,args.workers,args.iterations,args.edge_seconds)

if __name__=='__main__':main()
