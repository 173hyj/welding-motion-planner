"""All-catalog stage optimization anchored to a sealed incumbent's actual motion."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
from . import optimize as base
from . import upgraded
from .graph import shortest_path_avoiding
from experimental.welding_candidate_selection_v1.pool import atomic,digest
from experimental.welding_candidate_selection_v1.run import verify_bindings
from experimental.welding_candidate_selection_v1.trajectory import motion_metrics,acceptance,combine


def stage_bounds(data,stages,task_layers):
    result=[];states=np.asarray(data['states'])
    for angle,orders in stages:
        starts=[];ends=[]
        for task in [t for order in orders for t in order]:
            arm=task_layers[task]['arm'];mask=np.asarray(data['tasks'])[:,arm]==task
            ingress=np.flatnonzero(mask&(np.asarray(data['phases'])[:,arm]=='ingress'))
            retreat=np.flatnonzero(mask&(np.asarray(data['phases'])[:,arm]=='retreat'))
            if not len(ingress) or not len(retreat):raise ValueError('Missing stage action: '+str(task))
            starts.append(max(0,int(ingress[0])-1));ends.append(int(retreat[-1]))
        lo,hi=min(starts),max(ends)
        if result and lo<=result[-1][1]:raise ValueError('Overlapping or unordered incumbent stages')
        if not np.allclose(states[lo:hi+1,12],np.deg2rad(angle),atol=1e-10,rtol=0):raise ValueError('Stage crosses positioner angle')
        result.append((lo,hi))
    for start,stop in zip([0]+[b[1] for b in result],[b[0] for b in result]+[len(states)-1]):
        if np.any(np.asarray(data['phases'])[start:stop+1]=='weld'):raise ValueError('Weld outside anchored stage')
    return result


def literal_slice(data,start,stop):
    piece={k:np.asarray(v[start:stop+1]).copy() for k,v in data.items() if k not in ('time','times')}
    clock=data.get('time',data.get('times'));piece['time']=np.asarray(clock[start:stop+1])-clock[start]
    if 'times' in data:piece['times']=piece['time'].copy()
    return piece


def stage_gate(before,after):
    tolerances=dict(duration_s=1e-7,joint_travel_deg=1e-6,wrist_travel_deg=1e-6,total_positioner_travel_deg=1e-6,rotation_episodes=0)
    reasons=[key+'_regression' for key,tol in tolerances.items() if after[key]>before[key]+tol]
    return dict(accepted=not reasons,reasons=reasons)


def assemble_segments(old,bounds,replacements):
    if len(bounds)!=len(replacements):raise ValueError('One replacement per anchored stage required')
    unchanged=all(all(k in piece and np.array_equal(v,piece[k]) for k,v in literal_slice(old,*bound).items()) for bound,piece in zip(bounds,replacements))
    if unchanged:return {k:np.asarray(v).copy() for k,v in old.items()}
    pieces=[];cursor=bounds[0][0]
    prefix=literal_slice(old,0,cursor)
    for (start,stop),piece in zip(bounds,replacements):
        if start!=cursor:pieces.append(literal_slice(old,cursor,start))
        pieces.append(piece);cursor=stop
    if cursor!=len(old['states'])-1:pieces.append(literal_slice(old,cursor,len(old['states'])-1))
    return combine(prefix,pieces)


def _full_metrics(piece):return motion_metrics(piece['states'],piece.get('time',piece.get('times')))


def optimize(pool_folder,out,incumbent_folder,previous_folders=(),workers=4,iterations=6,known_rounds=2,edge_seconds=60):
    if workers<1 or not 1<=known_rounds<iterations or edge_seconds<=0:raise ValueError('Positive workers/budget and 1 <= known_rounds < iterations required')
    out=Path(out).resolve();pool_file=Path(pool_folder)/'report.json';incumbent_folder=Path(incumbent_folder)
    if out.exists():raise FileExistsError('Fresh output required')
    pool=json.loads(pool_file.read_text());inc_file=incumbent_folder/'report.json';inc_route=incumbent_folder/'route.npz';audit_file=incumbent_folder/'artifact-audit.json'
    inc=json.loads(inc_file.read_text());audit=json.loads(audit_file.read_text())
    if not pool.get('complete') or len(pool['layers'])!=21:raise ValueError('Complete all21 candidate pool required')
    if not inc.get('accepted') or not audit.get('passed') or not audit.get('accepted'):raise ValueError('Accepted sealed incumbent required')
    if digest(inc_route)!=audit['route_sha256'] or digest(inc_file)!=audit['result_report_sha256']:raise ValueError('Incumbent artifact changed')
    bindings=dict(inc['manifest']['bindings'])
    for field in ('bindings','source_sha256','code_sha256'):bindings.update(pool['manifest'].get(field,{}))
    for path in (pool_file,inc_file,inc_route,audit_file,Path(__file__),Path(base.__file__),Path(upgraded.__file__),Path(__file__).with_name('graph.py'),Path(__file__).with_name('raw_transfer.py'),Path(__file__).with_name('raw_checker.py')):bindings[str(path.absolute())]=digest(path)
    for path in (base.ROOT/'experimental/dual_arm_parallel_v1').glob('*.py'):bindings[str(path.resolve())]=digest(path)
    for layer in pool['layers']:
        for action in layer['actions']:bindings[action['path_file']]=action['path_sha256']
    reused={};reuse_sources=[]
    for folder in previous_folders:
        path=Path(folder)/'report.json';report=json.loads(path.read_text());verify_bindings(report['manifest']['bindings'])
        if report['pool_sha256']!=digest(pool_file):raise ValueError('Reusable graph pool differs')
        bindings[str(path.resolve())]=digest(path)
        for row in report['edges']:
            if row.get('passed'):
                verify_bindings({row['file']:row['sha256']});bindings[row['file']]=row['sha256'];reused[tuple(row['key'])]=row
        reuse_sources.append(str(path.resolve()))
    verify_bindings(bindings)
    old=dict(np.load(inc_route,allow_pickle=False));before=_full_metrics(old);task_layers={x['task']:x for x in pool['layers']};bounds=stage_bounds(old,base.STAGES,task_layers)
    if 'tcp' not in old:raise ValueError('Incumbent full TCP array required')
    sys.path[:0]=[str(base.ROOT),str(base.ROOT/'experimental/welding_retreat_transfer_v1'),str(base.ROOT/'experimental/welding_variable_open_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    from world_bridge_v2 import SourceBridge
    from experimental.dual_arm_parallel_v1.motion import new_chain,append_piece
    from experimental.dual_arm_parallel_v1.run_parallel import TimelineValidator,prepare_context,action_context
    from experimental.dual_arm_parallel_v1.source_pool import SourcePool
    from experimental.dual_arm_parallel_v1.offset_candidates import offset_candidates
    from experimental.welding_candidate_selection_v1.edges import compress_collinear
    from .raw_checker import RawDualSourceChecker
    print('ANCHORED_WORLD_START',flush=True);initialize_worker();base.ENGINE=load_raw_world(json.loads((base.ROOT/'build/scene_placement_corrected_v1.json').read_text()),SIGNS)
    for rid,model in list(base.ENGINE.world.models.items()):base.ENGINE.world.models[rid]=getattr(model,'source',model)
    base.ROBOT_IDS=pool['robot_ids'];base.PARKS=[None,None];nodes={};fk_cache=[{},{}]
    def tcp(arm,states):
        result=[]
        for state in states:
            key=tuple(state[:6])
            if key not in fk_cache[arm]:fk_cache[arm][key]=base.ENGINE.world.models[base.ROBOT_IDS[arm]].tcp(state[:6])[:3,3].copy()
            result.append(fk_cache[arm][key])
        return np.asarray(result)
    for layer in pool['layers']:
        arm=layer['arm'];group=[]
        for action in layer['actions']:
            path=json.loads(Path(action['path_file']).read_text());park=np.r_[path['other_q_rad'],0.];other=1-arm
            if base.PARKS[other] is not None and not np.array_equal(base.PARKS[other],park):raise ValueError('Canonical proposal passive park differs')
            base.PARKS[other]=park;states=np.asarray(local_states(path,base.ROBOT_IDS));own=states[:,np.r_[np.arange(arm*6,arm*6+6),13+arm]]
            ingress,close,opening,retreat=action_sections(path,own);chain=new_chain(own[-1]);pieces=[]
            for phase,section in [('ingress',ingress),('close',close),('weld',[close[-1],close[-1]]),('open',opening),('retreat',retreat)]:
                section=np.asarray(section);xyz=tcp(arm,section);append_piece(chain,section,phase,task=layer['task'],duration=.6 if phase=='weld' else None,tcp=xyz);pieces.append((phase,section,xyz))
            group.append(dict(id=action['id'],action=action,path=path,pieces=pieces,endpoint=own[-1].copy(),**base.local_metrics(chain['states'],chain['times'])))
        nodes[layer['task']]=group;print('ANCHORED_LAYER_READY',layer['task'],len(group),flush=True)
    original_choices={}
    for row in inc['selected_actions']:
        task=row['task'];checksum=row['action']['path_sha256'];matches=[i for i,n in enumerate(nodes[task]) if n['action']['path_sha256']==checksum]
        if not matches:raise ValueError('Incumbent action absent from all-catalog graph: '+str(task))
        original_choices[task]=matches[0]
    upgraded.SEEDS=[dict(waypoints=compress_collinear(old['states']).tolist())];upgraded.REUSED=reused
    bridge=SourceBridge(base.ENGINE);checker=RawDualSourceChecker(base.ENGINE);records={};rounds=[];stage_rows=[];participation=[];replacement_pieces=[];selected_choices={};new_stage_count=0
    out.mkdir();(out/'edges').mkdir()
    for si,((angle,orders),(lo,hi)) in enumerate(zip(base.STAGES,bounds)):
        starts=[np.r_[old['states'][lo,6*a:6*a+6],old['states'][lo,13+a]] for a in range(2)];goals=[np.r_[old['states'][hi,6*a:6*a+6],old['states'][hi,13+a]] for a in range(2)]
        ordered,_,_,_,potential=base.build_graph(task_layers,nodes,{},orders,starts,goals,si)
        participation.append(dict(stage=si,angle=angle,bounds=[lo,hi],layers=[dict(task=t,arm=task_layers[t]['arm'],count=len(nodes[t]),action_ids=[n['id'] for n in nodes[t]]) for t in ordered],potential_transfer_edges=len(potential),all_nodes_in_initial_graph=True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as executor,SourcePool(bridge,checker,workers) as source_pool:
        samples=old['states'][[0,-1]];expected=[]
        for state in samples:
            r=bridge.probe(state);expected.append(dict(passed=bool(r['passed']),reason=r.get('reason',''),blocking_pairs=r.get('blocking_pairs',[])))
        if source_pool.strict_states(samples)!=expected:raise RuntimeError('Source fork equivalence failed')
        for si,((angle,orders),(lo,hi)) in enumerate(zip(base.STAGES,bounds)):
            literal=literal_slice(old,lo,hi);baseline_metrics=_full_metrics(literal);best=literal;best_metrics=baseline_metrics;best_choices={t:original_choices[t] for order in orders for t in order};best_source='incumbent';best_validation=dict(passed=True,inherited_incumbent_sha256=digest(inc_route));forbidden=[];stage_attempts=[]
            starts=[np.r_[old['states'][lo,6*a:6*a+6],old['states'][lo,13+a]] for a in range(2)];goals=[np.r_[old['states'][hi,6*a:6*a+6],old['states'][hi,13+a]] for a in range(2)]
            print('ANCHORED_STAGE_START',si,lo,hi,baseline_metrics,flush=True)
            for iteration in range(iterations):
                known_only=iteration>=iterations-known_rounds
                shadow=dict(records);ordered,costs,sc,ec,possible=base.build_graph(task_layers,nodes,records,orders,starts,goals,si)
                if known_only:
                    for key in possible:
                        if key not in shadow:shadow[key]=dict(passed=False)
                    ordered,costs,sc,ec,_=base.build_graph(task_layers,nodes,shadow,orders,starts,goals,si)
                solution=shortest_path_avoiding([nodes[t] for t in ordered],costs,sc,ec,forbidden_paths=forbidden)
                attempt=dict(stage=si,iteration=iteration,known_only=known_only,solution=solution);rounds.append(attempt);stage_attempts.append(attempt);atomic(out/'progress.json',dict(rounds=rounds,stages=stage_rows))
                if solution is None:
                    attempt['status']='no_certified_route' if known_only else 'finite_graph_unresolved';continue
                transfers=base.selected_transfers(ordered,solution['indices'],nodes,orders,starts,goals,si);missing=[(key,a,b) for key,a,b in transfers if key not in records]
                if missing:
                    if known_only:attempt['status']='idle_transfer_unknown';continue
                    jobs=[(key,a,b,angle,edge_seconds,out/'edges'/('e'+str(len(records)+i).zfill(6)+'.npz')) for i,(key,a,b) in enumerate(missing)]
                    for key,row in executor.map(upgraded.transfer_worker,jobs):records[key]=row;print('ANCHORED_EDGE',key,row['passed'],row.get('duration_s'),flush=True)
                    attempt['new_edges']=[list(k) for k,a,b in missing];continue
                if any(not records[key]['passed'] for key,a,b in transfers):attempt['status']='idle_transfer_unresolved';continue
                choices=dict(zip(ordered,solution['indices']));chains=[];meta={}
                for arm,order in enumerate(orders):
                    chain=new_chain(starts[arm]);previous='start'
                    for task in order:
                        node=nodes[task][choices[task]];data=np.load(records[si,arm,previous,node['id']]['file']);append_piece(chain,data['states'],'transfer',tcp=data['tcp'])
                        for phase,section,xyz in node['pieces']:append_piece(chain,section,phase,task=task,duration=.6 if phase=='weld' else None,tcp=xyz)
                        meta[task]=dict(id=task,path=node['path'],angle=angle,arm=arm,weld_id=task_layers[task]['weld_id']);meta[task]['context']=prepare_context(base.ENGINE,meta[task]);previous=node['id']
                    data=np.load(records[si,arm,previous,'end']['file']);append_piece(chain,data['states'],'transfer',tcp=data['tcp']);chain['states']=np.asarray(chain['states']);chain['times']=np.asarray(chain['times']);chains.append(chain)
                contact_records=[]
                for task in ordered[:2]:
                    arm=task_layers[task]['arm'];node=nodes[task][choices[task]];state=np.asarray(local_states(node['path'],base.ROBOT_IDS))[0];contexts=[None,None];contexts[arm]=action_context(base.ENGINE,meta,dict(task=task,phase='open'),np.r_[state[6*arm:6*arm+6],state[13+arm]]);contact_records.append((state,contexts))
                expected=[]
                for state,contexts in contact_records:
                    r=checker.probe(state,contexts);minimal={k:r[k] for k in ('passed','reason','blocking_pairs')};minimal['touching']={a:list(r.get('tool_states',{}).get(rid,{}).get('touching',[])) for a,rid in enumerate(base.ROBOT_IDS)};expected.append(minimal)
                if source_pool.dual_states(contact_records)!=expected:raise RuntimeError('Source contact fork equivalence failed')
                validator=TimelineValidator(base.ENGINE,checker,meta,source_pool);durations=[float(c['times'][-1]) for c in chains];offsets=[(0.,0.),(0.,durations[0]),(durations[1],0.)]
                for shift in (.5,1.,2.,4.,8.,16.,32.,64.):
                    if shift<durations[0]:offsets.append((0.,shift))
                    if shift<durations[1]:offsets.append((shift,0.))
                valid=None;trials=[]
                for trial in offset_candidates(chains,base_offsets=offsets):
                    offset=trial['offsets'];coarse,_=validator.validate(chains,angle,offset,False);record=dict(offsets=list(offset),coarse=coarse)
                    if coarse['passed']:
                        fine,data=validator.validate(chains,angle,offset,True);record['fine']=fine
                        if fine['passed']:valid=(offset,data,fine)
                    trials.append(record)
                    if valid:break
                attempt['offset_trials']=trials;forbidden.append(tuple(solution['indices']))
                if valid is None:attempt['status']='finite_schedule_unresolved_complete_combination';continue
                offset,data,fine=valid;candidate=dict(states=data['states'],time=data['times'],phases=np.asarray(data['phases']),tasks=np.asarray(data['tasks']),stage=np.full(len(data['times']),si,int))
                candidate['tcp']=np.stack([tcp(a,candidate['states'][:,np.r_[np.arange(6*a,6*a+6),13+a]]) for a in range(2)],axis=1)
                if 'times' in old:candidate['times']=candidate['time'].copy()
                metrics=_full_metrics(candidate);gate=stage_gate(baseline_metrics,metrics);attempt.update(status='source_checked',metrics=metrics,stage_gate=gate)
                if not np.array_equal(candidate['states'][[0,-1]],old['states'][[lo,hi]]):raise ValueError('Anchored stage endpoints changed')
                objective=lambda m:np.deg2rad(m['joint_travel_deg']+.5*m['wrist_travel_deg'])+.05*m['duration_s']
                if gate['accepted'] and objective(metrics)<objective(best_metrics)-1e-9:
                    best=candidate;best_metrics=metrics;best_choices=choices;best_source='new_source_checked';best_validation=fine;attempt['adopted_stage_candidate']=True
            if best_source=='new_source_checked':new_stage_count+=1
            replacement_pieces.append(best);selected_choices.update(best_choices);stage_record=dict(stage=si,angle=angle,bounds=[lo,hi],source=best_source,before=baseline_metrics,after=best_metrics,choices=best_choices,task_ids=orders,validation=best_validation,attempt_count=len(stage_attempts))
            stage_rows.append(stage_record);np.savez_compressed(out/f'stage_{si:02d}.npz',**best);stage_record['file']=str(out/f'stage_{si:02d}.npz');stage_record['sha256']=digest(stage_record['file']);atomic(out/f'stage_{si:02d}.json',stage_record);print('ANCHORED_STAGE_FINISHED',si,best_source,best_metrics,flush=True)
    trial=assemble_segments(old,bounds,replacement_pieces);after=_full_metrics(trial);verdict=acceptance(before,after)
    if not np.array_equal(trial['states'][[0,-1]],old['states'][[0,-1]]):raise ValueError('Original global endpoints changed')
    for task in range(1,22):
        arm=task_layers[task]['arm'];mask=(trial['tasks'][:,arm]==task)&(trial['phases'][:,arm]=='weld')
        if int(np.sum(mask&~np.r_[False,mask[:-1]]))!=1:raise ValueError('Exactly one weld run per task required')
        node=nodes[task][selected_choices[task]]
        if not np.allclose(trial['states'][mask,6*arm:6*arm+6],node['path']['q_start_rad'],atol=1e-10,rtol=0) or not np.allclose(trial['states'][mask,12],np.deg2rad(task_layers[task]['angle']),atol=1e-10,rtol=0):raise ValueError('Selected source weld binding changed')
    verify_bindings(bindings);verify_bindings({r['file']:r['sha256'] for r in records.values() if r.get('passed')})
    np.savez_compressed(out/'trial_route.npz',**trial)
    if verdict['accepted']:np.savez_compressed(out/'route.npz',**trial)
    report=dict(**verdict,adopted=verdict['accepted'],complete=True,before=before,after=after,manifest=dict(bindings=bindings,settings=dict(workers=workers,iterations=iterations,known_rounds=known_rounds,edge_seconds=edge_seconds)),pool_sha256=digest(pool_file),comparison_route=str(inc_route.resolve()),comparison_sha256=digest(inc_route),incumbent_route=str(inc_route.resolve()),baseline_retained=None if verdict['accepted'] else str(inc_route.resolve()),all_candidate_action_count=sum(len(v) for v in nodes.values()),all21_graph_nodes=[dict(task=t,action_ids=[n['id'] for n in nodes[t]]) for t in range(1,22)],participation=participation,selected_actions=[dict(task=t,arm=task_layers[t]['arm'],angle=task_layers[t]['angle'],weld_id=task_layers[t]['weld_id'],action=nodes[t][selected_choices[t]]['action']) for t in range(1,22)],stages=stage_rows,stage_bounds=[list(b) for b in bounds],new_source_checked_stages=new_stage_count,edges=list(records.values()),solver_rounds=rounds,reused_graph_reports=reuse_sources,canonical_park_barriers=False,interstage_policy='Literal incumbent frames and relative clocks retained; only accepted stage interiors replaced.',no_hardware=True,timing_is_display_only=True,independent_cold_validation=False,scope='All certified pool nodes eligible. Finite anchored-stage search; failed or regressing replacements retain sealed incumbent source segments. Static transfer proposals use canonical passive park and can cause false negatives; changed dual timelines require complete raw source checks. No global-optimality claim.')
    atomic(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','out','incumbent'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--previous',type=Path,action='append',default=[]);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--iterations',type=int,default=6);parser.add_argument('--known-rounds',type=int,default=2);parser.add_argument('--edge-seconds',type=float,default=60.)
    args=parser.parse_args();optimize(args.pool,args.out,args.incumbent,args.previous,args.workers,args.iterations,args.known_rounds,args.edge_seconds)

if __name__=='__main__':main()
