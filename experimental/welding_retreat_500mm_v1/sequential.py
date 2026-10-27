"""Matched sequential source-certified comparison. Offline only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .run import ROOT,atomic,digest

STAGES=[(30,[[7],[1,2]]),(120,[[],[3,4,5,6]]),(60,[[8,9],[]]),(0,[[10,11],[]]),(120,[[12,13,14,15,16,17,18,19,20],[]]),(90,[[21],[]])]

def serial_offsets(durations):return (0.,float(durations[0]))

def require_generation(report):
    rows=report['rows']
    if not report.get('complete') or len(rows)!=42 or not all(r['passed'] for r in rows):raise ValueError('All 42 fresh actions must pass')
    if {(r['length_mm'],r['task']) for r in rows}!={(d,t) for d in (500.,1000.) for t in range(1,22)}:raise ValueError('Missing/duplicate fixed action')
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--generation',type=Path,required=True);p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--iterations',type=int,default=6000)
    args=p.parse_args();started=time.perf_counter()
    from run_initial import extended_source_hashes,assert_frozen_geometry_cache
    assert_frozen_geometry_cache()
    report=json.loads(args.generation.read_text());rows=require_generation(report)
    bindings=dict(report['manifest']['bindings']);bindings[str(args.generation)]=digest(args.generation)
    bindings.update({str(ROOT/name):sha for name,sha in extended_source_hashes().items()})
    for row in rows:bindings[row['path_file']]=row['path_sha256']
    for file in Path(__file__).parent.glob('*.py'):bindings[str(file)]=digest(file)
    for folder in [ROOT/'experimental/welding_full_candidate_selection_v1',ROOT/'experimental/dual_arm_parallel_v1',ROOT/'experimental/welding_candidate_selection_v1']:
        for file in folder.glob('*.py'):bindings[str(file)]=digest(file)
    bindings[str(args.incumbent)]=digest(args.incumbent)
    for name,sha in bindings.items():
        if digest(name)!=sha:raise ValueError('Frozen input changed '+name)
    if args.out.exists():raise FileExistsError('Fresh comparison directory required')
    args.out.mkdir(parents=True);atomic(args.out/'manifest.json',dict(bindings=bindings,no_hardware=True))
    from run_geometry import initialize_worker
    from run_integrated_retreat_v118 import load_raw_world
    from mixed_bridge_v95 import SIGNS
    from world_bridge_v2 import SourceBridge
    from state_space import interpolate
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    from experimental.welding_full_candidate_selection_v1.raw_transfer import make_transfer
    from experimental.welding_full_candidate_selection_v1.raw_checker import RawDualSourceChecker
    from experimental.dual_arm_parallel_v1.motion import new_chain,append_piece
    from experimental.dual_arm_parallel_v1.run_parallel import TimelineValidator,prepare_context,action_context
    from experimental.dual_arm_parallel_v1.source_pool import SourcePool,prefill_fine_cache
    from experimental.welding_candidate_selection_v1.edges import compress_collinear
    from experimental.welding_candidate_selection_v1.trajectory import motion_metrics,combine
    from experimental.welding_candidate_selection_v1 import run as timing
    initialize_worker();engine=load_raw_world(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()),SIGNS)
    for rid,model in list(engine.world.models.items()):engine.world.models[rid]=getattr(model,'source',model)
    rids=sorted(engine.world.models);timing.ENGINE=engine
    old=dict(np.load(args.incumbent,allow_pickle=False));initial=old['states'][0].copy();final=old['states'][-1].copy()
    seeds=[dict(waypoints=compress_collinear(old['states']).tolist())]
    parks=[None,None]
    for row in rows:
        other=1-row['arm'];park=np.r_[row['initial']['other_q_rad'],0.]
        if parks[other] is not None and not np.array_equal(parks[other],park):raise ValueError('Canonical passive park changed')
        parks[other]=park
    class CoarseBridge(SourceBridge):
        def edge_scaled(self,a,b):
            resolution=self.resolutions/self.scale;resolution[:12]=np.deg2rad(.5)
            return all(self.valid_scaled(s) for s in interpolate(a,b,resolution))
    bridge=CoarseBridge(engine);checker=RawDualSourceChecker(engine);outcomes=[]
    with SourcePool(bridge,checker,args.workers) as pool:
        strict=[initial,final];expected=[{k:r[k] for k in ('passed','reason','blocking_pairs')} for r in (bridge.probe(s) for s in strict)]
        if pool.strict_states(strict)!=expected:raise ValueError('Source pool equivalence failed')
        bridge.fine_prefill=lambda path:prefill_fine_cache(bridge,path,pool)
        for length in (1000.,500.):
            out=args.out/str(int(length));out.mkdir();pieces=[];current=initial.copy();stage_rows=[];edges=[];failure=None
            selected={r['task']:r for r in rows if r['length_mm']==length}
            for si,(angle,orders) in enumerate(STAGES):
                print('SERIAL_STAGE_START',length,si,angle,flush=True)
                if si:
                    target=current.copy();target[12]=np.deg2rad(angle)
                    prefill_fine_cache(bridge,[current/bridge.scale,target/bridge.scale],pool)
                    if not SourceBridge.edge_scaled(bridge,current/bridge.scale,target/bridge.scale):
                        failure=dict(stage=si,reason='canonical_park_rotation_rejected',source=bridge.last_failure);break
                    states=np.asarray(interpolate(current,target,bridge.resolutions));piece=timing.new_piece(states,rids,phase='rotate',stage=si)
                    pieces.append(piece);current=target
                chains=[];meta={}
                for arm,order in enumerate(orders):
                    chain=new_chain(np.r_[current[6*arm:6*arm+6],current[13+arm]])
                    def tcp(section):return np.array([engine.world.models[rids[arm]].tcp(q[:6])[:3,3] for q in section])
                    def transfer(goal):
                        start=np.asarray(chain['states'][-1]);index=len(edges)
                        print('SERIAL_EDGE_START',length,si,arm,index,flush=True)
                        corners,stats=make_transfer(engine.world,bridge,arm,angle,start,goal,existing_edges=seeds,
                            seed=13000+index,park7=parks[1-arm],max_iterations=args.iterations)
                        row=dict(index=index,stage=si,arm=arm,passed=corners is not None,stats=stats)
                        edges.append(row);atomic(out/'edges'/f'edge_{index:03d}.json',row)
                        if corners is None:return False
                        dense=[]
                        for a,b in zip(corners,corners[1:]):
                            sample=interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5]);dense.extend(sample if not dense else sample[1:])
                        dense=np.asarray(dense) if dense else corners
                        row['waypoints']=corners.tolist();atomic(out/'edges'/f'edge_{index:03d}.json',row)
                        append_piece(chain,dense,'transfer',tcp=tcp(dense));return True
                    for task in order:
                        row=selected[task];path=json.loads(Path(row['path_file']).read_text())
                        from integrated_retreat_v117 import check_path_contract
                        check_path_contract(path,row['initial'],length)
                        states=np.asarray(local_states(path,rids));own=states[:,np.r_[np.arange(arm*6,arm*6+6),13+arm]]
                        if not transfer(own[-1]):failure=dict(stage=si,arm=arm,task=task,reason='finite_transfer_unresolved_not_infeasible');break
                        ingress,close,opening,retreat=action_sections(path,own)
                        for phase,section in [('ingress',ingress),('close',close),('weld',[close[-1],close[-1]]),('open',opening),('retreat',retreat)]:
                            section=np.asarray(section);append_piece(chain,section,phase,task=task,duration=.6 if phase=='weld' else None,tcp=tcp(section))
                        meta[task]=dict(id=task,path=path,angle=angle,arm=arm,weld_id=row['weld_id'])
                        meta[task]['context']=prepare_context(engine,meta[task])
                    if failure:break
                    goal=np.r_[final[6*arm:6*arm+6],final[13+arm]] if si==len(STAGES)-1 else parks[arm]
                    if not transfer(goal):failure=dict(stage=si,arm=arm,reason='finite_park_transfer_unresolved_not_infeasible');break
                    chain['states']=np.asarray(chain['states']);chain['times']=np.asarray(chain['times']);chains.append(chain)
                if failure:break
                durations=[float(c['times'][-1]) for c in chains];offsets=serial_offsets(durations)
                # A fixed full source probe verifies forked contact semantics for this angle.
                if meta:
                    task=next(iter(meta));arm=meta[task]['arm'];state=np.asarray(local_states(meta[task]['path'],rids))[0]
                    contexts=[None,None];contexts[arm]=action_context(engine,meta,dict(task=task,phase='open'),np.r_[state[6*arm:6*arm+6],state[13+arm]])
                    result=checker.probe(state,contexts);expected={k:result[k] for k in ('passed','reason','blocking_pairs')}
                    expected['touching']={a:list(result.get('tool_states',{}).get(rid,{}).get('touching',[])) for a,rid in enumerate(rids)}
                    if pool.dual_states([(state,contexts)])!=[expected]:raise ValueError('Source contact pool equivalence failed')
                validator=TimelineValidator(engine,checker,meta,pool);verdict,data=validator.validate(chains,angle,offsets,True)
                if not verdict['passed']:failure=dict(stage=si,reason='serial_timeline_source_rejected',validation=verdict);break
                piece=dict(states=data['states'],time=data['times'],phases=np.asarray(data['phases']),tasks=np.asarray(data['tasks']),stage=np.full(len(data['times']),si,int))
                pieces.append(piece);current=data['states'][-1]
                stage_row=dict(stage=si,angle=angle,orders=orders,durations=durations,offsets=offsets,duration_s=float(data['times'][-1]),validation=verdict)
                stage_rows.append(stage_row);atomic(out/f'stage_{si:02d}.json',stage_row)
                print('SERIAL_STAGE_PASSED',length,si,stage_row['duration_s'],flush=True)
            complete=failure is None and len(stage_rows)==6
            result=dict(length_mm=length,complete=complete,accepted=complete,stages=stage_rows,edges=edges,failure=failure,no_hardware=True)
            if complete:
                prefix={k:v[:1] for k,v in pieces[0].items()};trial=combine(prefix,pieces)
                if not np.allclose(trial['states'][[0,-1]],[initial,final],atol=1e-12,rtol=0):raise ValueError('Matched global endpoints changed')
                # Every weld appears once, in approved serial task order.
                events=[]
                for i in range(len(trial['states'])):
                    for arm in (0,1):
                        if trial['phases'][i,arm]=='weld' and (i==0 or trial['phases'][i-1,arm]!='weld' or trial['tasks'][i-1,arm]!=trial['tasks'][i,arm]):events.append(int(trial['tasks'][i,arm]))
                if events!=[7,*range(1,7),*range(8,22)]:raise ValueError('Weld count/order changed')
                trial['tcp']=timing.tcp_values(trial['states'],rids);trial['times']=trial['time'].copy()
                np.savez_compressed(out/'route.npz',**trial)
                metrics=motion_metrics(trial['states'],trial['time'])
                tcp_distance=np.linalg.norm(np.diff(trial['tcp'],axis=0),axis=2).sum(axis=0)
                metrics.update(tcp_distance_mm_by_arm=tcp_distance.tolist(),tcp_distance_mm_total=float(tcp_distance.sum()),
                    max_adjacent_joint_change_deg=float(np.rad2deg(abs(np.diff(trial['states'][:,:12],axis=0))).max()),
                    max_joint_configuration_range_deg=float(np.rad2deg(np.ptp(trial['states'][:,:12],axis=0)).max()))
                result.update(metrics=metrics,route_sha256=digest(out/'route.npz'),weld_order=events)
            atomic(out/'report.json',result);outcomes.append(result)
            atomic(args.out/'progress.json',dict(outcomes=outcomes))
    for name,sha in bindings.items():
        if digest(name)!=sha:raise ValueError('Bound input changed during comparison '+name)
    accepted=all(o['accepted'] for o in outcomes)
    comparison=dict(complete=accepted,accepted=accepted,outcomes=outcomes,manifest=dict(bindings=bindings),elapsed_s=time.perf_counter()-started,
        display_timing_only=True,no_hardware=True,independent_cold_validation=False,scope='Fine sampled original source geometry; serial fixed arm assignment and canonical parks. Finite unresolved searches do not prove infeasibility.')
    if accepted:comparison['duration_saved_s']=outcomes[0]['metrics']['duration_s']-outcomes[1]['metrics']['duration_s']
    atomic(args.out/'report.json',comparison)

if __name__=='__main__':main()
