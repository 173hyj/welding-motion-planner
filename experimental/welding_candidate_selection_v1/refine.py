"""Continue a sealed candidate graph under joint, wrist and display-time budgets."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
from . import run as original
from . import core, edges, trajectory
from .pool import atomic, digest


def resources(piece):
    metrics=trajectory.motion_metrics(piece['states'],piece.get('time',piece.get('times')))
    return np.array([metrics['duration_s'],metrics['joint_travel_deg'],metrics['wrist_travel_deg']])


def fold_resources(layers,nodes,records,endpoints,known_only):
    sizes=[len(layer['actions']) for layer in layers]
    costs=[np.full((a,b),np.inf) for a,b in zip(sizes,sizes[1:])]
    rs=[np.full((*c.shape,3),np.inf) for c in costs]
    sc,ec=[np.full(n,np.inf) for n in (sizes[0],sizes[-1])]
    sr,er=[np.full((n,3),np.inf) for n in (sizes[0],sizes[-1])]
    for key,(a,b) in endpoints.items():
        row=records.get(key)
        if row is None:
            if known_only:continue
            f=core.raw_features(a,b)
            r=np.array([f['time_lower_bound_s'],f['sum_joint_delta_deg'],f['wrist_delta_deg']])
            c=core.configuration_cost(a,b)
        elif not row['passed']:continue
        else:r=np.asarray(row['resources']);c=row['cost']
        k,i,j=key
        if k==-1:sc[j]=c+nodes[0][j]['cost'];sr[j]=r+nodes[0][j]['resources']
        elif k==len(layers)-1:ec[i]=c;er[i]=r
        else:costs[k][i,j]=c+nodes[k+1][j]['cost'];rs[k][i,j]=r+nodes[k+1][j]['resources']
    return costs,rs,sc,ec,sr,er


def refinement_worker(job):
    key,a,b,seed,settings,path=job
    if key[0]!=-1 and a[12]!=b[12]:
        from .cross_angle import connect
        result=connect(original.ENGINE,a,b,baseline_states=seed,seconds=settings['edge_seconds'],seed=503+sum((n+2)*k for n,k in enumerate(key)),mesh_bounds=original.BOUNDS,do_trajopt=False)
        row=dict(key=list(key),passed=bool(result['passed']),stats=result['stats'])
        if result['passed']:
            piece=original.new_piece(result['states'],settings['robot_ids']);np.savez_compressed(path,**piece)
            row.update(file=str(path),sha256=digest(path),cost=original.measured_cost(piece['states'],piece['time']),duration_s=float(piece['time'][-1]),resources=resources(piece).tolist())
        atomic(Path(path).with_suffix('.json'),row)
        return key,row
    key,row=original.plan_worker(job)
    if row['passed']:row['resources']=resources(dict(np.load(row['file']))).tolist()
    atomic(Path(path).with_suffix('.json'),row)
    return key,row


def refine(pool_folder,previous_folder,out,workers=8,iterations=6,edge_seconds=60,max_labels=100000):
    from .resource import shortest_path_with_resources
    if workers<1 or iterations<0 or not np.isfinite(edge_seconds) or edge_seconds<=0:raise ValueError('Invalid finite refinement budget')
    pool_file=Path(pool_folder)/'report.json';previous_file=Path(previous_folder)/'report.json';out=Path(out).resolve()
    if out.exists():raise FileExistsError('Fresh output required')
    pool=json.loads(pool_file.read_text());previous=json.loads(previous_file.read_text())
    if digest(pool_file)!=previous['pool_sha256']:raise ValueError('Pool differs from sealed previous graph')
    bindings=dict(previous['manifest']['bindings'])
    bindings[str(previous_file.resolve())]=digest(previous_file)
    for name in ('refine.py','resource.py','cross_angle.py'):
        path=Path(__file__).with_name(name);bindings[str(path)]=digest(path)
    original.verify_bindings(bindings)
    records={tuple(row['key']):dict(row) for row in previous['edges']}
    inherited_hashes={r['file']:r['sha256'] for r in records.values() if r['passed']}
    original.verify_bindings(inherited_hashes)
    sys.path[:0]=[str(original.ROOT),str(original.ROOT/'experimental/welding_retreat_transfer_v1'),str(original.ROOT/'experimental/welding_variable_open_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from replay_full21_core_v105 import local_states
    from experimental.trajopt_transfer_v1.proximity import capture_mesh_bounds
    scene=original.ROOT/'build/scene_placement_corrected_v1.json';initialize_worker()
    with capture_mesh_bounds() as captured:original.ENGINE=load_raw_world(json.loads(scene.read_text()),SIGNS)
    original.BOUNDS=captured;original.TCP_CACHE=[{},{}]
    for rid,model in list(original.ENGINE.world.models.items()):original.ENGINE.world.models[rid]=getattr(model,'source',model)
    old=dict(np.load(pool['baseline_path'],allow_pickle=False));clock=old.get('time',old.get('times'));cut=33923
    if previous['prefix_end_index']!=cut or old['tasks'][cut,1]!=6 or old['phases'][cut,1]!='retreat':raise ValueError('Changed prefix boundary')
    layers=pool['layers'][7:];nodes=[];bounds=[]
    for layer in layers:
        mask=old['tasks'][:,layer['arm']]==layer['task']
        fi=int(np.flatnonzero(mask&(old['phases'][:,layer['arm']]=='ingress'))[0])-1
        li=int(np.flatnonzero(mask&(old['phases'][:,layer['arm']]=='retreat'))[-1]);bounds.append((fi,li));group=[]
        for i,action in enumerate(layer['actions']):
            if i==0:
                q=np.asarray(local_states(json.loads(Path(action['path_file']).read_text()),pool['robot_ids']))[-1]
                if not np.allclose(q,old['states'][[fi,li]],atol=1e-10,rtol=0):raise ValueError('Changed baseline action endpoint')
            piece=original.slice_piece(old,fi,li) if i==0 else original.make_action(action,layer,pool['robot_ids'])
            group.append(dict(piece=piece,cost=original.measured_cost(piece['states'],piece['time']),resources=resources(piece)))
        nodes.append(group)
    if bounds[0][0]!=pool['prefix_end_index']:raise ValueError('Pool action endpoint changed')
    seeds={-1:original.slice_piece(old,cut,bounds[0][0]),len(layers)-1:original.slice_piece(old,bounds[-1][1],len(clock)-1)}
    seeds.update({k:original.slice_piece(old,bounds[k][1],bounds[k+1][0]) for k in range(len(layers)-1)})
    endpoints={}
    for j,node in enumerate(nodes[0]):endpoints[-1,0,j]=(old['states'][cut],node['piece']['states'][0])
    for k,(left,right) in enumerate(zip(nodes,nodes[1:])):
        for i,a in enumerate(left):
            for j,b in enumerate(right):endpoints[k,i,j]=(a['piece']['states'][-1],b['piece']['states'][0])
    for i,node in enumerate(nodes[-1]):endpoints[len(layers)-1,i,0]=(node['piece']['states'][-1],old['states'][-1])
    for key,row in records.items():
        if key not in endpoints:raise ValueError('Foreign edge in sealed graph')
        if row['passed']:
            piece=dict(np.load(row['file']));a,b=endpoints[key]
            if not np.allclose(piece['states'][[0,-1]],[a,b],atol=1e-10,rtol=0):raise ValueError('Sealed edge endpoint mismatch')
            row['resources']=resources(piece).tolist()
    prefix={k:v[:cut+1] for k,v in old.items()}
    limits=resources(old)-resources(prefix)-np.array([0.,.001001,.001001])
    out.mkdir();(out/'edges').mkdir();rounds=[];replacements=[]
    settings=dict(edge_seconds=edge_seconds,trajopt=False,robot_ids=pool['robot_ids'])
    def solve(known_only):
        cs,rs,sc,ec,sr,er=fold_resources(layers,nodes,records,endpoints,known_only)
        return shortest_path_with_resources([x['actions'] for x in layers],cs,rs,limits,start_cost=sc,end_cost=ec,start_resources=sr,end_resources=er,max_labels=max_labels)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as executor:
        def execute(keys,refresh=False):
            jobs=[]
            for key in keys:
                a,b=endpoints[key];path=out/'edges'/('_'.join(map(str,key))+('_warp' if refresh else '')+'.npz')
                jobs.append((key,a,b,seeds[key[0]]['states'],settings,path))
            for key,row in executor.map(refinement_worker,jobs):
                prior=records.get(key)
                if refresh and prior and prior['passed']:
                    adopted=row['passed'] and row['cost']<=prior['cost']+1e-10 and np.all(np.asarray(row['resources'])<=np.asarray(prior['resources'])+1e-8)
                    replacements.append(dict(key=list(key),adopted=bool(adopted),trial=row))
                    if not adopted:continue
                records[key]=row
                print('REFINE_EDGE',key,row['passed'],row.get('resources'),flush=True)
        refresh=[key for key,row in records.items() if key[0]>=0 and endpoints[key][0][12]!=endpoints[key][1][12] and (not row['passed'] or row['stats'].get('strategy')=='inherited_baseline_with_adapters')]
        if refresh:
            atomic(out/'progress.json',dict(phase='cross_angle_refresh',edges=[list(k) for k in refresh]));execute(refresh,True)
        for iteration in range(iterations):
            known=solve(True);optimistic=solve(False)
            unknown=[] if optimistic is None else [k for k in original.route_keys(optimistic['indices']) if k not in records]
            rounds.append(dict(iteration=iteration,known=known,optimistic=optimistic,new_edges=[list(k) for k in unknown]))
            atomic(out/'progress.json',dict(phase='resource_selection',rounds=rounds))
            print('REFINE_ROUND',iteration,known,optimistic,unknown,flush=True)
            if not unknown:break
            execute(unknown)
        selected=solve(True)
    if selected is None:
        selected=dict(indices=[0]*len(layers),cost=sum(nodes[k][0]['cost'] for k in range(len(layers)))+sum(records[key]['cost'] for key in original.route_keys([0]*len(layers))),resources=resources(old)-resources(prefix),fallback='no_certified_route_with_strict_joint_and_wrist_improvement')
        selected['resources']=selected['resources'].tolist();trial={k:v.copy() for k,v in old.items()}
    else:
        pieces=[];keys=original.route_keys(selected['indices'])
        for k,index in enumerate(selected['indices']):
            pieces.extend([dict(np.load(records[keys[k]]['file'])),nodes[k][index]['piece']])
        pieces.append(dict(np.load(records[keys[-1]]['file'])))
        trial=trajectory.combine(prefix,pieces)
    before=trajectory.motion_metrics(old['states'],clock);after=trajectory.motion_metrics(trial['states'],trial.get('time',trial.get('times')))
    verdict=trajectory.acceptance(before,after)
    checks=dict(prefix_exact=all(np.array_equal(trial[k][:cut+1],v[:cut+1]) for k,v in old.items()),end_exact=np.array_equal(trial['states'][-1],old['states'][-1]))
    if not all(checks.values()):raise ValueError('Frozen prefix/end failed')
    chosen=[0]*7+selected['indices']
    selected_actions=[dict(task=x['task'],weld_id=x['weld_id'],angle=x['angle'],arm=x['arm'],action=x['actions'][i]) for x,i in zip(pool['layers'],chosen)]
    original.verify_bindings(bindings);original.verify_bindings(inherited_hashes)
    original.verify_bindings({r['file']:r['sha256'] for r in records.values() if r['passed']})
    np.savez_compressed(out/'trial_route.npz',**trial)
    if verdict['accepted']:np.savez_compressed(out/'route.npz',**trial)
    report=dict(**verdict,before=before,after=after,manifest=dict(bindings=bindings,settings=dict(workers=workers,iterations=iterations,edge_seconds=edge_seconds,max_labels=max_labels)),pool_sha256=digest(pool_file),selected_actions=selected_actions,edges=list(records.values()),solver_rounds=rounds,solution=selected,scp=[],cross_angle_refresh=replacements,resource_limits=limits.tolist(),checks=checks,prefix_end_index=cut,pool_action_endpoint_marker=pool['prefix_end_index'],prefix_boundary_reason=previous['prefix_boundary_reason'],baseline_retained=None if verdict['accepted'] else pool['baseline_path'],no_hardware=True,timing_is_display_only=True,independent_cold_validation=False,scope='Continuation of sealed sampled graph with strict joint/wrist and display-time resource bounds; finite unknown edges excluded from final route.')
    atomic(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','previous','out'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--edge-seconds',type=float,default=60.);parser.add_argument('--max-labels',type=int,default=100000)
    args=parser.parse_args();refine(args.pool,args.previous,args.out,args.workers,args.iterations,args.edge_seconds,args.max_labels)

if __name__=='__main__':main()
