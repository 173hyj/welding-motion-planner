"""Bounded lazy joint-candidate selection; all new edges require source certification."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
from . import core, edges, trajectory
from .pool import digest, atomic

ROOT = Path(__file__).absolute().parents[2]
ENGINE = BOUNDS = None
TCP_CACHE = [{}, {}]


def verify_bindings(bindings):
    for name, checksum in bindings.items():
        if digest(name) != checksum:
            raise ValueError('Changed frozen input: '+name)


def measured_cost(states, times):
    m = trajectory.motion_metrics(states, times)
    return float(np.deg2rad(m['joint_travel_deg'] + .5*m['wrist_travel_deg']) + .05*m['duration_s'])


def fold_graph(layers, nodes, records, endpoints, known_only=False):
    """Charge each node exactly once, on its incoming edge (including start)."""
    sizes = [len(x['actions']) for x in layers]
    matrices = [np.full((a,b),np.inf) for a,b in zip(sizes,sizes[1:])]
    times = [m.copy() for m in matrices]
    sc,ec,st,et = [np.full(n,np.inf) for n in (sizes[0],sizes[-1],sizes[0],sizes[-1])]
    for key,(a,b) in endpoints.items():
        row = records.get(key)
        if row is None:
            if known_only: continue
            c,t = core.configuration_cost(a,b),core.raw_features(a,b)['time_lower_bound_s']
        elif not row['passed']: continue
        else: c,t = row['cost'],row['duration_s']
        layer,i,j = key
        if layer == -1:
            sc[j],st[j] = c+nodes[0][j]['cost'],t+nodes[0][j]['duration_s']
        elif layer == len(layers)-1: ec[i],et[i] = c,t
        else:
            matrices[layer][i,j] = c+nodes[layer+1][j]['cost']
            times[layer][i,j] = t+nodes[layer+1][j]['duration_s']
    return matrices,times,sc,ec,st,et


def route_keys(indices):
    return [(-1,0,indices[0])]+[(k,i,j) for k,(i,j) in enumerate(zip(indices,indices[1:]))]+[(len(indices)-1,indices[-1],0)]


def slice_piece(data,start,stop):
    piece={k:np.asarray(v[start:stop+1]).copy() for k,v in data.items() if k not in ('time','times')}
    clock=data.get('time',data.get('times'))
    piece['time']=np.asarray(clock[start:stop+1])-clock[start]
    return piece


def tcp_values(states,robot_ids):
    result=[]
    for arm,rid in enumerate(robot_ids):
        model=ENGINE.world.models[rid];cache=TCP_CACHE[arm];values=[]
        for q in states[:,6*arm:6*arm+6]:
            key=tuple(q)
            if key not in cache: cache[key]=np.asarray(model.tcp(q))[:3,3]
            values.append(cache[key])
        result.append(values)
    return np.stack(result,axis=1)


def new_piece(states,robot_ids,phase='transfer',task=0,arm=0,stage=3):
    states=np.asarray(states,float);tcp=tcp_values(states,robot_ids)
    speed=np.r_[np.full(12,np.deg2rad(30)),np.deg2rad(15),80.,80.]
    dt=np.max(abs(np.diff(states,axis=0))/speed,axis=1)
    dt=np.maximum(dt,np.max(np.linalg.norm(np.diff(tcp,axis=0),axis=2),axis=1)/200.)
    dt=np.maximum(dt,1e-5)
    if phase=='weld':
        if len(states)!=2 or not np.array_equal(states[0],states[1]): raise ValueError('Weld must be stationary duplicate')
        dt[:]=.6
    phases=np.full((len(states),2),'idle',dtype='<U16');phases[:,arm]=phase
    tasks=np.zeros((len(states),2),int);tasks[:,arm]=task
    return dict(states=states,tcp=tcp,time=np.r_[0.,np.cumsum(dt)],phases=phases,tasks=tasks,stage=np.full(len(states),stage,int))


def make_action(action,layer,robot_ids):
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    path=json.loads(Path(action['path_file']).read_text());states=np.asarray(local_states(path,robot_ids))
    sections=action_sections(path,states)
    pieces=[new_piece(s,robot_ids,phase,layer['task'],layer['arm']) for phase,s in
            zip(('ingress','close','weld','open','retreat'),(sections[0],sections[1],[sections[1][-1]]*2,sections[2],sections[3]))]
    first=pieces[0]
    prefix={k:v[:1] for k,v in first.items()}
    return trajectory.combine(prefix,pieces)


def plan_worker(job):
    key,a,b,seed,settings,path = job
    if np.array_equal(a,b) and (seed is None or len(seed)==1):
        result=dict(passed=True,states=np.asarray([a]),stats=dict(strategy='identity'))
    else:
        connector=edges.connect
        if key[0]==-1:
            from .handoff import connect as connector
        result=connector(ENGINE,a,b,baseline_states=seed,seconds=settings['edge_seconds'],
                             seed=101+sum((n+2)*k for n,k in enumerate(key)),mesh_bounds=BOUNDS,do_trajopt=settings['trajopt'])
    row=dict(key=list(key),passed=bool(result['passed']),stats=result['stats'])
    if result['passed']:
        piece=new_piece(result['states'],settings['robot_ids'])
        np.savez_compressed(path,**piece)
        row.update(file=str(path),sha256=digest(path),cost=measured_cost(piece['states'],piece['time']),duration_s=float(piece['time'][-1]))
    atomic(Path(path).with_suffix('.json'),row)
    return key,row


def run(pool_folder,out,workers=8,iterations=6,edge_seconds=60,trajopt=True):
    global ENGINE,BOUNDS
    if workers<1 or iterations<1 or not np.isfinite(edge_seconds) or edge_seconds<=0: raise ValueError('Invalid search budget')
    pool_file=Path(pool_folder)/'report.json';pool=json.loads(pool_file.read_text());out=Path(out).resolve()
    if not pool.get('complete'): raise ValueError('Incomplete pool')
    if out.exists(): raise FileExistsError('Fresh output directory required')
    bindings={str(pool_file.resolve()):digest(pool_file),**pool['manifest']['source_sha256'],**pool['manifest']['code_sha256']}
    for name in ('run.py','core.py','edges.py','trajectory.py','handoff.py'):
        p=Path(__file__).with_name(name);bindings[str(p)]=digest(p)
    bindings[pool['baseline_path']]=pool['baseline_sha256']
    for layer in pool['layers']:
        for a in layer['actions']: bindings[a['path_file']]=a['path_sha256']
    verify_bindings(bindings)
    sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_retreat_transfer_v1'),str(ROOT/'experimental/welding_variable_open_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from replay_full21_core_v105 import local_states
    from experimental.trajopt_transfer_v1.proximity import capture_mesh_bounds
    scene=ROOT/'build/scene_placement_corrected_v1.json'
    if str(scene) not in bindings: raise ValueError('Scene not bound')
    initialize_worker()
    with capture_mesh_bounds() as captured:
        ENGINE=load_raw_world(json.loads(scene.read_text()),SIGNS)
    BOUNDS=captured
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    old=dict(np.load(pool['baseline_path'],allow_pickle=False));clock=old.get('time',old.get('times'))
    layers=pool['layers'][7:];action_endpoint_marker=pool['prefix_end_index'];cut=33923;bounds=[];nodes=[]
    if old['tasks'][cut,1]!=6 or old['phases'][cut,1]!='retreat':raise ValueError('Frozen handoff prefix boundary changed')
    for layer in layers:
        mask=old['tasks'][:,layer['arm']]==layer['task']
        fi=int(np.flatnonzero(mask&(old['phases'][:,layer['arm']]=='ingress'))[0])-1
        li=int(np.flatnonzero(mask&(old['phases'][:,layer['arm']]=='retreat'))[-1]);bounds.append((fi,li))
        group=[]
        for i,action in enumerate(layer['actions']):
            if i==0:
                source_states=np.asarray(local_states(json.loads(Path(action['path_file']).read_text()),pool['robot_ids']))
                if not np.allclose(source_states[-1],old['states'][[fi,li]],atol=1e-10,rtol=0):raise ValueError('Baseline action endpoint does not match sealed NPZ')
            piece=slice_piece(old,fi,li) if i==0 else make_action(action,layer,pool['robot_ids'])
            group.append(dict(piece=piece,cost=measured_cost(piece['states'],piece['time']),duration_s=float(piece['time'][-1])))
        nodes.append(group)
    if bounds[0][0]!=action_endpoint_marker:raise ValueError('Pool/action boundary changed')
    seeds={-1:slice_piece(old,cut,bounds[0][0]),len(layers)-1:slice_piece(old,bounds[-1][1],len(clock)-1)}
    seeds.update({k:slice_piece(old,bounds[k][1],bounds[k+1][0]) for k in range(len(layers)-1)})
    endpoints={};records={};out.mkdir();(out/'edges').mkdir()
    for j,node in enumerate(nodes[0]):endpoints[-1,0,j]=(old['states'][cut],node['piece']['states'][0])
    for k,(left,right) in enumerate(zip(nodes,nodes[1:])):
        for i,a in enumerate(left):
            for j,b in enumerate(right):endpoints[k,i,j]=(a['piece']['states'][-1],b['piece']['states'][0])
    for i,node in enumerate(nodes[-1]):endpoints[len(layers)-1,i,0]=(node['piece']['states'][-1],old['states'][-1])
    for key in endpoints:
        if key[1:]!=(0,0):continue
        piece=seeds[key[0]];p=out/'edges'/('_'.join(map(str,key))+'.npz');np.savez_compressed(p,**piece)
        records[key]=dict(key=list(key),passed=True,file=str(p),sha256=digest(p),cost=measured_cost(piece['states'],piece['time']),duration_s=float(piece['time'][-1]),stats=dict(strategy='inherited_baseline'))
    for key,row in records.items():atomic(Path(row['file']).with_suffix('.json'),row)
    rounds=[];scp_attempts=[];settings=dict(edge_seconds=edge_seconds,trajopt=False,robot_ids=pool['robot_ids'])
    ctx=multiprocessing.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize_worker) as executor:
        def execute(keys,optimize=False):
            jobs=[]
            for key in keys:
                a,b=endpoints[key];seed=seeds[key[0]]['states']
                if optimize:seed=np.load(records[key]['file'])['states']
                p=out/'edges'/('_'.join(map(str,key))+('_scp' if optimize else '')+'.npz')
                jobs.append((key,a,b,seed,dict(settings,trajopt=optimize),p))
            for key,row in executor.map(plan_worker,jobs):
                if optimize:
                    scp_attempts.append(dict(key=list(key),stats=row['stats']))
                    if not row['passed'] or not row['stats'].get('trajopt',{}).get('accepted'): continue
                records[key]=row
        for iteration in range(iterations):
            cs,ts,sc,ec,st,et=fold_graph(layers,nodes,records,endpoints)
            solution=core.shortest_path([x['actions'] for x in layers],cs,sc,ec)
            if solution is None:break
            unknown=[key for key in route_keys(solution['indices']) if key not in records]
            rounds.append(dict(iteration=iteration,solution=solution,new_edges=[list(k) for k in unknown]))
            atomic(out/'progress.json',dict(rounds=rounds))
            if not unknown:break
            execute(unknown)
        def constrained():
            cs,ts,sc,ec,st,et=fold_graph(layers,nodes,records,endpoints,True)
            return core.shortest_path_with_time([x['actions'] for x in layers],cs,ts,sc,ec,st,et,time_budget=float(clock[-1]-clock[cut])+1e-9)
        selected=constrained()
        if selected is None:raise RuntimeError('Certified baseline disappeared')
        scp=[]
        if trajopt:
            eligible=[]
            for key in route_keys(selected['indices']):
                row=records[key];a,b=endpoints[key]
                states=np.load(row['file'])['states']
                if key[0]!=-1 and a[12]==b[12] and 2<len(edges.compress_collinear(states))<=40:eligible.append(key)
            execute(eligible,True)
            scp=scp_attempts
            selected=constrained()
    pieces=[];keys=route_keys(selected['indices'])
    for k,index in enumerate(selected['indices']):
        row=records[keys[k]];verify_bindings({row['file']:row['sha256']});pieces.append(dict(np.load(row['file'])))
        pieces.append(nodes[k][index]['piece'])
    row=records[keys[-1]];pieces.append(dict(np.load(row['file'])))
    prefix={k:v[:cut+1] for k,v in old.items()}
    unchanged=all(i==0 for i in selected['indices']) and not any(records[k]['stats'].get('trajopt',{}).get('accepted') for k in keys)
    trial={k:v.copy() for k,v in old.items()} if unchanged else trajectory.combine(prefix,pieces)
    before=trajectory.motion_metrics(old['states'],clock);after=trajectory.motion_metrics(trial['states'],trial.get('time',trial.get('times')))
    checks=dict(prefix_exact=all(np.array_equal(trial[k][:cut+1],v[:cut+1]) for k,v in old.items()),end_exact=np.array_equal(trial['states'][-1],old['states'][-1]))
    if not all(checks.values()):raise ValueError('Frozen prefix or endpoint changed')
    verdict=trajectory.acceptance(before,after)
    selected_actions=[dict(task=x['task'],weld_id=x['weld_id'],angle=x['angle'],arm=x['arm'],action=x['actions'][0]) for x in pool['layers'][:7]]
    selected_actions.extend(dict(task=x['task'],weld_id=x['weld_id'],angle=x['angle'],arm=x['arm'],action=x['actions'][i]) for x,i in zip(layers,selected['indices']))
    verify_bindings(bindings)
    verify_bindings({r['file']:r['sha256'] for r in records.values() if r['passed']})
    np.savez_compressed(out/'trial_route.npz',**trial)
    if verdict['accepted']:np.savez_compressed(out/'route.npz',**trial)
    report=dict(**verdict,before=before,after=after,manifest=dict(bindings=bindings,settings=dict(workers=workers,iterations=iterations,edge_seconds=edge_seconds,trajopt=trajopt)),pool_sha256=digest(pool_file),selected_actions=selected_actions,edges=list(records.values()),solver_rounds=rounds,solution=selected,scp=scp,checks=checks,prefix_end_index=cut,pool_action_endpoint_marker=action_endpoint_marker,prefix_boundary_reason='Preserve completed first seven weld actions; unlock existing post-task6 dual-arm handoff for task8 candidate configuration.',baseline_retained=None if verdict['accepted'] else pool['baseline_path'],no_hardware=True,timing_is_display_only=True,independent_cold_validation=False,scope='Finite sampled candidate graph; unknown edges excluded from final constrained solution.')
    atomic(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--edge-seconds',type=float,default=60.)
    parser.add_argument('--trajopt',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args();run(args.pool,args.out,args.workers,args.iterations,args.edge_seconds,args.trajopt)

if __name__=='__main__':main()

