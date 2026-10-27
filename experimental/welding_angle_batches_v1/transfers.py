"""Full-source certified transfers for a single positioner welding batch.

Coarse source samples propose paths only. Every saved edge is rechecked at
the original fine source resolution before it can be accepted.
"""
import argparse
import concurrent.futures
import json
import itertools
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np
from .actions import atomic, digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = None
OUT = None
SECONDS = None


def order_nodes(states, start, finish):
    """Multistart nearest neighbour and endpoint-aware 2-opt, raw axes."""
    if len(states)==0:
        return []
    states = np.asarray(states, float)
    scale = np.r_[np.full(12, np.deg2rad(30)), np.deg2rad(15), 80., 80.]
    all_states = np.vstack([states, start, finish])/scale
    costs = np.max(abs(all_states[:, None]-all_states[None, :]), axis=2)
    n = len(states)
    def value(route):
        full = [n]+route+[n+1]
        return sum(costs[a,b] for a,b in zip(full, full[1:]))
    best = None
    for first in range(n):
        route = [first]
        remaining = set(range(n))-{first}
        while remaining:
            j = min(remaining, key=lambda j:(costs[route[-1],j],j))
            route.append(j); remaining.remove(j)
        changed = True
        while changed:
            changed = False
            for i in range(n-1):
                for j in range(i+1,n):
                    candidate = route[:i]+route[i:j+1][::-1]+route[j+1:]
                    if value(candidate)<value(route)-1e-9:
                        route = candidate; changed = True; break
                if changed: break
        if best is None or value(route)<value(best)-1e-9:
            best = route
    return best or []


def order_batches(states, start, finish):
    groups={}
    for i,state in enumerate(states):
        groups.setdefault(round(float(state[12]),12),[]).append(i)
    if len(groups)>8:
        raise ValueError('More than 8 distinct validated stops needs a larger scheduling search')
    candidates=[]
    for angles in itertools.permutations(sorted(groups)):
        pairs=list(zip((start[12],)+angles,angles+(finish[12],)))
        travel=sum(abs(b-a) for a,b in pairs)
        movements=sum(abs(b-a)>1e-10 for a,b in pairs)
        candidates.append(((movements,travel),angles))
    minimum=min((x[0] for x in candidates),default=(0,0.))
    best=None
    for metric,angles in candidates:
        if metric[0]!=minimum[0] or metric[1]>minimum[1]+1e-10: continue
        order=[]; current=start
        for i,angle in enumerate(angles):
            members=groups[angle]
            next_state=finish if i+1==len(angles) else np.mean([states[j] for j in groups[angles[i+1]]],axis=0)
            local=order_nodes([states[j] for j in members],current,next_state)
            order.extend(members[j] for j in local)
            current=states[order[-1]]
        chain=[start]+[states[i] for i in order]+[finish]
        score=sum(np.max(abs(b[:12]-a[:12])) for a,b in zip(chain,chain[1:]))
        if best is None or score<best[0]: best=(score,order)
    return best[1] if best else []


def transfer_job(index, first, last):
    from world_bridge_v2 import SourceBridge
    from run_transfer_limited_v156 import solve_segment, connect_via_park
    from signed_wrist_model_v40 import branch_parks
    from mixed_bridge_v95 import SIGNS
    started = time.perf_counter()
    class BudgetExpired(RuntimeError): pass
    class ProposalBridge(SourceBridge):
        def valid_scaled(self, state):
            if time.perf_counter()>deadline: raise BudgetExpired()
            return super().valid_scaled(state)
    proposal = ProposalBridge(ENGINE)
    # Proposal-only resolution; final checker below retains 0.05 deg / 0.25 deg / 0.5mm.
    proposal.resolutions[:12] = np.deg2rad(.5)
    a,b = np.asarray(first)/proposal.scale, np.asarray(last)/proposal.scale
    deadline = started+SECONDS
    trials = []
    path = None
    try:
        if proposal.edge_scaled(a,b):
            path = [a,b]; trials.append(dict(strategy='direct'))
        else:
            parks = branch_parks(ENGINE.world.models, SIGNS)
            for order in ((0,1),(1,0)):
                path, stages = connect_via_park(proposal,a,b,index*100+len(trials),250,
                                               parks,order,False)
                trials.append(dict(strategy='park_gateway',arm_order=order,stages=stages))
                if path is not None: break
            if path is None and abs(a[12]-b[12])<1e-12:
                free = list(np.flatnonzero(abs(a-b)>1e-12))
                # Keep the positioner fixed for all within-stop transfers.
                path, stats = solve_segment(proposal,a,b,free,index+10000,700)
                trials.append(dict(strategy='rrt',stats=stats))
    except BudgetExpired:
        trials.append(dict(strategy='proposal_budget_expired'))
    fine = SourceBridge(ENGINE)
    passed = path is not None and all(fine.edge_scaled(x,y) for x,y in zip(path,path[1:]))
    name = f'paths/transfer_{index:02d}.json'
    result = dict(passed=bool(passed),elapsed_s=time.perf_counter()-started,
                  trials=trials,fine_checks=fine.checks,
                  source_resolution=fine.resolutions.tolist())
    if passed:
        frames = fine.dense_frames(path)
        atomic(OUT/name,dict(passed=True,frames=frames,waypoints_scaled=[p.tolist() for p in path],
                            full_source_accepted=True,independent_cold_performed=False,
                            source_resolution=fine.resolutions.tolist()))
        result.update(path_file=name,path_sha256=digest(OUT/name))
    else:
        result['failure'] = fine.last_failure or proposal.last_failure
    atomic(OUT/f'edge_{index:02d}.json',result)
    print('BATCH_TRANSFER',index,result['passed'],round(result['elapsed_s'],2),fine.checks,flush=True)
    return index,result


def main():
    global ENGINE, OUT, SECONDS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actions',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--edge-seconds',type=float,default=180)
    args=p.parse_args()
    OUT=args.out.resolve(); SECONDS=args.edge_seconds
    if OUT.exists(): raise FileExistsError('Fresh output required')
    sys.path[:0]=[str(ROOT),str(HERE.parent/'welding_retreat_transfer_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from integrated_retreat_v117 import online_retreat_ids
    from world_bridge_v2 import node_state
    report=json.loads((args.actions/'report.json').read_text())
    bound={group:dict(report['manifest'][group]) for group in ('source_sha256','code_sha256')}
    bound['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    for group in ('source_sha256','code_sha256'):
        for name,expected in bound[group].items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Action input changed: '+name)
    ids=online_retreat_ids(report,args.actions)
    if len(ids)!=21 or ids!=set(report['manifest']['required_weld_ids']):
        raise ValueError('All 21 complete local actions required')
    rows=sorted(report['rows'],key=lambda row:row['weld_id'])
    baseline=np.load(args.baseline)
    start,finish=baseline['states'][[0,-1]]
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    initialize_worker(); ENGINE=load_raw_world(scene,SIGNS)
    for rid,model in list(ENGINE.world.models.items()):
        ENGINE.world.models[rid]=getattr(model,'source',model)
    rids=sorted(ENGINE.world.models)
    states=[node_state(json.loads((args.actions/r['best']['path_file']).read_text()),rids) for r in rows]
    order=order_batches(states,start,finish)
    sequence=[start]+[states[i] for i in order]+[finish]
    OUT.mkdir(parents=True)
    atomic(OUT/'proposal.json',dict(order_weld_ids=[rows[i]['weld_id'] for i in order],robot_ids=rids))
    results=[None]*(len(sequence)-1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
             mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        jobs=[pool.submit(transfer_job,i,a,b) for i,(a,b) in enumerate(zip(sequence,sequence[1:]))]
        for future in concurrent.futures.as_completed(jobs):
            i,result=future.result();results[i]=result
            atomic(OUT/'progress.json',dict(finished=sum(x is not None for x in results),passed=sum(bool(x and x['passed']) for x in results)))
    accepted=all(row['passed'] for row in results)
    for group in ('source_sha256','code_sha256'):
        for name,expected in bound[group].items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Input changed during transfer planning: '+name)
    edges=[dict(result,from_weld_id=rows[order[i]]['weld_id'],to_weld_id=rows[order[i+1]]['weld_id'])
           for i,result in enumerate(results[1:-1])]
    atomic(OUT/'report.json',dict(robot_ids=rids,order_weld_ids=[rows[i]['weld_id'] for i in order],
          entry=results[0],exit=results[-1],selected_edges=edges,full_source_accepted=accepted,
          manifest=bound,
          summary=dict(all_required_points_route_planned=accepted),
          actions_report_sha256=digest(args.actions/'report.json'),baseline_sha256=digest(args.baseline),
          output_sha256={row['path_file']:row['path_sha256'] for row in results if row['passed']},
          independent_cold_performed=False,sampled_source_validation=True))
    if not accepted: raise SystemExit(2)


if __name__=='__main__': main()
