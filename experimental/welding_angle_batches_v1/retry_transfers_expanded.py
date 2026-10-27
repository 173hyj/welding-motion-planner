"""Retry failed edges using expanded endpoint/folded gateway proposals first."""
import argparse,concurrent.futures,json,multiprocessing,shutil,sys,time
from pathlib import Path
import numpy as np
from .actions import atomic,digest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENGINE=None
OUT=None


def job(i,a,b):
    from world_bridge_v2 import SourceBridge
    from mixed_bridge_v95 import connect_fresh_park_options,SIGNS
    from run_transfer_limited_v156 import connect_via_park,solve_segment
    from signed_wrist_model_v40 import branch_parks
    started=time.perf_counter();deadline=started+600
    class Expired(RuntimeError):pass
    class Proposal(SourceBridge):
        def valid_scaled(self,state):
            if time.perf_counter()>deadline:raise Expired()
            return super().valid_scaled(state)
    bridge=Proposal(ENGINE);bridge.resolutions[:12]=np.deg2rad(.5)
    a,b=np.asarray(a)/bridge.scale,np.asarray(b)/bridge.scale
    path=None;trials=[]
    try:
        if path is None:
            try:
                path,detail=connect_fresh_park_options(bridge,a,b,200000+i,700)
                trials.append(dict(strategy='expanded_source_gateway',detail=detail))
            except ValueError as error:
                trials.append(dict(strategy='gateway_chart_unavailable',reason=str(error)))
        if path is None and abs(a[12]-b[12])<1e-12:
            path,detail=solve_segment(bridge,a,b,list(np.flatnonzero(abs(a-b)>1e-12)),300000+i,4000)
            trials.append(dict(strategy='full_rrt',detail=detail))
    except Expired:trials.append(dict(strategy='proposal_budget_expired'))
    fine=SourceBridge(ENGINE)
    passed=path is not None and all(fine.edge_scaled(x,y) for x,y in zip(path,path[1:]))
    result=dict(passed=bool(passed),elapsed_s=time.perf_counter()-started,trials=trials,fine_checks=fine.checks)
    if passed:
        name=f'paths/transfer_{i:02d}.json'
        atomic(OUT/name,dict(passed=True,full_source_accepted=True,frames=fine.dense_frames(path),
                            waypoints_scaled=[x.tolist() for x in path],independent_cold_performed=False,
                            source_resolution=fine.resolutions.tolist()))
        result.update(path_file=name,path_sha256=digest(OUT/name))
    else:result['failure']=fine.last_failure or bridge.last_failure
    atomic(OUT/f'edge_{i:02d}.json',result)
    print('RETRY_TRANSFER',i,passed,round(result['elapsed_s'],2),flush=True)
    return i,result


def main():
    global ENGINE,OUT
    p=argparse.ArgumentParser();p.add_argument('--resume-from',type=Path,required=True);p.add_argument('--actions',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();OUT=args.out.resolve()
    if OUT.exists():raise FileExistsError('Fresh retry output required')
    sys.path[:0]=[str(ROOT),str(HERE.parent/'welding_retreat_transfer_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    from world_bridge_v2 import node_state
    from integrated_retreat_v117 import online_retreat_ids
    source=args.resume_from.resolve();report=json.loads((source/'report.json').read_text())
    actions=json.loads((args.actions/'report.json').read_text());online_retreat_ids(actions,args.actions)
    if digest(args.actions/'report.json')!=report['actions_report_sha256'] or digest(args.baseline)!=report['baseline_sha256']:
        raise ValueError('Retry inputs differ from original transfer inputs')
    for section in ('source_sha256','code_sha256'):
        for name,h in report['manifest'][section].items():
            if digest(ROOT/name)!=h:raise ValueError('Bound transfer input changed: '+name)
    rows={r['weld_id']:r for r in actions['rows']};rids=report['robot_ids']
    states=[node_state(json.loads((args.actions/rows[wid]['best']['path_file']).read_text()),rids) for wid in report['order_weld_ids']]
    baseline=np.load(args.baseline);sequence=[baseline['states'][0]]+states+[baseline['states'][-1]]
    results=[report['entry']]+report['selected_edges']+[report['exit']]
    OUT.mkdir(parents=True);pending=[]
    for i,result in enumerate(results):
        if result['passed']:
            sourcepath=source/result['path_file']
            if digest(sourcepath)!=result['path_sha256']:raise ValueError('Accepted transfer changed')
            destination=OUT/result['path_file'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(sourcepath,destination)
            atomic(OUT/f'edge_{i:02d}.json',result)
        else:pending.append(i)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    initialize_worker();ENGINE=load_raw_world(scene,SIGNS)
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        jobs=[pool.submit(job,i,sequence[i],sequence[i+1]) for i in pending]
        for future in concurrent.futures.as_completed(jobs):
            i,result=future.result()
            if 0<i<len(results)-1:
                result.update(from_weld_id=report['order_weld_ids'][i-1],to_weld_id=report['order_weld_ids'][i])
            results[i]=result
            atomic(OUT/'progress.json',dict(finished=len(results)-len(pending)+sum((OUT/f'edge_{j:02d}.json').exists() for j in pending),passed=sum(bool(r['passed']) for r in results)))
    accepted=all(r['passed'] for r in results)
    report.update(entry=results[0],exit=results[-1],selected_edges=results[1:-1],full_source_accepted=accepted,
                  summary=dict(all_required_points_route_planned=accepted),
                  output_sha256={r['path_file']:r['path_sha256'] for r in results if r['passed']},
                  retry_source_report=dict(path=str(source/'report.json'),sha256=digest(source/'report.json')))
    report['manifest']['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    for section in ('source_sha256','code_sha256'):
        for name,h in report['manifest'][section].items():
            if digest(ROOT/name)!=h:raise ValueError('Input changed during retry')
    atomic(OUT/'report.json',report)
    if not accepted:raise SystemExit(2)


if __name__=='__main__':main()
