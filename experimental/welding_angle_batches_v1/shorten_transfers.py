"""Remove RRT detours with original fine checks and fixed rotation phases."""
import argparse,concurrent.futures,json,multiprocessing,shutil,sys,time
from pathlib import Path
import numpy as np
from .actions import atomic,digest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENGINE=None
OUT=None
SOURCE=None


def job(i,reference):
    from world_bridge_v2 import SourceBridge
    begun=time.perf_counter()
    path=json.loads((SOURCE/reference['path_file']).read_text())
    points=np.asarray(path['waypoints_scaled'])
    fine=SourceBridge(ENGINE)
    result=[points[0]];index=0
    while index<len(points)-1:
        target=index+1
        for j in range(len(points)-1,index+1,-1):
            # Never combine distinct external-axis rotation phases.
            if np.ptp(points[index:j+1,12])>1e-12:continue
            if fine.edge_scaled(points[index],points[j]):
                target=j;break
        result.append(points[target]);index=target
    result=np.asarray(result)
    # Keep the accepted source path if no corners could be removed.
    name=reference['path_file']
    if len(result)<len(points):
        path.update(frames=fine.dense_frames(result),waypoints_scaled=result.tolist(),
                    source_resolution=fine.resolutions.tolist(),
                    shortcut_source_sha256=reference['path_sha256'])
        atomic(OUT/name,path)
    else:
        (OUT/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,OUT/name)
    row=dict(reference,path_sha256=digest(OUT/name),shortcut_corners_before=len(points),shortcut_corners_after=len(result),fine_shortcut_checks=fine.checks)
    print('SHORTEN_TRANSFER',i,len(points),len(result),round(time.perf_counter()-begun,2),flush=True)
    return i,row


def main():
    global ENGINE,OUT,SOURCE
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();SOURCE=args.source.resolve();OUT=args.out.resolve()
    if OUT.exists():raise FileExistsError('Fresh output required')
    report=json.loads((SOURCE/'report.json').read_text())
    if not report['full_source_accepted']:raise ValueError('Complete certified route required')
    for section in ('source_sha256','code_sha256'):
        for name,h in report['manifest'][section].items():
            if digest(ROOT/name)!=h:raise ValueError('Bound input changed')
    for name,h in report['output_sha256'].items():
        if digest(SOURCE/name)!=h:raise ValueError('Bound path changed')
    sys.path[:0]=[str(ROOT),str(HERE.parent/'welding_retreat_transfer_v1')]
    from run_integrated_retreat_v118 import load_raw_world
    from run_geometry import initialize_worker
    from mixed_bridge_v95 import SIGNS
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    initialize_worker();ENGINE=load_raw_world(scene,SIGNS)
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    OUT.mkdir(parents=True)
    references=[report['entry']]+report['selected_edges']+[report['exit']]
    results=[None]*len(references)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        futures=[pool.submit(job,i,row) for i,row in enumerate(references)]
        for future in concurrent.futures.as_completed(futures):
            i,row=future.result();results[i]=row
    report.update(entry=results[0],exit=results[-1],selected_edges=results[1:-1],
                  shortcut_source_report=dict(path=str(SOURCE/'report.json'),sha256=digest(SOURCE/'report.json')),
                  output_sha256={r['path_file']:r['path_sha256'] for r in results})
    report['manifest']['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    for section in ('source_sha256','code_sha256'):
        for name,h in report['manifest'][section].items():
            if digest(ROOT/name)!=h:raise ValueError('Bound input changed during shortening')
    atomic(OUT/'report.json',report)


if __name__=='__main__':main()
