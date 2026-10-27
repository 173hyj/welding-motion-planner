"""Extend validated 200mm endpoints along their true retreat rays.

These are optional transfer gateways, not changes to the requested retreat.
Every saved joint edge undergoes the same complete source checks.
"""
import concurrent.futures
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'welding_variable_open_v1')]
from run_transfer_v5 import VariableToolWorld, assert_frozen_geometry_cache
from world_bridge_v2 import SourceBridge
from run_geometry import atomic,digest,initialize_worker
from arm_path import continuous_ik

ENGINE=None


def job(index,node,out):
    bridge=SourceBridge(ENGINE)
    start=np.asarray(node['state'])/bridge.scale
    source=json.loads(Path(node['retreat_path_file']).read_text())
    if digest(node['retreat_path_file'])!=node['retreat_path_sha256']:
        raise RuntimeError('Retreat changed')
    rid=source['robot_id'];i=bridge.robot_ids.index(rid)
    q=start[6*i:6*i+6].copy();initial=bridge.world.models[rid].tcp(q)
    direction=np.asarray(source['direction_world']);path=[start.copy()];current=start.copy()
    gateways=[];reason=None
    for distance in np.arange(5.,1200.+1e-8,5.):
        target=initial.copy();target[:3,3]+=distance*direction
        fit=continuous_ik(bridge.world.models[rid],target,q)
        if not fit['passed']:
            reason='continuous_ik_rejected';break
        next_state=current.copy();next_state[6*i:6*i+6]=fit['q_rad']
        if not bridge.edge_scaled(current,next_state):
            reason='full_source_edge_rejected';break
        path.append(next_state.copy());current=next_state;q=np.asarray(fit['q_rad'])
        if distance in (50.,100.,200.,400.,800.,1200.):
            gateways.append(dict(extra_cartesian_travel_mm=float(distance),
                                 state=(current*bridge.scale).tolist(),waypoint_count=len(path)))
    row=dict(index=index,weld_id=node['weld_id'],start_state=node['state'],
             gateways=gateways,reason=reason,full_source_checks=bridge.checks,
             waypoints=[(p*bridge.scale).tolist() for p in path],
             last_extra_cartesian_travel_mm=float((bridge.world.models[rid].tcp(q)[:3,3]-initial[:3,3])@direction))
    atomic(out/f'point_{index:02d}.json',row)
    print('CARTESIAN_EXTENSION',index,row['last_extra_cartesian_travel_mm'],reason,flush=True)
    return row


def main():
    global ENGINE
    out=ROOT/'build/welding_transfer_cartesian_gateways_v1'
    if out.exists():raise FileExistsError('Preserve evidence')
    assert_frozen_geometry_cache()
    source=ROOT/'build/welding_transfer_200mm_diagnostic_v6/nodes.json'
    nodes=json.loads(source.read_text())['nodes']
    manifest=dict(source_sha256={str(source.relative_to(ROOT)):digest(source)},
                  code_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                    'experimental/welding_retreat_transfer_v1/world_bridge_v2.py':digest(HERE/'world_bridge_v2.py')})
    out.mkdir();atomic(out/'manifest.json',manifest)
    t=time.perf_counter()
    ENGINE=VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,
        mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        futures=[pool.submit(job,i,n,out) for i,n in enumerate(nodes)]
        rows=[f.result() for f in concurrent.futures.as_completed(futures)]
    atomic(out/'report.json',dict(manifest=manifest,rows=rows,elapsed_s=time.perf_counter()-t,
        output_sha256={str(p.relative_to(out)):digest(p) for p in out.glob('point_*.json')},
        scope='Transfer gateway suggestions only; exact source-checked waypoint edges. No complete route claim, final combined cold verification required.'))


if __name__=='__main__':main()
