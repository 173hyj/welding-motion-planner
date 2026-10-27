"""Retry full-catalog selection with raw folded paths and sealed edge reuse."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from . import optimize as base

SEEDS=[]
REUSED={}


def transfer_worker(job):
    key,start,goal,angle,seconds,path=job
    from world_bridge_v2 import SourceBridge
    from state_space import interpolate
    from experimental.welding_candidate_selection_v1.pool import atomic,digest
    from experimental.dual_arm_parallel_v1.motion import section_times
    from .raw_transfer import make_transfer
    previous=REUSED.get(key)
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
    bridge=ProposalBridge(base.ENGINE)
    arm=key[1]
    try:
        corners,stats=make_transfer(base.ENGINE.world,bridge,arm,angle,start,goal,
            existing_edges=SEEDS,seed=17003+sum(ord(c) for c in str(key)),park7=base.PARKS[1-arm],max_iterations=1800)
    except Expired:
        corners=None;stats=dict(fine_certified=False,reason='finite_coarse_proposal_budget_expired')
    row=dict(key=list(key),passed=corners is not None,stats=stats)
    if corners is not None:
        dense=[]
        for a,b in zip(corners,corners[1:]):
            samples=interpolate(a,b,np.r_[np.full(6,np.deg2rad(.05)),.5])
            dense.extend(samples if not dense else samples[1:])
        states=np.asarray(dense) if dense else corners.copy();states[[0,-1]]=[start,goal]
        tcp=np.array([base.ENGINE.world.models[base.ROBOT_IDS[arm]].tcp(q[:6])[:3,3] for q in states])
        times=section_times(states,tcp)
        np.savez_compressed(path,states=states,times=times,tcp=tcp)
        row.update(file=str(path),sha256=digest(path),**base.local_metrics(states,times))
        stats.update(status='source_fine_certified',strategy=stats['method'],source_resolution_deg=.05,proposal_resolution_deg=.5)
    else:stats.update(status='unresolved',fine_certified=False)
    atomic(Path(path).with_suffix('.json'),row)
    return key,row


def main():
    global SEEDS,REUSED
    from experimental.welding_candidate_selection_v1.edges import compress_collinear
    from experimental.welding_candidate_selection_v1.run import verify_bindings
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','out','incumbent','previous'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--iterations',type=int,default=36)
    parser.add_argument('--edge-seconds',type=float,default=60.)
    args=parser.parse_args()
    previous_file=args.previous/'report.json';previous=json.loads(previous_file.read_text())
    verify_bindings(previous['manifest']['bindings'])
    from experimental.welding_candidate_selection_v1.pool import digest
    previous_sha=digest(previous_file)
    if previous['pool_sha256']!=digest(args.pool/'report.json'):raise ValueError('Previous graph uses a different pool')
    REUSED={tuple(row['key']):row for row in previous['edges'] if row['passed']}
    verify_bindings({row['file']:row['sha256'] for row in REUSED.values()})
    old=np.load(args.incumbent/'route.npz',allow_pickle=False)
    SEEDS=[dict(waypoints=compress_collinear(old['states']).tolist())]
    # base.optimize binds all source modules in this namespace, including this
    # worker and raw_transfer. Reused paths are hashed again before final output.
    base.transfer_worker=transfer_worker
    report=base.optimize(args.pool,args.out,args.incumbent,args.workers,args.iterations,args.edge_seconds)
    report['manifest']['bindings'][str(previous_file.resolve())]=previous_sha
    report['reused_graph_report_sha256']=previous_sha
    verify_bindings(report['manifest']['bindings'])
    from experimental.welding_candidate_selection_v1.pool import atomic
    atomic(args.out/'report.json',report)


if __name__=='__main__':main()
