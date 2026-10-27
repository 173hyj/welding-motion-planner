"""Finite lazy search followed by known-edge-only rounds; all library nodes retained."""
import argparse
import json
from pathlib import Path
import numpy as np
from experimental.welding_retreat300_planning_v1 import run as base
from experimental.welding_candidate_selection_v1.pool import atomic, digest

SEALED={}
ORIGINAL_WORKER=base.transfer_worker

class KnownEdges:
    def __init__(self, records):self.records=records
    def get(self,key):return self.records.get(key,dict(passed=False))

def load_sealed(folder):
    index={};bindings={}
    for receipt in sorted((Path(folder)/'edges').glob('*.json')):
        row=json.loads(receipt.read_text())
        if not row.get('passed') or row.get('stats',{}).get('fine_certified') is not True:continue
        path=Path(row['file']).absolute()
        if digest(path)!=row['sha256']:raise ValueError('Sealed edge SHA256 mismatch: '+str(path))
        key=tuple(row['key'])
        if len(key)!=4:raise ValueError('Invalid sealed edge key')
        if key in index:raise ValueError('Duplicate sealed edge key')
        record=dict(row,reuse_source_json=str(receipt.absolute()),reuse_source_json_sha256=digest(receipt))
        index[key]=record
        bindings[str(path)]=row['sha256'];bindings[str(receipt.absolute())]=record['reuse_source_json_sha256']
    return index,bindings

def reusable(index,key,start,goal):
    row=index.get(key)
    if row is None:return None
    if digest(row['reuse_source_json'])!=row['reuse_source_json_sha256'] or digest(row['file'])!=row['sha256']:
        raise ValueError('Sealed previous edge changed')
    with np.load(row['file'],allow_pickle=False) as data:
        states=data['states'];times=data['times'];tcp=data['tcp']
        if states.ndim!=2 or states.shape[1]!=7 or len(states)<2 or times.shape!=(len(states),) or tcp.shape!=(len(states),3):raise ValueError('Invalid sealed edge arrays')
        if not all(np.isfinite(x).all() for x in (states,times,tcp)) or times[0]!=0 or np.any(np.diff(times)<0):raise ValueError('Invalid sealed edge timing')
        if not np.array_equal(states[[0,-1]],np.asarray([start,goal])):return None
        metrics=base.local_metrics(states,times)
        if any(not np.isclose(row[k],v,rtol=1e-10,atol=1e-10) for k,v in metrics.items() if k in row):raise ValueError('Sealed metrics mismatch')
    return dict(row,**metrics)

def materialize(row,key,path):
    with np.load(row['file'],allow_pickle=False) as data:np.savez_compressed(path,states=data['states'],times=data['times'],tcp=data['tcp'])
    result=dict(row,file=str(path),sha256=digest(path),stats=dict(row['stats'],reused_sealed_edge=True),reuse_source_file=row['file'],reuse_source_file_sha256=row['sha256'])
    result['key']=list(key)
    atomic(Path(path).with_suffix('.json'),result)
    return result

def transfer_worker(job):
    key,start,goal,angle,seconds,path=job
    row=reusable(SEALED,key,start,goal)
    return (key,materialize(row,key,path)) if row is not None else ORIGINAL_WORKER(job)

class GraphRounds:
    def __init__(self,original,iterations,known_rounds,out):
        self.original=original;self.iterations=iterations;self.known_rounds=known_rounds;self.out=Path(out);self.initial_stages=set();self.counts={};self.coverage=[]
    def __call__(self,layers,nodes,records,orders,starts,goals,stage):
        # v1 makes exactly one initial participation call for every fixed stage.
        if stage not in self.initial_stages:
            self.initial_stages.add(stage)
            return self.original(layers,nodes,records,orders,starts,goals,stage)
        iteration=self.counts.get(stage,0);self.counts[stage]=iteration+1
        endpoints={n['id']:n['endpoint'] for group in nodes.values() for n in group}
        for key in SEALED:
            if key[0]!=stage or key in records:continue
            _,arm,left,right=key
            a=starts[arm] if left=='start' else endpoints.get(left)
            b=goals[arm] if right=='end' else endpoints.get(right)
            if a is None or b is None:continue
            row=reusable(SEALED,key,a,b)
            if row is not None:records[key]=materialize(row,key,self.out/'edges'/f'reused_{len(records):05d}.npz')
        known=iteration>=self.iterations-self.known_rounds
        self.coverage.append(dict(stage=stage,iteration=iteration,known_edges_only=known,known_records=len(records),candidate_count=sum(len(nodes[t]) for order in orders for t in order)))
        return self.original(layers,nodes,KnownEdges(records) if known else records,orders,starts,goals,stage)

def optimize(pool,out,incumbent,previous,workers=8,iterations=36,edge_seconds=60.,known_rounds=2,reference_route=None,reference_sha256=None):
    global SEALED
    from experimental.welding_candidate_selection_v1.run import verify_bindings
    if known_rounds<1 or iterations<=known_rounds:raise ValueError('Require optimistic rounds and final known-only rounds')
    if not (Path(previous)/'edges').is_dir():raise ValueError('Previous edge directory missing')
    SEALED,bindings=load_sealed(previous)
    for path in Path(__file__).parent.glob('*.py'):bindings[str(path.absolute())]=digest(path)
    verify_bindings(bindings)
    original_graph=base.build_graph;original_worker=base.transfer_worker
    rounds=GraphRounds(original_graph,iterations,known_rounds,out)
    base.build_graph=rounds;base.transfer_worker=transfer_worker
    try:report=base.optimize(pool,out,incumbent,workers,iterations,edge_seconds,reference_route,reference_sha256)
    finally:base.build_graph=original_graph;base.transfer_worker=original_worker
    verify_bindings(bindings)
    report['manifest']['bindings']=base.merge_bindings(report['manifest']['bindings'],bindings)
    report['finite_known_edge_completion']=dict(known_rounds=known_rounds,rounds=rounds.coverage,previous_edges_available=len(SEALED),previous_edge_source=str(Path(previous).absolute()),bindings_verified_before_and_after=True,all_nodes_retained=True,unknown_edges_forbidden_only_in_final_rounds=True,optimality_claim=False,scope='Finite evaluated edge coverage; unavailable edges remain unresolved. Final scheduling and source validation use unchanged v1 gates.')
    # Bind every current edge receipt as well as its payload and inherited sources.
    for receipt in (Path(out)/'edges').glob('*.json'):report['manifest']['bindings'][str(receipt.absolute())]=digest(receipt)
    verify_bindings(report['manifest']['bindings'])
    atomic(Path(out)/'report.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','out','incumbent','previous'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--iterations',type=int,default=36);p.add_argument('--edge-seconds',type=float,default=60.);p.add_argument('--known-rounds',type=int,default=2)
    p.add_argument('--reference-route',type=Path);p.add_argument('--reference-sha256')
    a=p.parse_args();optimize(a.pool,a.out,a.incumbent,a.previous,a.workers,a.iterations,a.edge_seconds,a.known_rounds,a.reference_route,a.reference_sha256)

if __name__=='__main__':main()


