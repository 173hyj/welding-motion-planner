"""Rebuild fixed fresh300 parent topology with per-arm nonweld timing margin.

All source dual-timeline checks are rerun by unchanged anchored_v3.optimize.
Only sealed edge geometry is reused. No RRT fallback or inherited stage receipt
is accepted as proof of the changed coupled timeline.
"""
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np

SEALED={}
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(bindings):
    for p,h in bindings.items():require(digest(p)==h,'Bound source changed: '+p)
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def with_headroom(original,factor):
    require(np.isfinite(factor) and 1.001<=factor<=1.05,'Headroom factor must be finite in [1.001,1.05]')
    def section_times(states,tcp=None):return original(states,tcp)*factor
    return section_times

class FixedGraph:
    def __init__(self,base,original,parent):
        self.base=base;self.original=original;self.initial=set()
        self.orders={s['stage']:copy.deepcopy(s['task_ids']) for s in parent['stages']}
        self.ids={int(s['task']):s['action']['id'] for s in parent['selected_actions']}
    def __call__(self,layers,nodes,records,orders,starts,goals,stage):
        if stage not in self.initial:
            self.initial.add(stage)
            # Keep the actual all680 initial graph/participation calculation.
            return self.original(layers,nodes,records,orders,starts,goals,stage)
        for arm,order in enumerate(self.orders[stage]):orders[arm][:]=order
        ordered=[t for order in orders for t in order];indices=[];edges=set()
        for task in ordered:
            choices=[i for i,node in enumerate(nodes[task]) if node['id']==self.ids[task]]
            require(len(choices)==1,'Fixed fresh300 parent candidate absent or duplicated')
            indices.append(choices[0])
        for arm,order in enumerate(orders):
            ids=['start']+[self.ids[t] for t in order]+['end']
            edges.update((stage,arm,a,b) for a,b in zip(ids,ids[1:]))
        self.base.CURRENT_SOLUTION=dict(indices=indices,cost=[0.,0.,0.])
        return ordered,[],np.array([]),np.array([]),edges

def reusable(index,key,start,goal,local_metrics):
    require(key in index,'Unknown edge forbidden; no RRT fallback: '+str(key))
    row=index[key]
    require(digest(row['file'])==row['sha256'] and digest(row['reuse_source_json'])==row['reuse_source_json_sha256'],'Sealed edge changed')
    stats=row['stats'];require(row['passed'] and stats.get('fine_certified') is True and stats.get('status')=='source_fine_certified','Missing sealed source fine edge')
    with np.load(row['file'],allow_pickle=False) as z:
        q,t,tcp=z['states'],z['times'],z['tcp']
        require(q.ndim==2 and q.shape[1]==7 and len(q)>=2 and t.shape==(len(q),) and tcp.shape==(len(q),3),'Invalid sealed edge shape')
        require(all(np.isfinite(a).all() for a in (q,t,tcp)) and t[0]==0 and np.all(np.diff(t)>=0),'Invalid sealed edge arrays')
        require(np.array_equal(q[[0,-1]],np.asarray([start,goal])),'Sealed edge endpoints differ')
        metrics=local_metrics(q,t)
        require(all(np.isclose(row[k],v,rtol=1e-10,atol=1e-10) for k,v in metrics.items() if k in row),'Sealed edge original metrics differ')
    return dict(row,**metrics)

def transfer_worker(job):
    from experimental.welding_retreat300_anchored_v3 import run as base
    key,start,goal,angle,seconds,path=job
    row=reusable(SEALED,key,start,goal,base.local_metrics)
    with np.load(row['file'],allow_pickle=False) as z:
        np.savez_compressed(path,states=z['states'],times=z['times'],tcp=z['tcp'])
    result=dict(row,file=str(path),sha256=digest(path),key=list(key),stats=dict(row['stats'],reused_sealed_edge=True),reuse_source_file=row['file'],reuse_source_file_sha256=row['sha256'])
    atomic(Path(path).with_suffix('.json'),result)
    return key,result

def load_parent(parent,root):
    folder=Path(parent).absolute();reportfile=folder/'report.json';r=json.loads(reportfile.read_text());bindings={}
    require(r.get('complete') is True and len(r.get('stages',[]))==6,'Complete six-stage parent required')
    require(r.get('canonical_park_barriers') is False,'Literal anchored parent required')
    require(r.get('all_candidate_action_count')==680,'Fresh300 all680 parent required')
    selected=r['selected_actions'];require(len(selected)==21 and {int(s['task']) for s in selected}==set(range(1,22)),'All21 selected actions required')
    require(all(s['action'].get('length_mm')==300 for s in selected),'Only fresh300 selected actions allowed')
    require(sorted(int(t) for s in r['stages'] for order in s['task_ids'] for t in order)==list(range(1,22)),'All21 stage coverage required')
    def bind(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else Path(root)/p
        h=digest(p);require(expected is None or h==expected,'Parent source hash differs: '+str(p));bindings[str(p.absolute())]=h
    require(bool(r.get('manifest',{}).get('bindings')),'Missing parent manifest')
    for p,h in r['manifest']['bindings'].items():bind(p,h)
    bind(reportfile);bind(folder/'trial_route.npz')
    for s in selected:bind(s['action']['path_file'],s['action'].get('path_sha256'))
    for s in r['stages']:
        bind(s['receipt_file'],s['receipt_sha256'])
        with np.load(s['receipt_file'],allow_pickle=False) as z:n=len(z['states'])
        v=s['validation'];require(v.get('passed') and v.get('fine') and v.get('samples')==n,'Parent stage source fine validation required')
    jsons={}
    for file in sorted((folder/'edges').glob('*.json')):
        row=json.loads(file.read_text())
        if row.get('passed'):jsons.setdefault(tuple(row['key']),[]).append((file,row))
    ids={s['task']:s['action']['id'] for s in selected};required=set()
    for s in r['stages']:
        for arm,order in enumerate(s['task_ids']):
            chain=['start']+[ids[t] for t in order]+['end']
            required.update((s['stage'],arm,a,b) for a,b in zip(chain,chain[1:]))
    parent_rows={tuple(e['key']):e for e in r['edges']};index={}
    for key in required:
        require(key in parent_rows and parent_rows[key]['passed'],'Parent selected edge unavailable')
        row=parent_rows[key];stats=row.get('stats',{})
        require(stats.get('fine_certified') is True and stats.get('status')=='source_fine_certified','Parent selected edge lacks fine proof')
        matches=[(p,e) for p,e in jsons.get(key,[]) if e['sha256']==row['sha256'] and e['file']==row['file']]
        require(len(matches)==1,'Parent edge JSON absent or ambiguous')
        receipt,_=matches[0];bind(receipt);bind(row['file'],row['sha256'])
        index[key]=dict(row,reuse_source_json=str(receipt.absolute()),reuse_source_json_sha256=digest(receipt))
    verify(bindings);return r,index,bindings

def optimize(parent,pool,out,incumbent,workers=8,headroom_factor=1.01,reference_route=None,reference_sha256=None):
    global SEALED
    from experimental.welding_retreat300_anchored_v3 import run as base
    from experimental.dual_arm_parallel_v1 import motion
    out=Path(out).absolute();require(not out.exists(),'Fresh output directory required')
    original_times=motion.section_times;patched=with_headroom(original_times,headroom_factor)
    report,SEALED,bindings=load_parent(parent,base.ROOT)
    poolfile=Path(pool)/'report.json'
    require(digest(poolfile)==report['pool_sha256'],'Pool differs from parent')
    require(digest(Path(incumbent)/'route.npz')==report['incumbent_sha256'],'Incumbent differs from parent')
    for file in Path(__file__).absolute().parent.glob('*.py'):bindings[str(file)]=digest(file)
    verify(bindings)
    original_graph=base.build_graph;original_worker=base.transfer_worker
    original_stages=copy.deepcopy(base.STAGES);original_solution=base.CURRENT_SOLUTION
    fixed=FixedGraph(base,original_graph,report)
    motion.section_times=patched;base.build_graph=fixed;base.transfer_worker=transfer_worker
    try:
        new=base.optimize(pool,out,incumbent,workers,3,1.,reference_route,reference_sha256)
    finally:
        motion.section_times=original_times;base.build_graph=original_graph;base.transfer_worker=original_worker
        base.STAGES=original_stages;base.CURRENT_SOLUTION=original_solution
    verify(bindings)
    if new.get('complete'):
        require([s['task_ids'] for s in new['stages']]==[s['task_ids'] for s in report['stages']],'Rebuild order differs')
        require({s['task']:s['action']['id'] for s in new['selected_actions']}=={s['task']:s['action']['id'] for s in report['selected_actions']},'Rebuild selected300 IDs differ')
        require(all(s['validation'].get('passed') and s['validation'].get('fine') for s in new['stages']),'Fresh full coupled source validation missing')
    new['manifest']['bindings']=base.merge_bindings(new['manifest']['bindings'],bindings)
    for receipt in (out/'edges').glob('*.json'):new['manifest']['bindings'][str(receipt.absolute())]=digest(receipt)
    for filename in ('trial_route.npz','route.npz'):
        file=out/filename
        if file.exists():new['manifest']['bindings'][str(file)]=digest(file)
    new['timing_headroom_rebuild']=dict(parent_report=str((Path(parent)/'report.json').absolute()),parent_report_sha256=digest(Path(parent)/'report.json'),headroom_factor=headroom_factor,nonweld_per_arm_section_clock_scaled=True,weld_duration_override_s=.6,parent_edge_geometry_only_reused=True,all680_initial_participation_preserved=True,fixed_parent_selected_topology=True,rrt_fallback=False,source_full_coupled_timeline_revalidated=bool(new.get('complete')),source_fine_stage_count=len(new['stages']),full_collision_revalidation_scope='Each newly coupled stage reruns coarse then fine TimelineValidator against raw source; inherited literal interstage geometry retained.',bindings_verified_before_and_after=True,scope='Fresh coupled geometry/time source receipts; no hardware execution claim. Global final timing gate may still reject inherited interstage speed and must not be relaxed.')
    verify(new['manifest']['bindings']);atomic(out/'report.json',new);return new

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','pool','out','incumbent'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--headroom-factor',type=float,default=1.01)
    p.add_argument('--reference-route',type=Path);p.add_argument('--reference-sha256')
    a=p.parse_args();optimize(a.parent,a.pool,a.out,a.incumbent,a.workers,a.headroom_factor,a.reference_route,a.reference_sha256)
if __name__=='__main__':main()
