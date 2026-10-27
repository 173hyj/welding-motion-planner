"""Normalize only START metadata at exact source-stage global seams.

Every kinematic, TCP, clock, stage and other array remains byte-identical.
This is a metadata correction, never a time-only retiming or collision rerun.
"""
import argparse,copy,hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
BASELINE=459.4458715768552
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def normalize(data,stages):
    out={k:v.copy() for k,v in data.items()};mapping=[];repairs=[];cursor=0
    q=data['states'];require(q.ndim==2 and q.shape[1]==15,'Full15 states required')
    require(same(data['time'],data['times']),'Duplicate parent clocks differ')
    for s in stages:
        piece=s['data'];n=len(piece['states'])
        require(n>=2,'Stage must include an interval')
        candidates=np.flatnonzero(np.all(q[cursor:]==piece['states'][0],axis=1))+cursor
        matches=[int(i) for i in candidates if i+n<=len(q) and same(q[i:i+n],piece['states'])]
        require(len(matches)==1,'Stage geometry absent or ambiguous')
        lo=matches[0];hi=lo+n-1;cursor=hi+1
        require(np.allclose(data['time'][lo:hi+1]-data['time'][lo],piece['time']-piece['time'][0],atol=1e-9,rtol=0),'Source stage clock differs')
        for key in ('phases','tasks'):
            # Source-stage and concatenated-global storage dtypes may differ.
            # Exact label values suffice across artifacts; child preserves its
            # parent's dtype and all untouched bytes below.
            require(np.array_equal(data[key][lo+1:hi+1],piece[key][1:]),'Nonboundary stage '+key+' mismatch')
        before_phases=out['phases'][lo].tolist();before_tasks=out['tasks'][lo].tolist()
        changed=not np.array_equal(out['phases'][lo],piece['phases'][0]) or not np.array_equal(out['tasks'][lo],piece['tasks'][0])
        if changed:
            out['phases'][lo]=piece['phases'][0];out['tasks'][lo]=piece['tasks'][0]
            require(np.array_equal(out['phases'][lo],piece['phases'][0]) and np.array_equal(out['tasks'][lo],piece['tasks'][0]),'Parent label dtype cannot represent source labels')
            repairs.append(dict(stage=s['stage'],index=lo,before_phases=before_phases,before_tasks=before_tasks,after_phases=out['phases'][lo].tolist(),after_tasks=out['tasks'][lo].tolist(),source_receipt_sha256=s['receipt_sha256']))
        mapping.append(dict(stage=s['stage'],global_start=lo,global_end_inclusive=hi,parent_receipt_file=s['receipt_file'],parent_receipt_sha256=s['receipt_sha256']))
    require(mapping and mapping[0]['global_start']==0 and mapping[-1]['global_end_inclusive']==len(q)-1,'Unaccounted route prefix/suffix')
    permitted=np.zeros(len(q),bool);permitted[[r['index'] for r in repairs]]=True
    require(all(same(out[k],v) for k,v in data.items() if k not in ('phases','tasks')),'Nonmetadata array changed')
    for key in ('phases','tasks'):require(same(out[key][~permitted],data[key][~permitted]),'Nonboundary metadata changed')
    return out,mapping,repairs

def run(parent,out,root):
    parent=Path(parent).absolute();out=Path(out).absolute();root=Path(root).absolute()
    require(not out.exists(),'Fresh output directory required')
    reportfile=parent/'report.json';r=json.loads(reportfile.read_text())
    require(r.get('complete') is True and len(r.get('stages',[]))==6,'Complete six-stage parent required')
    require(r.get('all_candidate_action_count')==680,'All680 source parent required')
    selected=r['selected_actions'];arms={int(s['task']):int(s['arm']) for s in selected}
    require(len(selected)==21 and set(arms)==set(range(1,22)) and all(a in (0,1) for a in arms.values()),'All21 selected source actions required')
    require(all(s['action'].get('length_mm')==300 for s in selected),'Fresh300 selected actions required')
    require(sorted(int(t) for s in r['stages'] for order in s['task_ids'] for t in order)==list(range(1,22)),'All21 stage coverage required exactly once')
    bindings={}
    def bind(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else root/p
        h=sha(p);require(expected is None or expected==h,'Bound source hash differs: '+str(p));bindings[str(p.absolute())]=h
    require(bool(r.get('manifest',{}).get('bindings')),'Missing source manifest')
    for p,h in r['manifest']['bindings'].items():bind(p,h)
    routefile=parent/'trial_route.npz';bind(reportfile);bind(routefile);bind(__file__)
    stages=[]
    for s in r['stages']:
        bind(s['receipt_file'],s['receipt_sha256']);piece=load(s['receipt_file']);v=s['validation']
        require(v.get('passed') and v.get('fine') and v.get('samples')==len(piece['states']),'Source fine stage proof required')
        stages.append(dict(s,data=piece))
    for e in r['edges']:
        if e['passed']:
            bind(e['file'],e['sha256']);v=e['stats'];require(v.get('fine_certified') and v.get('status')=='source_fine_certified','Source fine edge proof required')
    for s in selected:bind(s['action']['path_file'],s['action'].get('path_sha256'))
    data=load(routefile);normalized,mapping,repairs=normalize(data,stages)
    sys.path.insert(0,str(root));timing=importlib.import_module('experimental.welding_retreat300_planning_v1.timing');bind(timing.__file__)
    receipt=timing.audit_timing(normalized,arms)
    require(len(receipt.get('dwells',[]))==21 and all(w['passed'] for w in receipt['dwells']),'Normalization cannot repair or invalidate 0.6s dwells')
    duration=float(normalized['time'][-1]-normalized['time'][0])
    reasons=[x for x in r.get('reasons',[]) if x not in ('actual_fullroute_timing_invalid','display_time_regression','display_time_regression_or_hard_limit')]
    if not receipt['passed']:reasons.append('actual_fullroute_timing_invalid')
    if duration>BASELINE+1e-7:reasons.append('display_time_regression_or_hard_limit')
    accepted=not reasons
    new=copy.deepcopy(r);new.update(accepted=accepted,adopted=accepted,reasons=reasons,timing_audit=receipt,baseline_retained=None if accepted else r.get('incumbent_route'))
    new['boundary_label_normalization']=dict(parent_report=str(reportfile),parent_report_sha256=sha(reportfile),parent_route=str(routefile),parent_route_sha256=sha(routefile),source_stage_receipts=mapping,repairs=repairs,kinematics_and_clocks_unchanged=True,changed_arrays=['phases','tasks'] if repairs else [],duration_s=duration,phase_convention='START phase governs [time[i],time[i+1])',scope='Only stage-start phases/tasks restored from exact source-fine stage receipt; all ordered coupled geometry, TCP and clocks unchanged. Parent source geometry proofs retained; no hardware execution claim.')
    for p,h in bindings.items():require(sha(p)==h,'Source changed during normalization: '+p)
    out.mkdir();trial=out/'trial_route.npz';np.savez_compressed(trial,**normalized);bindings[str(trial)]=sha(trial)
    if accepted:
        target=out/'route.npz';target.write_bytes(trial.read_bytes());bindings[str(target)]=sha(target)
    new['manifest']['bindings']=bindings;(out/'report.json').write_text(json.dumps(new,indent=2));return new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','out','root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a.parent,a.out,a.root)['boundary_label_normalization'],indent=2))
