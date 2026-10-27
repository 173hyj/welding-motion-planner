"""Fail-closed global time-only retimer for complete source-certified 300mm trials.

Stage receipts retain their parent local clocks. interval_clock_map is the exact
mapping to the authoritative global clocks; geometry and all labels are untouched.
"""
import argparse, copy, hashlib, importlib, json, sys
from pathlib import Path
import numpy as np

BASELINE = 459.4458715768552
def require(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def same(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
def require_stage_labels(data,piece,lo,hi):
    # Concatenating incumbent and stage labels can promote unicode/int dtypes.
    # Cross-artifact identity is exact values; parent->child stays byte-exact.
    for k in ('tasks','phases'):
        require(np.array_equal(data[k][lo+1:hi+1],piece[k][1:]), 'Stage labels differ')

def retime(data, task_arms, audit_timing):
    a = {k:v.copy() for k,v in data.items()}
    q,t,tcp = a['states'],a['time'],a['tcp']
    require(same(t,a['times']), 'Duplicate clocks differ')
    require(q.shape == (len(t),15) and tcp.shape == (len(t),2,3), 'Invalid full15/TCP shape')
    require(len(t)>1 and all(np.isfinite(x).all() for x in (q,t,tcp)), 'Nonfinite samples')
    dt=np.diff(t); require(np.all(dt>0), 'Nonmonotone clock')
    d=abs(np.diff(q,axis=0))
    need=np.maximum.reduce([dt,np.rad2deg(d[:,:12]).max(1)/30,np.rad2deg(d[:,12])/15,d[:,13:].max(1)/80,np.linalg.norm(np.diff(tcp,axis=0),axis=2).max(1)/200])
    weld=np.any(a['phases'][:-1]=='weld',axis=1)
    # Allow only floating point noise; the unchanged v1 speed/dwell audit remains
    # mandatory after reconstruction. Never stretch any protected interval.
    require(not np.any(need[weld]>dt[weld]*(1+1e-12)+1e-14), 'Speed correction would stretch protected START weld interval')
    need[weld]=dt[weld]
    changed=need>dt
    need[changed]*=1+1e-10
    nt=np.r_[t[0],t[0]+np.cumsum(need)]
    require(np.all(np.diff(nt)>=dt-1e-10), 'Clock shortened')
    require(np.max(abs(np.diff(nt)[weld]-dt[weld]),initial=0)<1e-9, 'Weld clock changed')
    a['time']=nt; a['times']=nt.copy()
    require(all(same(a[k],v) for k,v in data.items() if k not in ('time','times')), 'Nonclock array changed')
    receipt=audit_timing(a,task_arms)
    require(receipt['passed'], 'v1 timing rejection: '+str(receipt.get('reasons')))
    return a,receipt,dict(parent_dt_s=dt.tolist(),retimed_dt_s=np.diff(nt).tolist(),changed_intervals=np.flatnonzero(changed).tolist(),protected_start_weld_intervals=np.flatnonzero(weld).tolist())

def run(parent, out, root):
    parent=Path(parent).absolute(); out=Path(out).absolute(); root=Path(root).absolute()
    require(not out.exists(), 'Output must be a fresh directory')
    reportfile=parent/'report.json'; report=json.loads(reportfile.read_text())
    require(report.get('complete') is True, 'Incomplete parent refused')
    bindings={}
    def bind(p,h=None):
        p=Path(p); p=p if p.is_absolute() else root/p
        require(p.is_file(), 'Missing bound source: '+str(p))
        actual=sha(p); require(h is None or actual==h, 'Source hash differs: '+str(p))
        bindings[str(p.absolute())]=actual
    require(bool(report.get('manifest',{}).get('bindings')), 'Missing parent manifest')
    for p,h in report['manifest']['bindings'].items(): bind(p,h)
    bind(reportfile); routefile=parent/'trial_route.npz'; bind(routefile); bind(__file__)
    selected=report['selected_actions']; task_arms={int(s['task']):int(s['arm']) for s in selected}
    require(len(selected)==21 and set(task_arms)==set(range(1,22)), 'All21 selection required')
    require(all(a in (0,1) for a in task_arms.values()), 'Invalid arm')
    stages=report['stages']; require(len(stages) in (5,6), 'Complete5/6 stages required')
    covered=[int(t) for s in stages for arm in s['task_ids'] for t in arm]
    require(sorted(covered)==list(range(1,22)), 'All21 stage coverage required exactly once')
    data=load(routefile); mapping=[]; cursor=0
    for stage in stages:
        bind(stage['receipt_file'],stage['receipt_sha256']); piece=load(stage['receipt_file'])
        v=stage['validation']; require(v.get('passed') and v.get('fine') and v.get('samples')==len(piece['states']), 'Missing source fine stage receipt')
        n=len(piece['states']); candidates=np.flatnonzero(np.all(data['states'][cursor:]==piece['states'][0],axis=1))+cursor
        matches=[int(i) for i in candidates if i+n<=len(data['states']) and same(data['states'][i:i+n],piece['states'])]
        require(len(matches)==1, 'Stage sequence absent or ambiguous')
        lo=matches[0]; hi=lo+n-1; cursor=hi+1
        require_stage_labels(data,piece,lo,hi)
        mapping.append(dict(stage=stage['stage'],global_start=lo,global_end_inclusive=hi,parent_receipt_file=stage['receipt_file'],parent_receipt_sha256=stage['receipt_sha256']))
    require(mapping[0]['global_start']==0 and mapping[-1]['global_end_inclusive']==len(data['states'])-1, 'Unaccounted prefix/suffix')
    for edge in report['edges']:
        if edge['passed']:
            bind(edge['file'],edge['sha256']); stats=edge['stats']
            require(stats.get('fine_certified') and stats.get('status')=='source_fine_certified', 'Missing source edge receipt')
    for s in selected: bind(s['action']['path_file'],s['action'].get('path_sha256'))
    sys.path.insert(0,str(root))
    timing=importlib.import_module('experimental.welding_retreat300_planning_v1.timing'); bind(timing.__file__)
    result,receipt,clockmap=retime(data,task_arms,timing.audit_timing)
    # Recheck every input immediately before publishing an artifact.
    for p,h in bindings.items(): require(sha(p)==h, 'Input changed during retiming: '+p)
    duration=float(result['time'][-1]-result['time'][0])
    reasons=[x for x in report.get('reasons',[]) if x not in ('actual_fullroute_timing_invalid','display_time_regression')]
    if duration>BASELINE+1e-7: reasons.append('valid_time_only_trial_exceeds_baseline')
    adopted=not reasons
    new=copy.deepcopy(report)
    new.update(complete=True,accepted=adopted,adopted=adopted,timing_audit=receipt,
        reasons=reasons,baseline_retained=None if adopted else report.get('incumbent_route'),
        interstage_policy='Exact parent geometry; time-only speed correction',
        source_stage_clock_parent=True,source_stage_global_mapping=mapping,
        time_only_retiming=dict(parent_report=str(reportfile),parent_report_sha256=sha(reportfile),parent_trial=str(routefile),parent_trial_sha256=sha(routefile),duration_s=duration,baseline_limit_s=BASELINE,interval_clock_map=clockmap,unchanged_arrays=[k for k in data if k not in ('time','times')]),
        source_validation_receipt=dict(ordered_source_geometry_unchanged=True,time_only=True,scope='Parent source geometry receipts retained for identical ordered coupled full15 polyline; no physical robot execution claim; parent stage clocks retained, global retimed clock authoritative.'))
    if 'after' in new: new['after']['duration_s']=duration
    out.mkdir(); trial=out/'trial_route.npz'; np.savez_compressed(trial,**result)
    bindings[str(trial)]=sha(trial)
    if adopted:
        target=out/'route.npz'; target.write_bytes(trial.read_bytes()); bindings[str(target)]=sha(target)
    new['manifest']['bindings']=bindings
    (out/'report.json').write_text(json.dumps(new,indent=2))
    return new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','out','root'): p.add_argument('--'+name,required=True)
    args=p.parse_args(); print(json.dumps(run(args.parent,args.out,args.root)['time_only_retiming'],indent=2))

