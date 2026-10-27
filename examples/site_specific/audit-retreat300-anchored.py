"""Independent literal six-stage all300 artifact and time-only retiming audit."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

POOL_SHA='d89408cc7a45ac2626be71f085b43421b02bed519740998c8a28788067c016d6'
BASE_SHA='ba7e3c00918e053a7fbc9ed58d828836f0e83ad895481748c729a2f82fa7ec0f'
HELPER_SHA='aa6db257fa40dc0dee0b337bc1a3095209167073b64ffb2dd02b0623f85b8974'
def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def on_polyline(samples,line,tol=1e-8):
    from scipy.spatial import cKDTree
    samples,line=np.asarray(samples,float),np.asarray(line,float)
    require(len(samples)>0 and len(line)>0,'Missing action/transfer polyline')
    if len(line)==1: error=float(abs(samples-line[0]).max())
    else:
        near=cKDTree(line).query(samples,k=min(8,len(line)))[1]
        near=np.asarray(near).reshape(len(samples),-1); best=np.full(len(samples),np.inf)
        for col in range(near.shape[1]):
            for offset in (-1,0):
                i=np.clip(near[:,col]+offset,0,len(line)-2); a,b=line[i],line[i+1]; d=b-a; n=(d*d).sum(1)
                u=np.divide(((samples-a)*d).sum(1),n,out=np.zeros_like(n),where=n>0)
                best=np.minimum(best,abs(samples-(a+np.clip(u,0,1)[:,None]*d)).max(1))
        error=float(best.max())
    require(error<=tol,'Samples depart from sealed polyline'); return error

def require_action_angles(q,tasks,phases,task,arm,angle):
    active=(np.asarray(tasks)[:,arm]==task)&np.isin(np.asarray(phases)[:,arm],('ingress','close','weld','open','retreat'))
    require(np.any(active),'Missing named action phases')
    require(np.allclose(np.asarray(q)[active,12],np.deg2rad(angle),atol=1e-12,rtol=0),'Weld/action angle differs')

def audit_geometry(args):
    root=Path(args.root).resolve(); folder=Path(args.result).resolve(); poolfile=Path(args.pool).resolve()
    require(sha(poolfile)==POOL_SHA,'Library SHA differs'); pool=read(poolfile)
    result=dict(passed=False,independent_cold_full_collision=False,auditor_sha256=sha(__file__),pool_sha256=POOL_SHA)
    helper=root/'experimental/welding_retreat_250mm_v1/exact_prefix_v4.py'
    require(sha(helper)==HELPER_SHA,'Additional seed helper changed')
    result['additional_binding']={str(helper):HELPER_SHA}
    for p,h in pool['manifest']['bindings'].items():
        p=Path(p); require(sha(p if p.is_absolute() else root/p)==h,'Pool binding changed: '+str(p))
    expected={int(l['task']):l for l in pool['layers']}; require(len(expected)==21,'Expected21 layers')
    require(sum(len(l['actions']) for l in expected.values())==680,'Expected680 eligible actions')
    for l in expected.values():
        for a in l['actions']:
            require(a['length_mm']==300.,'Non300 library action')
            require(sha(a['path_file'])==a['path_sha256'],'Library action changed')
    if not (folder/'report.json').exists():
        require(args.allow_incomplete,'No final report'); result.update(passed=True,complete=False,scope='Library and helper binding metadata only; no final route available'); return result
    r=read(folder/'report.json');result['report_sha256']=sha(folder/'report.json')
    for p,h in r['manifest']['bindings'].items():
        p=Path(p);require(sha(p if p.is_absolute() else root/p)==h,'Planner binding changed: '+str(p))
    require(r['pool_sha256']==POOL_SHA and r['incumbent_sha256']==BASE_SHA and sha(r['incumbent_route'])==BASE_SHA,'Input provenance changed')
    require(r['all_candidate_action_count']==680,'Eligible action count differs')
    for rows in (r['all21_graph_nodes'],[x for s in r['participation'] for x in s['layers']]):
        require(len(rows)==21 and len({x['task'] for x in rows})==21,'Missing participation layer')
        for x in rows:require(x['action_ids']==[a['id'] for a in expected[x['task']]['actions']],'Graph participation differs')
    require(r['fixed_angle_sequence']==[30,120,60,0,120,90],'Angle batches differ')
    for e in r['edges']:
        if e['passed']:
            require(sha(e['file'])==e['sha256'],'Edge changed')
            require(e['stats']['fine_certified'] and e['stats']['status']=='source_fine_certified' and e['stats']['source_resolution_deg']==.05,'Missing source fine edge receipt')
    if not r['complete']:
        require(args.allow_incomplete and not r['accepted'] and not r['adopted'] and not (folder/'route.npz').exists(),'Incomplete route adopted')
        result.update(passed=True,complete=False,accepted=False,scope='All680 participation and existing binding/edge receipts checked; final route unavailable');return result
    routefile=folder/('route.npz' if r['accepted'] else 'trial_route.npz'); d=dict(np.load(routefile,allow_pickle=False)); q=d['states']; t=d['time']
    require(q.shape==(len(t),15) and np.isfinite(q).all() and np.all(np.diff(t)>0),'Invalid trajectory')
    require(np.array_equal(t,d['times']),'Duplicate clocks differ'); base=np.load(r['incumbent_route'],allow_pickle=False)
    require(np.allclose(q[[0,-1]],base['states'][[0,-1]],atol=1e-12,rtol=0),'Original full15 endpoints changed')
    require(len(r['stages'])==6 and len(r['selected_actions'])==21,'Incomplete final selection')
    original=[(30,[[7],[1,2]]),(120,[[],[3,4,5,6]]),(60,[[8,9],[]]),(0,[[10,11],[]]),(120,[[12,13,14,15,16,17,18,19,20],[]]),(90,[[21],[]])]
    source_bounds=[]
    for angle,orders in original:
        starts=[];ends=[]
        for arm,order in enumerate(orders):
            for task in order:
                require(expected[task]['arm']==arm and expected[task]['angle']==angle,'Original task assignment differs')
                taskmask=base['tasks'][:,arm]==task
                ingress=np.flatnonzero(taskmask&(base['phases'][:,arm]=='ingress'));retreat=np.flatnonzero(taskmask&(base['phases'][:,arm]=='retreat'))
                require(len(ingress)>0 and len(retreat)>0,'Original stage task absent')
                starts.append(max(0,int(ingress[0])-1));ends.append(int(retreat[-1]))
        source_bounds.append([min(starts),max(ends)])
    require(source_bounds==r['stage_bounds'],'Original baseline stage bounds differ')
    require(sorted(int(task) for st in r['stages'] for order in st['task_ids'] for task in order)==list(range(1,22)),'Stage task coverage differs')

    for st,(angle,orders) in zip(r['stages'],original):
        require(st['angle']==angle and len(st['task_ids'])==2 and all(sorted(st['task_ids'][arm])==sorted(orders[arm]) for arm in range(2)),'Original stage/arm task assignment differs')
    selected={x['task']:x for x in r['selected_actions']};require(set(selected)==set(expected),'Task set differs')
    sys.path[:0]=[str(root),str(root/'experimental/welding_retreat_transfer_v1'),str(root/'experimental/welding_variable_open_v1')]
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    # Source stage labels are authoritative at the shared first sample: legacy combine drops them.
    labels={k:d[k].copy() for k in ('phases','tasks')};canonical_ranges=[];search=0
    for st in r['stages']:
        require(sha(st['receipt_file'])==st['receipt_sha256'],'Source stage changed')
        piece=dict(np.load(st['receipt_file'],allow_pickle=False));n=len(piece['states'])
        candidates=np.flatnonzero(np.all(q[search:]==piece['states'][0],axis=1))+search
        matches=[int(i) for i in candidates if i+n<=len(q) and np.array_equal(q[i:i+n],piece['states'])]
        require(len(matches)==1,'Source stage absent or ambiguous');lo=matches[0];hi=lo+n-1;search=hi+1
        for k in labels:
            require(np.array_equal(d[k][lo+1:hi+1],piece[k][1:]),'Nonboundary stage labels differ')
            labels[k][lo]=piece[k][0]
        canonical_ranges.append(dict(stage=st['stage'],global_start=lo,global_end_inclusive=hi,parent_receipt_file=st['receipt_file'],parent_receipt_sha256=st['receipt_sha256']))
    canonical_valid=all(np.array_equal(labels[k],d[k]) for k in labels)
    checks=[]
    for task,l in expected.items():
        s=selected[task]; a=s['action']; require(a in l['actions'] and all(s[k]==l[k] for k in ('arm','angle','weld_id')),'Selected library identity changed')
        p=read(a['path_file']);arm=l['arm']; own=q[:,np.r_[np.arange(arm*6,arm*6+6),13+arm]]
        ids=labels['tasks'][:,arm]==task; weld=ids&(labels['phases'][:,arm]=='weld')
        require(np.sum(weld&~np.r_[False,weld[:-1]])==1,'Weld missing or repeated')
        require(np.allclose(q[weld,arm*6:arm*6+6],p['q_start_rad'],atol=1e-10,rtol=0),'Raw weld joint differs')
        require_action_angles(q,labels['tasks'],labels['phases'],task,arm,l['angle'])
        local=np.asarray(local_states(p,pool['robot_ids']))[:,np.r_[np.arange(arm*6,arm*6+6),13+arm]]
        sections=action_sections(p,local); errors={}
        for phase,line in zip(('ingress','close','open','retreat'),sections):errors[phase]=on_polyline(own[ids&(labels['phases'][:,arm]==phase)],line)
        checks.append(dict(task=task,polyline_errors=errors))
    dt=np.diff(t);delta=abs(np.diff(q,axis=0));tcp=np.asarray(d['tcp'])
    require(tcp.shape==(len(t),2,3) and np.isfinite(tcp).all(),'Invalid TCP array')
    speed=dict(joint_deg_s=float(np.rad2deg(delta[:,:12]/dt[:,None]).max()),positioner_deg_s=float(np.rad2deg(delta[:,12]/dt).max()),gun_mm_s=float((delta[:,13:]/dt[:,None]).max()),tcp_mm_s=float((np.linalg.norm(np.diff(tcp,axis=0),axis=2)/dt[:,None]).max()))
    timing_valid=all(speed[k]<=v+1e-7 for k,v in dict(joint_deg_s=30,positioner_deg_s=15,gun_mm_s=80,tcp_mm_s=200).items())
    require(timing_valid or not r['accepted'],'Timing-invalid trajectory adopted')
    for task,l in expected.items():
        arm=l['arm'];mask=(labels['tasks'][:-1,arm]==task)&(labels['phases'][:-1,arm]=='weld'); axes=np.r_[np.arange(arm*6,arm*6+6),12,13+arm]
        require(abs(dt[mask].sum()-.6)<=1e-7 and np.max(delta[mask][:,axes])<=1e-10,'Invalid0.6s stationary dwell')
    stage_ranges=[]; previous=None
    for s in r['stages']:
        require(sha(s['receipt_file'])==s['receipt_sha256'],'Stage changed'); a=np.load(s['receipt_file'],allow_pickle=False)
        v=s['validation'];require(v['passed'] and v['fine'] and v['samples']==len(a['states']),'Missing whole-stage source fine receipt')
        for arm,order in enumerate(s['task_ids']):
            mask=np.asarray(a['phases'])[:,arm]=='transfer'
            starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
            ends=np.flatnonzero(mask&~np.r_[mask[1:],False])+1
            ids=['start']+[selected[int(task)]['action']['id'] for task in order]+['end']
            require(len(starts)==len(ids)-1,'Transfer phase/run count differs from selected edges')
            axes=np.r_[np.arange(arm*6,arm*6+6),13+arm]
            for lo0,hi0,leftid,rightid in zip(starts,ends,ids,ids[1:]):
                key=[s['stage'],arm,leftid,rightid]
                edge=[e for e in r['edges'] if e['key']==key]
                require(len(edge)==1 and edge[0]['passed'],'Selected transfer lacks source receipt')
                edgepath=np.load(edge[0]['file'],allow_pickle=False)['states']
                on_polyline(a['states'][lo0:hi0,axes],edgepath)
        require(s['bounds']==r['stage_bounds'][s['stage']],'Stage bounds differ')
        blo,bhi=s['bounds']
        require(np.array_equal(a['states'][[0,-1]],base['states'][[blo,bhi]]),'Original full15 stage anchors changed')
        require(s['original_full15_start']==base['states'][blo].tolist() and s['original_full15_end']==base['states'][bhi].tolist(),'Reported full15 anchors differ')
        require(np.allclose(a['states'][:,12],np.deg2rad(s['angle']),atol=1e-12,rtol=0),'Stage angle differs')
        piece=dict(a);piece['stage']=np.full(len(a['states']),s['stage'],int);stage_ranges.append(piece)
    # Rebuild every literal source bridge, prefix/suffix, and replacement, without importing planner assembly.
    bounds=r['stage_bounds'];require(len(bounds)==6,'Six source bounds required')
    def literal(lo,hi):
        part={k:np.asarray(v[lo:hi+1]).copy() for k,v in base.items() if k not in ('time','times')}
        part['time']=base['time'][lo:hi+1]-base['time'][lo]
        return part
    parts=[literal(0,bounds[0][0])];cursor=bounds[0][0]
    for index,((lo,hi),piece) in enumerate(zip(bounds,stage_ranges)):
        require(0<=lo<hi<len(base['states']) and (index==0 or lo>bounds[index-1][1]),'Invalid source bounds')
        if cursor!=lo:
            bridge=literal(cursor,lo);require(not np.any(bridge['phases']=='weld'),'Weld outside source stage');parts.append(bridge)
        parts.append(piece);cursor=hi
    if cursor!=len(base['states'])-1:parts.append(literal(cursor,len(base['states'])-1))
    arrays={k:[] for k in base.files if k not in ('time','times')};clock=[];offset=0.;last=None
    for index,piece in enumerate(parts):
        require(piece['time'][0]==0 and np.all(np.diff(piece['time'])>0),'Invalid source piece clock')
        if last is not None:require(np.array_equal(last,piece['states'][0]),'Literal full15 seam changed')
        first=0 if index==0 else 1
        for key in arrays:arrays[key].append(piece[key][first:])
        clock.append(offset+piece['time'][first:]);offset+=float(piece['time'][-1]);last=piece['states'][-1]
    for key,rows in arrays.items():
        assembled=np.concatenate(rows)
        if key=='tcp':require(np.allclose(assembled,d[key],atol=1e-8,rtol=0),'Recomputed TCP differs from literal geometry')
        else:require(np.array_equal(assembled,d[key]),'Literal assembly differs: '+key)
    require(np.allclose(np.concatenate(clock),t,atol=1e-10,rtol=0),'Literal global source clocks differ')
    moving=abs(np.diff(q[:,12]))>np.deg2rad(1e-7);episodes=int(np.sum(moving&~np.r_[False,moving[:-1]]))
    bm=abs(np.diff(base['states'][:,12]))>np.deg2rad(1e-7)
    require(episodes==int(np.sum(bm&~np.r_[False,bm[:-1]])),'Rotation episodes differ from baseline')
    require(r['canonical_park_barriers'] is False,'Mandatory canonical parks introduced')
    require(bool(r['accepted'])==bool(r['adopted']),'Adoption flags disagree')
    if r['accepted']:require(t[-1]-t[0]<=459.4458715768552+1e-7,'Adopted duration regressed')
    else:require(not (folder/'route.npz').exists(),'Rejected parent published route')
    result.update(passed=True,complete=True,accepted=bool(r['accepted'] and canonical_valid),parent_claimed_accepted=bool(r['accepted']),invalid_parent_adoption_claim=bool(r['accepted'] and not canonical_valid),route_sha256=sha(routefile),speeds=speed,action_checks=checks,rotation_episodes=episodes,canonical_boundary_labels_valid=canonical_valid,source_stage_global_mapping=canonical_ranges,strict_global_timing_valid=timing_valid,literal_interstage_geometry_and_clocks_verified=True,scope='Artifact identity, all680 graph participation, fine stage/edge receipts, all six original full15 anchors, exact literal assembly, action polylines, raw welds/dwells and measured speeds; inherited bridges rely on sealed baseline source provenance; no cold collision rerun')
    return result

def audit_normalization(args,r):
    from types import SimpleNamespace
    folder=Path(args.result).resolve();proof=r['boundary_label_normalization'];parentfile=Path(proof['parent_report']);parentroute=Path(proof['parent_route'])
    require(sha(parentfile)==proof['parent_report_sha256'] and sha(parentroute)==proof['parent_route_sha256'],'Normalization parent changed')
    parent=read(parentfile);require(parentroute.resolve() in [(parentfile.parent/'trial_route.npz').resolve(),(parentfile.parent/'route.npz').resolve()] and (parentroute.name=='trial_route.npz' or parent['accepted']),'Foreign normalization parent route')
    if parent['accepted']:require(sha(parentfile.parent/'route.npz')==sha(parentfile.parent/'trial_route.npz'),'Accepted parent trial/route disagree')
    pargs=SimpleNamespace(**vars(args));pargs.result=str(parentfile.parent);result=audit(pargs)
    require(result['complete'],'Incomplete normalization parent')
    for path,h in r['manifest']['bindings'].items():require(sha(path)==h,'Normalization binding changed: '+path)
    for key in ('stages','edges','selected_actions','participation','all21_graph_nodes','stage_bounds','incumbent_sha256','pool_sha256'):
        require(r[key]==parent[key],'Normalization changed parent metadata: '+key)
    require(proof['kinematics_and_clocks_unchanged'] is True,'Missing normalization byte contract')
    ranges=result['source_stage_global_mapping'];require(proof['source_stage_receipts']==ranges,'Normalization source-stage ranges differ')
    old=dict(np.load(parentroute,allow_pickle=False));route=folder/('route.npz' if r['accepted'] else 'trial_route.npz');new=dict(np.load(route,allow_pickle=False))
    require(set(old)==set(new),'Normalization changed array set')
    for key in old:require(old[key].dtype==new[key].dtype and old[key].shape==new[key].shape,'Normalization changed array type/shape')
    expected={k:v.copy() for k,v in old.items()};repairs=[]
    for m in ranges:
        piece=dict(np.load(m['parent_receipt_file'],allow_pickle=False));lo=m['global_start'];hi=m['global_end_inclusive']
        for k in ('tasks','phases'):require(np.array_equal(old[k][lo+1:hi+1],piece[k][1:]),'Nonboundary normalization source labels differ')
        changed=any(not np.array_equal(old[k][lo],piece[k][0]) for k in ('phases','tasks'))
        if changed:
            repairs.append(dict(stage=m['stage'],index=lo,before_phases=old['phases'][lo].tolist(),before_tasks=old['tasks'][lo].tolist(),after_phases=piece['phases'][0].tolist(),after_tasks=piece['tasks'][0].tolist(),source_receipt_sha256=m['parent_receipt_sha256']))
        for k in ('phases','tasks'):expected[k][lo]=piece[k][0]
    require(proof['repairs']==repairs,'Normalization repair proof differs')
    for key in expected:require(expected[key].tobytes()==new[key].tobytes(),'Normalization changed unauthorized bytes: '+key)
    require(bool(r['accepted'])==bool(r['adopted']),'Normalization adoption flags disagree')
    if r['accepted']:require(result['strict_global_timing_valid'] and new['time'][-1]-new['time'][0]<=459.4458715768552+1e-7,'Invalid normalized route adopted')
    else:require(not (folder/'route.npz').exists(),'Rejected normalization published route')
    result.update(report_sha256=sha(folder/'report.json'),route_sha256=sha(route),accepted=r['accepted'],canonical_boundary_labels_valid=True,boundary_label_normalization_verified=True,boundary_label_repairs=repairs,kinematics_and_clocks_byte_equal=True)
    return result

def audit(args):
    from types import SimpleNamespace
    folder=Path(args.result).resolve(); reportfile=folder/'report.json'
    if not reportfile.exists():return audit_geometry(args)
    r=read(reportfile)
    if 'boundary_label_normalization' in r and 'time_only_retiming' not in r:return audit_normalization(args,r)
    if 'time_only_retiming' not in r:return audit_geometry(args)
    receipt=r['time_only_retiming'];parentfile=Path(receipt['parent_report'])
    require(sha(parentfile)==receipt['parent_report_sha256'],'Retiming parent report changed')
    parent=read(parentfile);parentroute=Path(receipt['parent_trial'])
    require(sha(parentroute)==receipt['parent_trial_sha256'],'Retiming parent route changed')
    expected=parentfile.parent/('route.npz' if parent['accepted'] else 'trial_route.npz')
    require(parentroute.resolve() in [expected.resolve(),(parentfile.parent/'trial_route.npz').resolve()],'Foreign retiming parent trial')
    if parent['accepted']:require(sha(expected)==sha(parentfile.parent/'trial_route.npz'),'Accepted timing parent trial/route disagree')
    pargs=SimpleNamespace(**vars(args));pargs.result=str(parentfile.parent)
    result=audit(pargs);require(result['complete'],'Incomplete retiming parent')
    require(result['canonical_boundary_labels_valid'],'Retiming parent boundary labels invalid')
    for path,h in r['manifest']['bindings'].items():require(sha(path)==h,'Retiming binding changed: '+path)
    for key in ('stages','edges','selected_actions','participation','all21_graph_nodes','stage_bounds','incumbent_sha256','pool_sha256'):
        require(r[key]==parent[key],'Retiming changed parent metadata: '+key)
    require(r.get('source_stage_clock_parent') is True,'Missing source-stage parent clock declaration')
    old=dict(np.load(parentroute,allow_pickle=False));route=folder/('route.npz' if r['accepted'] else 'trial_route.npz');new=dict(np.load(route,allow_pickle=False))
    require(set(old)==set(new),'Retiming changed array set')
    for key in old:
        if key not in ('time','times'):require(old[key].dtype==new[key].dtype and old[key].shape==new[key].shape and old[key].tobytes()==new[key].tobytes(),'Retiming changed nonclock bytes: '+key)
    t=new['time'];dt=np.diff(t);prior=np.diff(old['time'])
    require(np.array_equal(t,new['times']) and np.isfinite(t).all() and np.all(dt>0),'Invalid retimed clocks')
    require(t[0]==old['time'][0] and np.all(dt>=prior-1e-12),'Retiming shortened source intervals')
    protected=np.any(old['phases'][:-1]=='weld',axis=1)
    require(np.allclose(dt[protected],prior[protected],atol=1e-12,rtol=0),'Protected START weld intervals changed')
    mapping=receipt['interval_clock_map']
    require(np.array_equal(mapping['parent_dt_s'],prior) and np.array_equal(mapping['retimed_dt_s'],dt),'Clock map differs')
    dq=abs(np.diff(old['states'],axis=0))
    required=np.maximum.reduce([prior,np.rad2deg(dq[:,:12]).max(1)/30,np.rad2deg(dq[:,12])/15,dq[:,13:].max(1)/80,np.linalg.norm(np.diff(old['tcp'],axis=0),axis=2).max(1)/200])
    require(not np.any(required[protected]>prior[protected]*(1+1e-12)+1e-14),'Protected weld interval needs actual stretch')
    required[protected]=prior[protected]
    require(mapping['changed_intervals']==np.flatnonzero(required>prior).tolist(),'Changed interval map differs')
    require(mapping['protected_start_weld_intervals']==np.flatnonzero(protected).tolist(),'Protected interval map differs')
    require(len(r['source_stage_global_mapping'])==6,'Missing stage global mapping')
    for stage,m in zip(r['stages'],r['source_stage_global_mapping']):
        require(m['stage']==stage['stage'] and m['parent_receipt_file']==stage['receipt_file'] and m['parent_receipt_sha256']==stage['receipt_sha256'],'Stage mapping identity differs')
        piece=dict(np.load(stage['receipt_file'],allow_pickle=False));lo=m['global_start'];hi=m['global_end_inclusive']
        require(np.array_equal(piece['states'],new['states'][lo:hi+1]),'Stage global mapping geometry differs')
        require(np.allclose(old['time'][lo:hi+1]-old['time'][lo],piece['time'],atol=1e-10,rtol=0),'Parent stage source clock differs')
    q=new['states'];delta=abs(np.diff(q,axis=0));tcp=new['tcp']
    speed=dict(joint_deg_s=float(np.rad2deg(delta[:,:12]/dt[:,None]).max()),positioner_deg_s=float(np.rad2deg(delta[:,12]/dt).max()),gun_mm_s=float((delta[:,13:]/dt[:,None]).max()),tcp_mm_s=float((np.linalg.norm(np.diff(tcp,axis=0),axis=2)/dt[:,None]).max()))
    require(all(speed[k]<=v+1e-7 for k,v in dict(joint_deg_s=30,positioner_deg_s=15,gun_mm_s=80,tcp_mm_s=200).items()),'Retimed actual speed violates limit')
    require(bool(r['accepted'])==bool(r['adopted']),'Retimed adoption flags disagree')
    if r['accepted']:require(t[-1]-t[0]<=459.4458715768552+1e-7,'Retimed adopted duration regressed')
    else:require(not (folder/'route.npz').exists(),'Rejected retiming published route')
    result.update(report_sha256=sha(reportfile),route_sha256=sha(route),accepted=r['accepted'],speeds=speed,strict_global_timing_valid=True,time_only_parent_verified=True,nonclock_arrays_byte_equal=True,protected_start_weld_intervals_unchanged=True,parent_report_sha256=sha(parentfile),parent_trial_sha256=sha(parentroute))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pool',required=True);p.add_argument('--result',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-incomplete',action='store_true');p.add_argument('--cold-rotations',action='store_true');a=p.parse_args()
    try: result=audit(a)
    except Exception as e:
        result=dict(passed=False,error=str(e),independent_cold_full_collision=False,auditor_sha256=sha(__file__));Path(a.out).write_text(json.dumps(result,indent=2));raise
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()

