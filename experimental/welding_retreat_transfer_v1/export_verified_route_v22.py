"""Export a cold-verified zero-positioner subset as exact source FK replay."""
import argparse,copy,gzip,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from welding_kinematics import RobotKinematics
from gun_kinematics import GunKinematics


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')));temporary.replace(path)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--retreat-out',type=Path,required=True)
    parser.add_argument('--transfer-out',type=Path,required=True)
    parser.add_argument('--cold',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Use fresh route output')
    retreat_dir=args.retreat_out.resolve();transfer_dir=args.transfer_out.resolve();cold_path=args.cold.resolve()
    retreat=json.loads((retreat_dir/'report.json').read_text());transfer=json.loads((transfer_dir/'report.json').read_text());cold=json.loads(cold_path.read_text())
    if (cold['retreat_report_sha256']!=digest(retreat_dir/'report.json') or cold['transfer_report_sha256']!=digest(transfer_dir/'report.json') or
        not cold['summary']['all_selected_passed'] or not cold['summary']['retained_subset_route_verified'] or
        cold['summary']['selected_retreats']!=len(transfer['order']) or cold['summary']['selected_transfers']!=max(0,len(transfer['order'])-1)):
        raise RuntimeError('Matching complete subset cold evidence required')
    if retreat['summary']['required_straight_distance_mm']!=2000.:raise RuntimeError('Literal2m required')
    sources={}
    for directory,report in ((retreat_dir,retreat),(transfer_dir,transfer)):
        sources.update(report['manifest']['source_sha256']);sources.update(report['manifest']['code_sha256'])
        sources[str((directory/'report.json').relative_to(ROOT))]=digest(directory/'report.json')
        for name,sha in report['output_sha256'].items():
            p=directory/name
            if digest(p)!=sha:raise RuntimeError('Output seal mismatch')
            sources[str(p.relative_to(ROOT))]=sha
    sources[str(cold_path.relative_to(ROOT))]=digest(cold_path)
    source_scene=ROOT/'build/scene_placement_corrected_v1.json';scene=json.loads(source_scene.read_text());sources[str(source_scene.relative_to(ROOT))]=digest(source_scene)
    for p in [ROOT/'build/robot.json',ROOT/'welding_kinematics.py',ROOT/'gun_kinematics.py',ROOT/'scripts/audit_contact_release_v17.py',ROOT/'scripts/audit_counterbalance_fk_v17.py',Path(__file__),HERE/'verified_route_loader_v22.py',HERE/'replay_verified_route_v22.py']:
        sources[str(p.relative_to(ROOT))]=digest(p)
    for name,sha in sources.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Input changed '+name)
    rows={r['weld_id']:r for r in retreat['rows']};rids=transfer['robot_ids'];order=transfer['order_weld_ids']
    paths={wid:json.loads((retreat_dir/rows[wid]['best']['path_file']).read_text()) for wid in order}
    if any(p['positioner_deg']!=0. for p in paths.values()):raise RuntimeError('This renderer requires stationary zero fixture')
    for p in paths.values():
        if not p['passed'] or p['length_mm']!=2000.:raise RuntimeError('Invalid retained path')
    states=[];seams=[]
    def retreat_state(path,frame):
        state=np.zeros(15)
        for i,rid in enumerate(rids):
            state[6*i:6*i+6]=frame['q_rad'] if rid==path['robot_id'] else path['other_q_rad']
            state[13+i]=frame['driver_mm'] if rid==path['robot_id'] else 0.
        return state
    def append_section(section, label, reverse=False):
        if not section:raise RuntimeError('Empty section')
        if states:
            error=float(np.max(abs(states[-1]['state']-section[0]['state'])))
            if error>1e-12:raise RuntimeError('15-axis seam mismatch '+label)
            seams.append(dict(label=label,max_axis_difference=error,first_index=len(states)))
            section=section[1:]
        states.extend(section)
    def local_section(path,wid,task,reverse=False):
        result=[]
        frames=path['frames'][::-1] if reverse else path['frames']
        for f in frames:
            phase=('close' if f['phase'] in ('opening','closed') else 'ingress') if reverse else ('open' if f['phase'] in ('opening','closed') else 'retreat')
            result.append(dict(state=retreat_state(path,f),phase=phase,weld_id=wid,task_index=task,source_phase=f['phase']))
        return result
    for index,wid in enumerate(order):
        path=paths[wid];task=index+1
        if index==0:
            state=retreat_state(path,path['frames'][0]);states.append(dict(state=state,phase='weld_display',weld_id=wid,task_index=task,dwell_s=.5))
        else:
            append_section(local_section(path,wid,task,True), 'transfer→ingress:'+wid)
            states.append(dict(state=states[-1]['state'].copy(),phase='weld_display',weld_id=wid,task_index=task,dwell_s=.5))
        append_section(local_section(path,wid,task), 'weld→retreat:'+wid)
        if index+1<len(order):
            next_wid=order[index+1]
            edge=next(e for e in transfer['selected_edges'] if e['from_weld_id']==wid and e['to_weld_id']==next_wid)
            edge_path=json.loads((transfer_dir/edge['path_file']).read_text())
            if not edge['passed'] or digest(transfer_dir/edge['path_file'])!=edge['path_sha256']:raise RuntimeError('Transfer identity')
            section=[dict(state=np.asarray(f['state']),phase='transfer',weld_id=wid+'→'+next_wid,task_index=task+1) for f in edge_path['frames']]
            append_section(section,'retreat→transfer:'+wid)
    models={};gun_models={};guns={}
    for rid in rids:
        asset=next(a for a in scene['assets'] if a['id']==rid);model=RobotKinematics(scene,asset)
        gun=next(a for a in scene['assets'] if a['kind']=='gun' and ('R1' if 'R1' in asset['name'] else 'R2') in a['name'])
        raw_path=ROOT/gun['gun']['source_detail'];sources[str(raw_path.relative_to(ROOT))]=digest(raw_path)
        models[rid]=model;guns[rid]=gun;gun_models[rid]=GunKinematics(json.loads(raw_path.read_text()))
        model.gun_home=np.linalg.inv(model.links(model.q_initial,world=True)['245'])@np.asarray(gun['transform']).reshape(4,4)
    fk=load_module('route_independent_fk',ROOT/'scripts/audit_contact_release_v17.py')
    cb=load_module('route_independent_cb',ROOT/'scripts/audit_counterbalance_fk_v17.py')
    raw=json.loads((ROOT/'build/robot.json').read_text());joints=fk.joints_from_raw(raw)[:6]
    tcp_homes={rid:fk.source_tcp_home(raw,models[rid].asset) for rid in rids}
    caches={rid:{} for rid in rids};frames=[];elapsed=0.;previous=None;max_fk=max_cb=0.;started=time.perf_counter()
    for record in states:
        state=record['state']
        if state[12]!=0.:raise RuntimeError('Nonzero fixture unsupported')
        robots={};render_guns={}
        for i,rid in enumerate(rids):
            q=state[6*i:6*i+6];driver=float(state[13+i]);model=models[rid];key=tuple(np.r_[q,driver])
            if not model.limits_valid(q):raise RuntimeError('Source coupled limits')
            if key not in caches[rid]:
                links=model.links(q,world=True);tcp=model.tcp(q)
                truth=model.base@fk.flange_source(q,joints)@tcp_homes[rid]
                max_fk=max(max_fk,float(np.max(abs(tcp-truth))))
                _,truth_cb,_,_=cb.counterbalance(q,scene)
                max_cb=max(max_cb,*(float(np.max(abs(links[k]-model.base@truth_cb[k]))) for k in ('246','247')))
                gun=guns[rid];deltas=gun_models[rid].transforms(driver)
                parts={pid:(links['245']@model.gun_home@deltas[p['link_key']]).ravel().tolist() for pid,p in gun['gun']['link_models'].items()}
                caches[rid][key]=(dict(q_rad=q.tolist(),links={k:v.ravel().tolist() for k,v in links.items()},tcp_matrix=tcp.ravel().tolist()),dict(driver_mm=driver,parts=parts))
            robots[rid],render_guns[guns[rid]['id']]=caches[rid][key]
        if previous is not None:
            dt=max(float(np.max(abs(state[:12]-previous['state'][:12])))/np.deg2rad(30.),float(np.max(abs(state[13:]-previous['state'][13:])))/80.,1e-5)
            for rid in rids:
                tcp=np.asarray(robots[rid]['tcp_matrix']).reshape(4,4)[:3,3];old_tcp=np.asarray(frames[-1]['robots'][rid]['tcp_matrix']).reshape(4,4)[:3,3]
                dt=max(dt,float(np.linalg.norm(tcp-old_tcp))/200.)
            dt=max(dt,record.get('dwell_s',0.),previous.get('dwell_s',0.))
            elapsed+=dt
        frame=dict(time_s=elapsed,phase=record['phase'],weld_id=record['weld_id'],task_index=record['task_index'],completed_welds=record['task_index']-1,robots=robots,guns=render_guns,positioner_deg=0.,source_state=state.tolist())
        frames.append(frame);previous=record
    if max_fk>1e-6 or max_cb>1e-6:raise RuntimeError('Independent FK mismatch')
    out.mkdir(parents=True)
    summary=dict(frame_count=len(frames),duration_s=elapsed,retained_weld_count=len(order),required_straight_distance_mm=2000.,conditional_cold_verified_subset=True,collision_sampled=True,system_safe=False,strict_planned_task_count=0,full21_route_verified=False,source_to_initial_state_verified=False,weld_dwell_display_only=True,timing_display_only=True,source_tcp_max_matrix_error=max_fk,counterbalance_max_matrix_error=max_cb)
    plan=dict(schema='cold_verified_zero_positioner_retreat_transfer_replay_v22',summary=summary,selected_weld_ids=order,frames=frames)
    data=json.dumps(plan,ensure_ascii=False,separators=(',',':')).encode();name='welding_plan_verified_200cm_subset_v22.json.gz'
    with (out/name).open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as packed:packed.write(data)
    for name_in,sha in sources.items():
        if digest(ROOT/name_in)!=sha:raise RuntimeError('Inputs changed during export')
    audit=dict(plans={name:dict(relative_path=str((out/name).relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),file_sha256=digest(out/name),conditional_cold_verified_subset=True)},source_sha256=sources,cold_path=str(cold_path.relative_to(ROOT)),export_summary=summary,seams=seams,full_system_continuous_collision_certified=False,wall_s=time.perf_counter()-started)
    atomic(out/'audit.json',audit);atomic(out/'summary.json',summary)
    print('VERIFIED_ROUTE_REPLAY_EXPORT',summary,flush=True)


if __name__=='__main__':main()
