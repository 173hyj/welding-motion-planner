"""Export the latest online-accepted full route, including adopted node states."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_positioner_v1')]
from welding_kinematics import RobotKinematics
from replay_full21_core_v105 import local_states,append_states
from integrated_retreat_v117 import online_retreat_ids


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def action_sections(path, states):
    opening_count=sum(f['travel_mm']<=1e-8 for f in path['frames'])
    if not 1<=opening_count<len(states):
        raise ValueError('Opening prefix and nonempty straight retreat required')
    opening=list(states[:opening_count])
    retreat=list(states[opening_count-1:])
    return list(reversed(retreat)),list(reversed(opening)),opening,retreat


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--transfers',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():raise FileExistsError('Fresh output required')
    td=args.transfers.resolve()
    if not td.is_relative_to(ROOT):raise ValueError('Transfer output must be inside project')
    t=json.loads((td/'report.json').read_text())
    if not t['summary']['all_required_points_route_planned']:
        raise ValueError('Complete latest route required')
    rd=ROOT/t['final_retreat_out']
    r=json.loads((rd/'report.json').read_text())
    accepted=online_retreat_ids(r,rd)
    if accepted != set(t['order_weld_ids']) or len(accepted)!=21:
        raise ValueError('All 21 action receipts required')
    sources={}
    for directory,report in [(rd,r),(td,t)]:
        sources.update(report['manifest']['source_sha256']);sources.update(report['manifest']['code_sha256'])
        sources[str((directory/'report.json').relative_to(ROOT))]=sha(directory/'report.json')
        for name,h in report['output_sha256'].items():sources[str((directory/name).relative_to(ROOT))]=h
    dependencies=[Path(__file__),HERE/'replay_full21_core_v105.py',HERE/'replay_latest_factory_v161.py',
        ROOT/'experimental/mujoco_replay_v1/scene.py',ROOT/'experimental/mujoco_replay_v1/materials.py',
        ROOT/'experimental/mujoco_replay_v1/timeline.py',ROOT/'experimental/welding_positioner_v1/positioner.py',
        HERE/'replay_full21_v112.py',ROOT/'build/welding_full_factory_v109/report.json',ROOT/'build/welding_full_factory_v109/factory.glb',ROOT/'build/positionerL_detail.json',ROOT/'build/welding_positioner_static_v2/DONG.glb',ROOT/'build/welding_positioner_static_v2/JING.glb']
    for p in dependencies:sources[str(p.relative_to(ROOT))]=sha(p)
    for name,h in sources.items():
        if sha(ROOT/name)!=h:raise ValueError('Changed bound input '+name)
    rows={row['weld_id']:row for row in r['rows']};order=t['order_weld_ids'];rids=t['robot_ids']
    states=[];phases=[];tasks=[];sections=[]
    def add(section,phase,task):
        start=len(states);append_states(states,section);count=len(states)-start
        phases.extend([phase]*count);tasks.extend([task]*count)
        sections.append(dict(start=start,end=len(states),phase=phase,task=task))
    for index,wid in enumerate(order):
        path=json.loads((rd/rows[wid]['best']['path_file']).read_text())
        if path['length_mm']!=300.:raise ValueError('300mm required')
        local=local_states(path,rids)
        ingress,close,opening,retreat=action_sections(path,local)
        add(ingress,'ingress',index+1)
        add(close,'close',index+1)
        add([states[-1],states[-1].copy()],'weld',index+1)
        add(opening,'open',index+1)
        add(retreat,'retreat',index+1)
        if index+1<len(order):
            edge=next(e for e in t['selected_edges'] if e['from_weld_id']==wid and e['to_weld_id']==order[index+1])
            transfer=json.loads((td/edge['path_file']).read_text())
            add([np.asarray(f['state']) for f in transfer['frames']],'transfer',index+1)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    models=[RobotKinematics(scene,next(a for a in scene['assets'] if a['id']==rid)) for rid in rids]
    array=np.asarray(states);tcp=np.empty((len(array),2,3));started=time.perf_counter()
    for j,model in enumerate(models):
        cache={}
        for i,state in enumerate(array):
            q=state[6*j:6*j+6];key=tuple(q)
            if key not in cache:
                if not model.limits_valid(q):raise ValueError('Invalid source state')
                cache[key]=model.tcp(q)[:3,3]
            tcp[i,j]=cache[key]
        print('TCP_EXPORTED',rids[j],len(cache),flush=True)
    dt=np.maximum(np.max(abs(np.diff(array[:,:12],axis=0)),axis=1)/np.deg2rad(30.),
                  np.max(abs(np.diff(array[:,13:],axis=0)),axis=1)/80.)
    dt=np.maximum(dt,abs(np.diff(array[:,12]))/np.deg2rad(15.))
    dt=np.maximum(dt,np.max(np.linalg.norm(np.diff(tcp,axis=0),axis=2),axis=1)/200.)
    dt=np.maximum(dt,1e-5)
    for i in range(1,len(array)):
        if phases[i]=='weld':dt[i-1]=.6
    times=np.r_[0.,np.cumsum(dt)]
    out.mkdir(parents=True);np.savez_compressed(out/'route.npz',states=array,times=times,tcp=tcp,
        phases=np.asarray(phases),tasks=np.asarray(tasks))
    for name,h in sources.items():
        if sha(ROOT/name)!=h:raise ValueError('Input changed during export '+name)
    audit=dict(source_sha256=sources,array_sha256=sha(out/'route.npz'),transfer_report=str((td/'report.json').relative_to(ROOT)),
        robot_ids=rids,order=order,sections=sections,frame_count=len(array),duration_s=float(times[-1]),
        required_retreat_mm=300.,full_route_online_accepted=True,independent_cold_performed=False,timing_display_only=True,
        initial_state='First online-accepted retreat endpoint; source home-to-start not planned',
        ingress='Exact reversal of the same online-accepted local opening/retreat; welding dwell is display only',
        export_tcp_seconds=time.perf_counter()-started)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('EXPORTED_FULL21',len(array),'duration',times[-1],flush=True)


if __name__=='__main__':main()
