"""Assemble sealed local actions and accepted entry/transfer/exit paths.

No new collision certification is performed here. Timing is display timing
from speed limits; source acceptance remains sampled, without independent cold.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = HERE.parent/'welding_retreat_transfer_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_section(data, section):
    section = [np.asarray(state,dtype=float) for state in section]
    if not section or any(state.shape!=(15,) or not np.isfinite(state).all() for state in section):
        raise ValueError('Finite nonempty full15 state section required')
    if data:
        if not np.allclose(data[-1],section[0],atol=1e-12,rtol=0):
            raise ValueError('Full15 state seam mismatch')
        section = section[1:]
    data.extend(section)


def rotation_metrics(states):
    angles = np.rad2deg(np.asarray(states)[:,12])
    changes = np.abs(np.diff(angles))
    moving = changes>1e-7
    starts = moving & ~np.r_[False,moving[:-1]]
    return dict(rotation_episodes=int(starts.sum()),
                total_positioner_travel_deg=float(changes.sum()))


def route_times(states, tcp, phases):
    states,tcp = np.asarray(states),np.asarray(tcp)
    if len(states)!=len(phases) or tcp.shape!=(len(states),2,3):
        raise ValueError('Timing array shapes disagree')
    changes = abs(np.diff(states,axis=0))
    dt = np.maximum(np.max(changes[:,:12],axis=1)/np.deg2rad(30.),
                    np.max(changes[:,13:],axis=1)/80.)
    dt = np.maximum(dt,changes[:,12]/np.deg2rad(15.))
    dt = np.maximum(dt,np.max(np.linalg.norm(np.diff(tcp,axis=0),axis=2),axis=1)/200.)
    dt = np.maximum(dt,1e-5)
    # A stationary welding frame represents dwell. Never shorten a moving
    # edge to force 0.6s if a malformed source section contains motion.
    for index in range(1,len(states)):
        if phases[index]=='weld':
            dt[index-1] = max(dt[index-1],.6)
    return np.r_[0.,np.cumsum(dt)]


def verify_manifest(report):
    for section in ('source_sha256','code_sha256'):
        for name,expected in report.get('manifest',{}).get(section,{}).items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Bound source/code changed: '+name)


def load_report(folder):
    folder = Path(folder).resolve()
    path = folder/'report.json' if folder.is_dir() else folder
    report = json.loads(path.read_text())
    verify_manifest(report)
    for name,expected in report.get('output_sha256',{}).items():
        if digest(path.parent/name)!=expected:
            raise ValueError('Bound output changed: '+name)
    return path.parent,report,path


def check_transfer_binding(report, actions_sha256, baseline_sha256):
    if report.get('actions_report_sha256')!=actions_sha256 or report.get('baseline_sha256')!=baseline_sha256:
        raise ValueError('Transfer receipt does not bind supplied actions and baseline')


def check_transfer_payload(payload):
    if payload.get('passed') is not True or payload.get('full_source_accepted') is not True:
        raise ValueError('Transfer path lacks passed source acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions',type=Path,required=True)
    parser.add_argument('--transfers',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh replay output directory required')
    sys.path[:0] = [str(ROOT),str(LEGACY),str(ROOT/'experimental/welding_positioner_v1')]
    from integrated_retreat_v117 import online_retreat_ids
    from replay_full21_core_v105 import local_states
    from export_latest_route_v160 import action_sections
    from welding_kinematics import RobotKinematics
    ad,actions,action_report_path = load_report(args.actions)
    td,transfers,transfer_report_path = load_report(args.transfers)
    baseline = args.baseline.resolve()
    if baseline.is_dir():
        baseline = baseline/'route.npz'
    baseline_sha256 = digest(baseline)
    check_transfer_binding(transfers,digest(action_report_path),baseline_sha256)
    if transfers.get('full_source_accepted') is not True:
        raise ValueError('All source-accepted transfers and entry/exit required')
    accepted = online_retreat_ids(actions,ad)
    order = transfers['order_weld_ids']
    if len(order)!=21 or len(set(order))!=21 or accepted!=set(order):
        raise ValueError('Exactly 21 accepted weld actions must appear once')
    rids = transfers['robot_ids']
    if len(rids)!=2 or len(set(rids))!=2:
        raise ValueError('Two distinct source robot IDs required')
    rows = {row['weld_id']:row for row in actions['rows']}
    inputs = {str(action_report_path):digest(action_report_path),
              str(transfer_report_path):digest(transfer_report_path)}
    for report in (actions,transfers):
        for section in ('source_sha256','code_sha256'):
            for name,expected in report.get('manifest',{}).get(section,{}).items():
                inputs[str(ROOT/name)] = expected
    for dependency in (Path(__file__),LEGACY/'replay_full21_core_v105.py',
                       LEGACY/'export_latest_route_v160.py',LEGACY/'integrated_retreat_v117.py',
                       ROOT/'welding_kinematics.py'):
        inputs[str(dependency)] = digest(dependency)
    def read_transfer(reference):
        name = reference['path_file']
        path = td/name
        expected = reference.get('path_sha256',transfers.get('output_sha256',{}).get(name))
        if not expected or digest(path)!=expected:
            raise ValueError('Transfer path is unsealed or changed: '+name)
        inputs[str(path)] = expected
        payload = json.loads(path.read_text())
        check_transfer_payload(payload)
        return [frame['state'] for frame in payload['frames']]
    states,phases,tasks,sections = [],[],[],[]
    def add(section,phase,task):
        start = len(states)
        append_section(states,section)
        count = len(states)-start
        phases.extend([phase]*count)
        tasks.extend([task]*count)
        sections.append(dict(start=start,end=len(states),phase=phase,task=task))
    add(read_transfer(transfers['entry']),'entry',0)
    welding_angles = []
    for index,wid in enumerate(order):
        path_file = ad/rows[wid]['best']['path_file']
        inputs[str(path_file)] = rows[wid]['best']['path_sha256']
        path = json.loads(path_file.read_text())
        if path['length_mm']!=1000.:
            raise ValueError('Literal 1000 mm action required')
        ingress,close,opening,retreat = action_sections(path,local_states(path,rids))
        add(ingress,'ingress',index+1)
        add(close,'close',index+1)
        add([states[-1],states[-1].copy()],'weld',index+1)
        welding_angles.append(float(path['positioner_deg']))
        add(opening,'open',index+1)
        add(retreat,'retreat',index+1)
        if index+1<len(order):
            matches = [edge for edge in transfers['selected_edges']
                       if edge['from_weld_id']==wid and edge['to_weld_id']==order[index+1]]
            if len(matches)!=1:
                raise ValueError('Exactly one selected transfer required for adjacent welds')
            add(read_transfer(matches[0]),'transfer',index+1)
    add(read_transfer(transfers['exit']),'exit',22)
    array = np.asarray(states)
    inputs[str(baseline)] = baseline_sha256
    with np.load(baseline,allow_pickle=False) as original:
        baseline_start = original['states'][0]
        baseline_end = original['states'][-1]
    if not np.allclose(array[0],baseline_start,atol=1e-12,rtol=0) or not np.allclose(array[-1],baseline_end,atol=1e-12,rtol=0):
        raise ValueError('Route does not share baseline full15 start and end states')
    scene_path = ROOT/'build/scene_placement_corrected_v1.json'
    inputs[str(scene_path)] = digest(scene_path)
    scene = json.loads(scene_path.read_text())
    models = [RobotKinematics(scene,next(asset for asset in scene['assets'] if asset['id']==rid)) for rid in rids]
    tcp = np.empty((len(array),2,3))
    for i,model in enumerate(models):
        cache = {}
        for j,state in enumerate(array):
            q = state[6*i:6*i+6]
            key = tuple(q)
            if key not in cache:
                if not model.limits_valid(q):
                    raise ValueError('Source robot limit violation')
                cache[key] = model.tcp(q)[:3,3]
            tcp[j,i] = cache[key]
    times = route_times(array,tcp,phases)
    for name,expected in inputs.items():
        if digest(name)!=expected:
            raise ValueError('Input changed during export: '+name)
    out.mkdir(parents=True)
    np.savez_compressed(out/'route.npz',states=array,times=times,tcp=tcp,
                        phases=np.asarray(phases),tasks=np.asarray(tasks))
    distinct = []
    for angle in welding_angles:
        if not any(abs(angle-old)<=1e-7 for old in distinct):
            distinct.append(angle)
    source_hashes = {}
    for name,expected in inputs.items():
        path = Path(name)
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            pass
        source_hashes[name] = expected
    audit = dict(array_sha256=digest(out/'route.npz'),input_sha256=inputs,
                 source_sha256=source_hashes,
                 robot_ids=rids,order=order,sections=sections,frame_count=len(array),
                 duration_s=float(times[-1]),required_retreat_mm=1000.,
                 full_route_online_accepted=True,full_source_accepted=True,
                 independent_cold_performed=False,timing_display_only=True,
                 same_baseline_start_end=True,all21_coverage_once=True,
                 distinct_welding_stops=len(distinct),welding_stop_angles_deg=distinct,
                 initial_state='Explicit source-accepted connection from baseline start',
                 final_state='Explicit source-accepted connection to baseline end',
                 ingress='Exact reversal of accepted local retreat and opening prefix',
                 transfer_report=str(transfer_report_path),**rotation_metrics(array))
    # Legacy viewer resolves an absolute transfer_report path correctly.
    audit['scope'] = 'Assembled existing sampled source-online receipts. No new collision check or independent cold certification. Timing and welding dwell are display estimates.'
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({key:audit[key] for key in ('duration_s','rotation_episodes','distinct_welding_stops','same_baseline_start_end')}),flush=True)


if __name__=='__main__':
    main()
