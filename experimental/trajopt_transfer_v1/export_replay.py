"""Export a hash-bound TrajOpt route for the existing display-only viewer."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT/'experimental/welding_retreat_transfer_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(sources):
    for path, expected in sources.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Changed bound input '+path)


def verify_route(route_file):
    route_file = Path(route_file).resolve()
    route = json.loads(route_file.read_text())
    source = Path(route['source_report'])
    optimization = route_file.parent/'optimization_report.json'
    sources = {str(source):route['source_report_sha256'],
               str(optimization):route['optimization_report_sha256'], str(route_file):sha(route_file)}
    _check(sources)
    baseline = json.loads(source.read_text())
    for key in ('order_weld_ids', 'order', 'robot_ids', 'nodes', 'final_retreat_out'):
        if route[key] != baseline[key]:
            raise ValueError('Route changed complete source '+key)
    order = route['order_weld_ids']
    if not route.get('complete') or len(route['selected_edges']) != len(order)-1:
        raise ValueError('Complete route required')
    for index, edge in enumerate(route['selected_edges']):
        if (edge['from_weld_id'], edge['to_weld_id']) != (order[index], order[index+1]):
            raise ValueError('Transfer order mismatch')
        path = Path(edge['path_file'])
        if not path.is_absolute():
            raise ValueError('Absolute optimized transfer path required')
        sources[str(path)] = edge['path_sha256']
    _check(sources)
    return route, baseline, sources


def export(route_file, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Fresh output required')
    route, baseline, sources = verify_route(route_file)
    sys.path[:0] = [str(ROOT), str(LEGACY), str(ROOT/'experimental/welding_positioner_v1')]
    from welding_kinematics import RobotKinematics
    from export_latest_route_v160 import action_sections
    from replay_full21_core_v105 import local_states, append_states
    from integrated_retreat_v117 import online_retreat_ids
    if not baseline['summary']['all_required_points_route_planned']:
        raise ValueError('Complete sealed source route required')
    rd = ROOT/route['final_retreat_out']
    report = json.loads((rd/'report.json').read_text())
    order = route['order_weld_ids']
    rids = route['robot_ids']
    if online_retreat_ids(report, rd) != set(order):
        raise ValueError('All source action receipts required')
    for directory, document in [(rd, report), (Path(route['source_report']).parent, baseline)]:
        sources.update(document['manifest']['source_sha256'])
        sources.update(document['manifest']['code_sha256'])
        sources[str(directory/'report.json')] = sha(directory/'report.json')
        for name, digest in document['output_sha256'].items():
            sources[str(directory/name)] = digest
    dependencies = [Path(__file__), LEGACY/'export_latest_route_v160.py', LEGACY/'replay_full21_core_v105.py',
        LEGACY/'replay_latest_factory_v161.py', LEGACY/'replay_full21_v112.py',
        ROOT/'experimental/mujoco_replay_v1/scene.py', ROOT/'experimental/mujoco_replay_v1/materials.py',
        ROOT/'experimental/mujoco_replay_v1/timeline.py', ROOT/'experimental/welding_positioner_v1/positioner.py',
        ROOT/'build/welding_full_factory_v109/report.json', ROOT/'build/welding_full_factory_v109/factory.glb',
        ROOT/'build/positionerL_detail.json', ROOT/'build/welding_positioner_static_v2/DONG.glb',
        ROOT/'build/welding_positioner_static_v2/JING.glb']
    dependencies.extend(Path(__file__).parent/name for name in ('solver.py','run_optimize.py','proximity.py','replay_optimized.py'))
    for path in dependencies:
        if path.exists():
            sources[str(path)] = sha(path)
    _check(sources)
    rows = {row['weld_id']:row for row in report['rows']}
    states, phases, tasks, sections = [], [], [], []
    def add(section, phase, task):
        start = len(states)
        append_states(states, section)
        count = len(states)-start
        phases.extend([phase]*count)
        tasks.extend([task]*count)
        sections.append(dict(start=start,end=len(states),phase=phase,task=task))
    for index, wid in enumerate(order):
        path_file = rd/rows[wid]['best']['path_file']
        path = json.loads(path_file.read_text())
        sources[str(path_file)] = sha(path_file)
        if path['length_mm'] != 1000.:
            raise ValueError('1000mm source retreat required')
        ingress, close, opening, retreat = action_sections(path, local_states(path, rids))
        add(ingress,'ingress',index+1)
        add(close,'close',index+1)
        add([states[-1], states[-1].copy()],'weld',index+1)
        add(opening,'open',index+1)
        add(retreat,'retreat',index+1)
        if index+1 < len(order):
            transfer = json.loads(Path(route['selected_edges'][index]['path_file']).read_text())
            add([np.asarray(frame['state']) for frame in transfer['frames']],'transfer',index+1)
    scene_file = ROOT/'build/scene_placement_corrected_v1.json'
    sources[str(scene_file)] = sha(scene_file)
    scene = json.loads(scene_file.read_text())
    array = np.asarray(states)
    tcp = np.empty((len(array),len(rids),3))
    started = time.perf_counter()
    for j, rid in enumerate(rids):
        model = RobotKinematics(scene,next(asset for asset in scene['assets'] if asset['id']==rid))
        cache = {}
        for i,state in enumerate(array):
            q = state[6*j:6*j+6]
            key = tuple(q)
            if key not in cache:
                if not model.limits_valid(q):
                    raise ValueError('Invalid replay joint state')
                cache[key] = model.tcp(q)[:3,3]
            tcp[i,j] = cache[key]
        print('TCP_EXPORTED',rid,len(cache),flush=True)
    dt = np.maximum(np.max(abs(np.diff(array[:,:12],axis=0)),axis=1)/np.deg2rad(30.),
                    np.max(abs(np.diff(array[:,13:],axis=0)),axis=1)/80.)
    dt = np.maximum(dt,abs(np.diff(array[:,12]))/np.deg2rad(15.))
    dt = np.maximum(dt,np.max(np.linalg.norm(np.diff(tcp,axis=0),axis=2),axis=1)/200.)
    dt = np.maximum(dt,1e-5)
    for i in range(1,len(array)):
        if phases[i]=='weld':
            dt[i-1] = .6
    times = np.r_[0., np.cumsum(dt)]
    _check(sources)
    out.mkdir(parents=True)
    np.savez_compressed(out/'route.npz',states=array,times=times,tcp=tcp,phases=np.asarray(phases),tasks=np.asarray(tasks))
    adapter = dict(route, summary=baseline['summary'],
                   trajopt_postprocessing=True, acceptance=route.get('acceptance'),
                   continuous_collision_certified=False)
    adapter_file = out/'adapter_transfer.json'
    adapter_file.write_text(json.dumps(adapter,ensure_ascii=False,indent=2))
    sources[str(adapter_file)] = sha(adapter_file)
    _check(sources)
    audit = dict(source_sha256=sources,array_sha256=sha(out/'route.npz'),transfer_report=str(adapter_file),
        robot_ids=rids,order=order,sections=sections,frame_count=len(array),duration_s=float(times[-1]),
        required_retreat_mm=1000.,full_route_online_accepted=True,trajopt_postprocessing=True,
        acceptance='TrajOpt replacements fine checked; retained transfers and local actions inherit sealed baseline acceptance',
        continuous_collision_certified=False,independent_cold_performed=False,timing_display_only=True,
        initial_state='First sealed retreat endpoint; home-to-start not planned',
        ingress='Exact reversal of sealed local action; welding dwell is display only',
        export_tcp_seconds=time.perf_counter()-started)
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('EXPORTED_TRAJOPT_REPLAY',len(array),flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimization-route',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    export(args.optimization_route,args.out)


if __name__=='__main__':
    main()
