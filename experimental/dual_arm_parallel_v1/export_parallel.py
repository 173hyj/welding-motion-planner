"""Seal an accepted simultaneous trajectory without retiming or resampling it."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).absolute().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(sources):
    if not sources:
        raise ValueError('Bound source inputs required')
    for name, digest in sources.items():
        path=Path(name)
        if not path.is_absolute():
            raise ValueError('Absolute bound source path required')
        if sha(path)!=digest:
            raise ValueError('Changed bound input '+name)


def validate_report(report):
    if not report.get('complete') or not report.get('display_only') or not report.get('validation',{}).get('passed'):
        raise ValueError('Complete source-validated display-only run required')
    if report.get('candidate_policy')!='existing_selected_tasks_only':
        raise ValueError('Original selected task policy required')
    order=report['order_weld_ids']
    if len(order)!=21 or len(set(order))!=21 or len(report['robot_ids'])!=2:
        raise ValueError('Complete original 21 weld actions required')
    arms=[[],[]]
    for stage in report['stages']:
        if len(stage['pieces'])!=2: raise ValueError('Two arm piece timelines required')
        for arm in (0,1):
            arms[arm].extend(int(piece['task']) for piece in stage['pieces'][arm] if piece['phase']=='weld')
    if sorted(arms[0]+arms[1])!=list(range(1,22)):
        raise ValueError('Exactly 21 weld actions, each once, required')
    if any(arm!=sorted(arm) for arm in arms):
        raise ValueError('Original per-arm order changed')
    assignment=report.get('original_assignment')
    if assignment is None or len(assignment)!=21 or any(assignment[task-1]!=arm for arm in (0,1) for task in arms[arm]):
        raise ValueError('Original robot assignment missing or changed')
    return arms


def original_tcp(states, robot_ids, sources):
    sys.path.insert(0,str(ROOT))
    from welding_kinematics import RobotKinematics
    scene_file=ROOT/'build/scene_placement_corrected_v1.json'
    sources[str(scene_file)]=sha(scene_file)
    scene=json.loads(scene_file.read_text())
    tcp=np.empty((len(states),2,3))
    for arm,rid in enumerate(robot_ids):
        model=RobotKinematics(scene,next(a for a in scene['assets'] if a['id']==rid))
        cache={}
        for index,state in enumerate(states):
            q=state[arm*6:arm*6+6];key=tuple(q)
            if key not in cache:
                if not model.limits_valid(q):raise ValueError('Invalid replay joint state')
                cache[key]=model.tcp(q)[:3,3]
            tcp[index,arm]=cache[key]
    return tcp


def export_parallel(run_dir, output_dir, *, tcp_provider=None):
    run,out=Path(run_dir).absolute(),Path(output_dir).absolute()
    if out.exists(): raise FileExistsError('Fresh replay output required')
    report_file=run/'report.json';trajectory_file=run/'trajectory.npz'
    report=json.loads(report_file.read_text())
    arms=validate_report(report)
    sources=dict(report['source_sha256']);check_sources(sources)
    if sha(trajectory_file)!=report['trajectory_sha256']:raise ValueError('Changed bound trajectory')
    sources[str(report_file)]=sha(report_file);sources[str(trajectory_file)]=report['trajectory_sha256']
    with np.load(trajectory_file,allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in ('states','time','phases','tasks','stage')}
    states,times=arrays['states'],arrays['time'];n=len(states)
    if states.shape!=(n,15) or n<2 or times.shape!=(n,) or not np.isfinite(states).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Finite 15-axis states and strictly ascending times required')
    if arrays['phases'].shape!=(n,2) or arrays['tasks'].shape!=(n,2) or arrays['stage'].shape!=(n,):
        raise ValueError('Per-arm phase/task and per-frame stage required')
    if not np.issubdtype(arrays['tasks'].dtype,np.integer) or np.any(arrays['tasks']<0) or np.any(arrays['tasks']>21):
        raise ValueError('Original task numbers 0..21 required')
    for arm in (0,1):
        observed=set(arrays['tasks'][arrays['phases'][:,arm]=='weld',arm].tolist())
        if observed!=set(arms[arm]):raise ValueError('Trajectory weld task coverage mismatch')
    if tcp_provider is None:
        tcp=original_tcp(states,report['robot_ids'],sources)
        for path in (ROOT/'welding_kinematics.py',ROOT/'experimental/trajopt_transfer_v1/replay_optimized.py',
                     ROOT/'build/welding_full_factory_v109/report.json',ROOT/'build/welding_full_factory_v109/factory.glb'):
            sources[str(path)]=sha(path)
    else: tcp=tcp_provider(states,report['robot_ids'])
    tcp=np.asarray(tcp,float)
    if tcp.shape!=(n,2,3) or not np.isfinite(tcp).all():raise ValueError('Finite original-source TCP array required')
    sources[str(Path(__file__).absolute())]=sha(__file__)
    viewer=Path(__file__).absolute().with_name('replay_parallel.py')
    if viewer.exists():sources[str(viewer)]=sha(viewer)
    check_sources(sources)
    out.mkdir(parents=True)
    np.savez_compressed(out/'route.npz',**arrays,times=times,tcp=tcp)
    audit=dict(complete=True,full_route_online_accepted=True,display_only=True,
        array_sha256=sha(out/'route.npz'),source_sha256=sources,run_report=str(report_file),robot_ids=report['robot_ids'],
        order=report['order_weld_ids'],original_assignment=report.get('original_assignment'),
        stages=report['stages'],timing=report['timing'],validation=report['validation'],
        continuous_collision_certified=False,frame_count=n,duration_s=float(times[-1]-times[0]),
        trajectory_preserved=True,tcp_units='mm',candidate_policy=report['candidate_policy'])
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return audit


def verify_replay(folder):
    folder=Path(folder)
    audit=json.loads((folder/'audit.json').read_text())
    if not audit.get('complete') or not audit.get('full_route_online_accepted') or not audit.get('display_only'):
        raise ValueError('Accepted display-only replay required')
    if sha(folder/'route.npz')!=audit['array_sha256']:raise ValueError('Changed replay array')
    check_sources(audit['source_sha256'])
    report_path=audit['run_report']
    if report_path not in audit['source_sha256']:raise ValueError('Unbound run report')
    report=json.loads(Path(report_path).read_text());validate_report(report)
    for key in ('robot_ids','original_assignment','stages','timing','validation','candidate_policy'):
        if audit[key]!=report[key]:raise ValueError('Replay changed bound '+key)
    if audit['order']!=report['order_weld_ids']:raise ValueError('Replay changed bound original order')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(export_parallel(args.run,args.out),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
