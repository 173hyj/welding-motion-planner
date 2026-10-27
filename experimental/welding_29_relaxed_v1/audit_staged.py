"""Fresh source-FK audit of all 29 process cycles and Cartesian straight strokes."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from welding_kinematics import RobotKinematics
from contracts import validate_workflow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'build/welding_29_staged_v2')
    args=parser.parse_args();out=args.folder
    scene=json.loads((ROOT/'build/scene.json').read_text())
    raw=gzip.decompress((out/'welding_plan_29_staged_v2.json.gz').read_bytes())
    plan=json.loads(raw);audit=json.loads((out/'audit.json').read_text())
    expected=audit['plans']['welding_plan_29_staged_v2.json.gz']['sha256']
    assert hashlib.sha256(raw).hexdigest()==expected
    local=json.loads((out/'stage2_frozen_local_blocks.json').read_text())['rows']
    stage3=json.loads((out/'stage3_transfer_routes.json').read_text())
    assert stage3['stage2_sha256']==hashlib.sha256((out/'stage2_frozen_local_blocks.json').read_bytes()).hexdigest()
    models={a['id']:RobotKinematics(scene,a) for a in scene['assets'] if a['kind']=='robot'}
    source_ids=[w['id'] for w in scene['welds']]
    validate_workflow(source_ids,plan['tasks'])
    frames=plan['frames'];seams=0;maximum_fk=maximum_seam=0.;rows=[]
    for frame in frames:
        for rid,entry in frame['robots'].items():
            q=np.array(entry['q_rad']);assert models[rid].limits_valid(q)
            expected=models[rid].tcp(q);stored=np.array(entry['tcp_matrix']).reshape(4,4)
            maximum_fk=max(maximum_fk,float(np.max(np.abs(expected-stored))))
    for a,b in zip(frames,frames[1:]):
        assert b['time_s']>=a['time_s']
        if b['phase']!=a['phase'] or b['task_index']!=a['task_index']:
            seams+=1;assert abs(a['time_s']-b['time_s'])<1e-9
            for rid in models:
                maximum_seam=max(maximum_seam,float(np.max(np.abs(np.array(a['robots'][rid]['q_rad'])-b['robots'][rid]['q_rad']))))
            for gid in a['guns']:assert abs(a['guns'][gid]['driver_mm']-b['guns'][gid]['driver_mm'])<1e-9
    assert maximum_seam<1e-8 and maximum_fk<1e-7
    max_line=max_angle=max_endpoint=0.
    for index,(task,block,weld) in enumerate(zip(plan['tasks'],local,scene['welds'])):
        rid=task['assigned_robot'];model=models[rid]
        weld_pose=model.tcp(np.array(task['weld_q_rad']))
        endpoint_error=float(np.linalg.norm(weld_pose[:3,3]-weld['world_xyz']))
        max_endpoint=max(max_endpoint,endpoint_error);assert endpoint_error<.1
        entry=model.tcp(np.array(task['entry_q_rad']))
        exit=model.tcp(np.array(task['exit_q_rad']))
        assert np.max(np.abs(entry-exit))<1e-8
        direction=entry[:3,3]-weld_pose[:3,3];length=np.linalg.norm(direction);unit=direction/length
        assert abs(length-block['stroke']['distance_mm'])<.02
        selected=[f for f in frames if f['task_index']==index+1]
        assert set(['ingress','close','weld','open','retreat']).issubset({f['phase'] for f in selected})
        line_error=angle=0.
        for frame in selected:
            if frame['phase'] not in ('ingress','retreat'):continue
            actual=model.tcp(np.array(frame['robots'][rid]['q_rad']))
            delta=actual[:3,3]-weld_pose[:3,3]
            error=float(np.linalg.norm(delta-(delta@unit)*unit));line_error=max(line_error,error)
            angle=max(angle,float(np.linalg.norm(Rotation.from_matrix(weld_pose[:3,:3].T@actual[:3,:3]).as_rotvec())))
            assert -.02<=delta@unit<=length+.02
        assert line_error<.02 and angle<2e-4
        # Entire welding dwell is at the target with gun fully closed.
        dwell=[f for f in selected if f['phase']=='weld'];assert dwell
        assert abs(dwell[-1]['time_s']-dwell[0]['time_s']-.8)<1e-8
        for f in dwell:assert all(g['driver_mm']==0 for g in f['guns'].values())
        max_line=max(max_line,line_error);max_angle=max(max_angle,angle)
        rows.append(dict(weld_id=weld['id'],entry_exit_distance_mm=length,
                         tcp_error_mm=endpoint_error,line_error_mm=line_error,orientation_error_rad=angle,
                         frame_count=len(selected),local_max_observed_depth_mm=block['local_max_depth_mm']))
    for rid in models:
        assert np.max(np.abs(np.array(frames[0]['robots'][rid]['q_rad'])-frames[-1]['robots'][rid]['q_rad']))<1e-8
    for gid in frames[0]['guns']:assert abs(frames[0]['guns'][gid]['driver_mm']-frames[-1]['guns'][gid]['driver_mm'])<1e-8
    source_report=json.loads((out/'report.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in source_report['source_hashes_before'].items())
    report=dict(passed=True,task_count=29,frame_count=len(frames),duration_s=frames[-1]['time_s'],
                phase_seams=seams,maximum_q_seam_rad=maximum_seam,maximum_saved_fk_error_mm=maximum_fk,
                maximum_weld_position_error_mm=max_endpoint,maximum_tcp_line_deviation_mm=max_line,
                maximum_fixed_orientation_error_rad=max_angle,all_source_joint_limits_valid=True,
                frozen_local_blocks_unchanged=True,loop_continuous=True,source_hashes_unchanged=True,
                tasks=rows,strict_planned_task_count=0,system_safe=False,
                collision_depth_note='Uses separately saved finite FCL witnesses; no continuous collision certificate')
    (out/'independent_workflow_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
