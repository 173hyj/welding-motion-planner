"""Reproduce a completed point-angle full-arm rejection and name source pairs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import probe_analytic_retreat_v2 as exp
from analytic_trace_motion_v2 import lift_path
from run_geometry import atomic,digest
from run_qp import opening_frames
ROOT=exp.ROOT;HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--candidates',type=int,default=2)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Fresh output')
    exp.retreat.assert_frozen_geometry_cache()
    evidence=args.evidence.resolve();report=json.loads(evidence.read_text())
    parent_manifest=json.loads((evidence.parent/'manifest.json').read_text())
    sources={**parent_manifest['source_sha256'],**parent_manifest['code_sha256'],str(evidence.relative_to(ROOT)):digest(evidence),str((evidence.parent/'manifest.json').relative_to(ROOT)):digest(evidence.parent/'manifest.json')}
    codes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),HERE/'analytic_ik_v2.py',HERE/'analytic_trace_motion_v2.py')}
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Input changed '+name)
    bank=[]
    for pose in report.get('alternative_pose_trials',[]):
        for initial in pose['trials']:
            for trial in initial['directions']:
                if trial.get('reason')=='full_retreat_blocked':bank.append((pose,initial,trial))
    bank.sort(key=lambda t:-(t[2].get('last_accepted_travel_mm') or 0.))
    if not bank:raise ValueError('No full-arm collision candidates')
    out.mkdir(parents=True);atomic(out/'manifest.json',dict(source_sha256=sources,code_sha256=codes,settings=dict(diagnostic_only=True,candidates=args.candidates)))
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());engine=exp.retreat.VariableToolWorld(scene)
    angle=report['initial']['positioner_deg']
    if engine.environment.layout_collisions(angle)['blocking_pairs']:raise RuntimeError('Layout changed')
    engine.set_angle(angle)
    weld=next(w for w in scene['welds'] if w['id']==report['weld_id'])
    rows=[];started=time.perf_counter()
    for pose,initial,trial in bank[:args.candidates]:
        rid=pose['candidate']['robot_id'];other=next(r for r in engine.world.models if r!=rid)
        q=np.asarray(initial['q_rad']);model=engine.world.models[rid];target=model.tcp(q)
        selected=dict(report['original_initial'],**pose['candidate'],other_robot_id=other,q_rad=q.tolist(),other_q_rad=[-3.,-.8,.5,0.,0.,float(np.pi/2)],positioner_deg=angle,actual_tcp_matrix=target.ravel().tolist(),target_tcp_matrix=target.ravel().tolist())
        row=dict(initial=selected,best=None,opening_profile=initial['opening_profile'])
        driver=-trial['amplitude_mm']
        path=lift_path(engine,weld,selected,selected,dict(direction_world=trial['direction_world'],direction_frame='current_world',length_mm=2000.),driver_mm=driver,opening_frames=opening_frames(row,target,driver))
        detail=dict(candidate=pose['candidate'],initial=selected,amplitude_mm=trial['amplitude_mm'],direction_world=trial['direction_world'],archived_last_accepted_travel_mm=trial['last_accepted_travel_mm'],passed=path['passed'],reason=path.get('reason'),last_accepted_travel_mm=path.get('last_accepted_travel_mm'),failure_q_rad=path.get('failure_q_rad'),failure=path.get('failure'))
        rows.append(detail);atomic(out/'progress.json',dict(completed=len(rows),elapsed_s=time.perf_counter()-started))
        print('FULL_COLLISION_DIAG',report['weld_id'],detail['last_accepted_travel_mm'],detail['reason'],json.dumps(detail['failure']),flush=True)
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Source changed during diagnostic')
    atomic(out/'report.json',dict(manifest=json.loads((out/'manifest.json').read_text()),rows=rows,summary=dict(diagnostic_only=True,candidates=len(rows),wall_s=time.perf_counter()-started)))


if __name__=='__main__':main()
