"""Finite 5-degree sphere refinement around a source-reproduced full-arm collision ray."""
import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import probe_analytic_retreat_v2 as exp
from analytic_ik_v2 import analytical_seed_bank, analytical_continuous_ik
from analytic_trace_motion_v2 import lift_path
from run_geometry import atomic, digest
from run_arm import full_probe
from run_qp import opening_frames
from variable_core import opening_profile, prefix_available

ROOT = exp.ROOT
HERE = Path(__file__).resolve().parent


def neighbor_bank(target, reference, count):
    local = []
    for elevation in range(-85, 90, 5):
        e = math.radians(elevation)
        for azimuth in range(0, 360, 5):
            a = math.radians(azimuth)
            local.append([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)])
    local.extend([[0.,0.,1.],[0.,0.,-1.]])
    world = np.asarray(local) @ target[:3,:3].T
    order = np.argsort(-(world @ reference), kind='stable')[:count]
    return world[order]


def kinematic_rejection(model, start, q0, direction):
    q = np.asarray(q0).copy()
    travel = 0.; step = 2.
    while travel < 2000.-1e-8:
        requested = min(2000.,travel+step)
        target = start.copy(); target[:3,3] += requested*direction
        fit = analytical_continuous_ik(model,target,q)
        if not fit['passed']:
            if step > .100001:
                step = max(.1,step/2.); continue
            return dict(passed=False,travel_mm=travel,failure=fit)
        q = np.asarray(fit['q_rad']); travel = requested; step = min(2.,step*2.)
    return dict(passed=True, q_end_rad=q.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--directions',type=int,default=256)
    parser.add_argument('--evidence-folder',type=Path,required=True)
    args = parser.parse_args()
    if not 1 <= args.directions <= 2522:
        raise ValueError('Finite sphere budget')
    out = args.out.resolve()
    if out.exists():raise FileExistsError('Use fresh output')
    exp.retreat.assert_frozen_geometry_cache()
    prior_dir = ROOT/'build/welding_retreat_200cm_other_park_v12'
    evidence_dir = args.evidence_folder.resolve()
    prior = json.loads((prior_dir/'report.json').read_text())
    evidence = json.loads((evidence_dir/'report.json').read_text())
    evidence_report = json.loads((evidence_dir/'report.json').read_text())
    sources = {**prior['manifest']['source_sha256'], **prior['manifest']['code_sha256'],
               **evidence_report['manifest']['source_sha256'], **evidence_report['manifest']['code_sha256']}
    for directory, report in ((prior_dir,prior),(evidence_dir,evidence_report)):
        sources[str((directory/'report.json').relative_to(ROOT))] = digest(directory/'report.json')
        for name,sha in report.get('output_sha256',{}).items():
            if digest(directory/name) != sha:raise RuntimeError('Output changed')
            sources[str((directory/name).relative_to(ROOT))] = sha
    codes = {str(p.relative_to(ROOT)):digest(p) for p in
             (Path(__file__), HERE/'analytic_ik_v2.py', HERE/'analytic_trace_motion_v2.py')}
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Source changed '+name)
    out.mkdir(parents=True)
    atomic(out/'manifest.json',dict(source_sha256=sources,code_sha256=codes,
        settings=dict(distance_mm=2000., azimuth_step_deg=5, elevation_step_deg=5,
                      direction_budget=args.directions, no_qp=True, explicit_angle_deg=evidence['rows'][0]['initial']['positioner_deg'],
                      kinematic_prefilter='reject-only; positives require full source trace and cold audit')))
    scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    engine = exp.retreat.VariableToolWorld(scene)
    wid = evidence['rows'][0]['initial']['transformed_weld']['id']
    original = next(r for r in prior['rows'] if r['weld_id']==wid)['initial']
    old = dict(evidence['rows'][0]['initial'])
    angle=old['positioner_deg']
    if engine.environment.layout_collisions(angle)['blocking_pairs']:raise RuntimeError('Blocked layout')
    engine.set_angle(angle)
    weld = next(w for w in scene['welds'] if w['id']==wid)
    moved,_ = engine.target(weld,old)
    rid = old['robot_id']; model = engine.world.models[rid]
    records=[]; best=None; selected_best=None; started=time.perf_counter()
    unique={}
    for record in evidence['rows']:
        record=dict(record,q_rad=record['initial']['q_rad'],other_q_rad=record['initial']['other_q_rad'])
        if record['initial']['robot_id'] != old['robot_id'] or record['initial']['positioner_deg']!=angle:
            raise RuntimeError('This finite neighborhood must keep one source robot and angle')
        key=tuple(np.round(record['q_rad'],10))
        unique.setdefault(key,record)
    seeds=sorted(unique.values(),key=lambda t:-(t.get('last_accepted_travel_mm') or 0.))
    for si,seed in enumerate(seeds):
        q=np.asarray(seed['q_rad']); start=model.tcp(q)
        park=np.asarray(seed['other_q_rad'])
        selected=dict(old,**seed['candidate'],q_rad=q.tolist(),other_q_rad=park.tolist(),
            actual_tcp_matrix=start.ravel().tolist(),target_tcp_matrix=start.ravel().tolist())
        caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
        profile=opening_profile(lambda driver:full_probe(engine,rid,q,old['other_robot_id'],park,
                    driver,moved,'closed' if driver==0. else 'opening'),caps,maximum_amplitude_mm=80.)
        bank=neighbor_bank(start,np.asarray(seed['direction_world']),args.directions)
        for di,direction in enumerate(bank):
            end=start.copy();end[:3,3]+=2000.*direction
            endpoints,stats=analytical_seed_bank(model,end,[q],random_count=0)
            record=dict(seed_index=si,direction_index=di,q_rad=q.tolist(),candidate=seed['candidate'],
                        direction_world=direction.tolist(),endpoint_candidates=len(endpoints),amplitudes=[])
            records.append(record)
            if not endpoints:record['reason']='endpoint_ik_rejected';continue
            kin=None
            for amplitude in (30.,60.,80.,10.):
                if not prefix_available(profile,amplitude):continue
                driver=-amplitude
                coarse=True
                for travel in np.arange(15.,2000.+1e-7,20.):
                    target=start.copy();target[:3,3]+=travel*direction
                    state=engine.state(rid,target,driver,moved,engine.world.static,'separated')
                    if state['blocking_pairs']:coarse=False;break
                trial=dict(amplitude_mm=amplitude)
                record['amplitudes'].append(trial)
                if not coarse:trial['reason']='coarse_geometry_rejected';continue
                geom=exp.retreat.geometry_prefix(engine,rid,start,moved,driver,direction,2000.)
                if not geom['passed']:trial.update(reason='geometry_certificate_rejected',geometry=geom);continue
                if kin is None:kin=kinematic_rejection(model,start,q,direction)
                trial['kinematic_prefilter']=kin
                if not kin['passed']:trial['reason']='finite_continuation_prefilter_rejected';continue
                row=dict(initial=selected,opening_profile=profile,best=None)
                path=lift_path(engine,weld,selected,selected,
                    dict(direction_world=direction.tolist(),direction_frame='current_world',length_mm=2000.),
                    driver_mm=driver,opening_frames=opening_frames(row,start,driver))
                trial.update(passed=path['passed'],reason=path.get('reason'),
                             last_accepted_travel_mm=path.get('last_accepted_travel_mm'),failure=path.get('failure'))
                print('NEIGHBOR_TRACE',si,di,amplitude,path['passed'],path.get('reason'),flush=True)
                if path['passed']:
                    name=f'paths/retreat_neighbor_{wid}_5deg.json'
                    path.update(weld_id=wid,fixed_distance_mm=2000.,no_qp=True,geometry_certificate=geom)
                    atomic(out/name,path)
                    best=dict(driver_mm=driver,amplitude_mm=amplitude,path_file=name,
                              path_sha256=digest(out/name),length_mm=2000.)
                    selected_best=selected;break
            if di%16==0 or best:
                atomic(out/'progress.json',dict(seed_index=si,direction_index=di,passed=2+bool(best),elapsed_s=time.perf_counter()-started))
                print('NEIGHBOR_PROGRESS',si,di,bool(best),time.perf_counter()-started,flush=True)
            if best:break
        if best:break
    rows=copy.deepcopy(prior['rows'])
    for i,row in enumerate(rows):
        if row['best']:
            src=prior_dir/row['best']['path_file'];name=f'paths/reused_retreat_{i:02d}.json'
            (out/name).parent.mkdir(parents=True,exist_ok=True);(out/name).write_bytes(src.read_bytes())
            row['best'].update(path_file=name,path_sha256=digest(out/name))
    if best:
        replacement=dict(weld_id=wid,initial=selected_best,original_initial=original,best=best,
            passed=True,required_straight_distance_mm=2000.,trials=[],neighbor_ray_reselected=True)
        rows=[replacement if r['weld_id']==wid else r for r in rows]
    atomic(out/'trials.json',dict(rows=records))
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Inputs changed during run')
    outputs={str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')}
    outputs['trials.json']=digest(out/'trials.json')
    summary=dict(complete=True,points=21,passed=sum(bool(r['best']) for r in rows),
                 required_straight_distance_mm=2000.,wall_s=time.perf_counter()-started,directions_tried=len(records))
    atomic(out/'report.json',dict(manifest=json.loads((out/'manifest.json').read_text()),
        summary=summary,rows=rows,output_sha256=outputs,
        scope='Finite nearest 5-degree sphere rays, explicit sealed source angle and source-legal other-arm park. Negative is not infeasibility proof; complete source trace positives require cold audit.'))
    print('NEIGHBOR_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
