"""Cold source-geometry and source-FK recheck of grid-minimum successful paths."""
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from variable_world import VariableToolWorld
from variable_core import driver_samples
from run import ROOT, OUT
from run_initial import assert_frozen_geometry_cache, extended_source_hashes
from run_arm import full_probe
from run_geometry import atomic, digest, initialize_worker
from arm_path import line_error
from contracts import ContactTracker
import verify_initial as old_verify

_ENGINE=None;_FK=None;_JOINTS=None;_TCP=None


def source_fk_agrees(maximum_matrix_difference):
    # Translation entries are mm. Require <=1e-6 mm and <=1e-6 in R entries.
    return bool(np.isfinite(maximum_matrix_difference) and maximum_matrix_difference<=1e-6)


def independent_tcp(model,q):
    return model.base@_FK.flange_source(q,_JOINTS)@_TCP[model.asset['id']]


def verify_point(row,weld,out,root_out):
    engine=_ENGINE;started=time.perf_counter();initial=row['initial'];best=row['best']
    rid=initial['robot_id'];q0=np.asarray(initial['q_rad']);other=initial['other_robot_id'];park=np.asarray(initial['other_q_rad'])
    driver=best['driver_mm'];model=engine.world.models[rid]
    engine.set_angle(initial['positioner_deg']);moved,_=engine.target(weld,initial);start=independent_tcp(model,q0)
    path=json.loads((root_out/best['path_file']).read_text());errors=[]
    if digest(root_out/best['path_file'])!=best['path_sha256']:raise RuntimeError('Selected saved path changed')
    if (path['q_start_rad']!=initial['q_rad'] or path['frames'][0]['q_rad']!=initial['q_rad'] or
        path['driver_mm']!=driver or path['positioner_deg']!=initial['positioner_deg'] or
        path['other_q_rad']!=initial['other_q_rad'] or path['robot_id']!=rid):raise RuntimeError('Archived input or selected driver changed')
    result=dict(weld_id=weld['id'],amplitude_mm=best['amplitude_mm'],opening_passed=True,opening_samples=0,
                retreat_reverified=False,retreat_samples=0,remaining_pairs=[],planned_path_sha256=best['path_sha256'])
    for d in driver_samples(driver,.5):
        phase='closed' if d==0. else 'opening';state=full_probe(engine,rid,q0,other,park,float(d),moved,phase)
        result['opening_samples']+=1;errors.append(float(np.max(abs(independent_tcp(model,q0)-model.tcp(q0)))))
        caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
        if not state['passed'] or (d==0. and not all(state.get('certificates',{}).get(pid,{}).get('passed',False) for pid in caps)):
            result.update(opening_passed=False,opening_failure=dict(driver_mm=float(d),state=state));break
    if result['opening_passed']:
        opened=engine.state(rid,start,driver,moved,engine.world.static,'opening')
        context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
        tracker=ContactTracker(context['initial_touching']);direction=np.asarray(path['direction_world']);passed=True;last_s=0.;max_line=0.;max_rotation=0.
        frames=path['frames']
        for a,b in zip(frames,frames[1:]):
            if a['driver_mm']!=driver or b['driver_mm']!=driver:continue
            qa,qb=np.asarray(a['q_rad']),np.asarray(b['q_rad'])
            for u in (.5,1.):
                q=qa+u*(qb-qa);actual=independent_tcp(model,q)
                errors.append(float(np.max(abs(actual-model.tcp(q)))))
                s=float((actual[:3,3]-start[:3,3])@direction)
                if s<=1e-8:continue
                lateral=line_error(actual[:3,3],start[:3,3],direction)
                rotation=float(np.degrees(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@actual[:3,:3]).as_rotvec())))
                phase='release' if s<=5.000001 else 'separated';state=full_probe(engine,rid,q,other,park,driver,moved,phase,context)
                result['retreat_samples']+=1;max_line=max(max_line,lateral);max_rotation=max(max_rotation,rotation)
                if (not state['passed'] or lateral>.05 or rotation>.01 or s<last_s-1e-5 or
                    not tracker.advance(state.get('touching',[]),s)):
                    passed=False;result['retreat_failure']=dict(travel_mm=s,lateral_mm=lateral,rotation_deg=rotation,state=state)
                    result['remaining_pairs']=state.get('remaining_pairs',[]);break
                last_s=s
            if not passed:break
        endpoint=independent_tcp(model,np.asarray(path['q_end_rad']))
        clearance=engine.minimum_distance(rid,endpoint,driver,engine.world.static)
        result.update(retreat_reverified=passed and last_s>=path['length_mm']-.05 and path['length_mm']>=100. and clearance>=5.,
                      maximum_line_error_mm=max_line,maximum_rotation_error_deg=max_rotation,final_clearance_mm=clearance,final_travel_mm=last_s)
    fk_difference=max(errors,default=float('inf'))
    result.update(maximum_source_fk_matrix_difference=fk_difference,source_fk_agreement_passed=source_fk_agrees(fk_difference),
                  passed=result['opening_passed'] and result['retreat_reverified'] and source_fk_agrees(fk_difference),elapsed_s=time.perf_counter()-started)
    atomic(out/(weld['id']+'.json'),result)
    print('VARIABLE_COLD',weld['name'],'AMPLITUDE',best['amplitude_mm'],'PASS',result['passed'],flush=True)
    return result


def main():
    global _ENGINE,_FK,_JOINTS,_TCP
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('One to four isolated workers')
    out=args.out.resolve();report_path=out/'report.json';report=json.loads(report_path.read_text())
    if not report['summary']['complete'] or not report['summary']['full_opening_baseline_matches']:raise RuntimeError('Complete grid and matching baseline required')
    assert_frozen_geometry_cache();baseline=report['manifest']['source_sha256']
    for name,value in {**baseline,**report['manifest']['code_sha256']}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Tested source/code changed: '+name)
    for name,value in report['output_sha256'].items():
        if digest(out/name)!=value:raise RuntimeError('Saved output changed: '+name)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());welds={w['id']:w for w in scene['welds']}
    started=time.perf_counter()
    # This helper reloads source fixture DONG/JING triangles, avoiding adapted GLBs.
    _ENGINE=old_verify.fresh_source_world(scene)
    # Same source objects, but explicitly driver-parameterized query methods.
    _ENGINE.__class__=VariableToolWorld
    spec=importlib.util.spec_from_file_location('raw_fk',ROOT/'scripts/audit_contact_release_v17.py')
    _FK=importlib.util.module_from_spec(spec);spec.loader.exec_module(_FK)
    raw=json.loads((ROOT/'build/robot.json').read_text());_JOINTS=_FK.joints_from_raw(raw)[:6]
    _TCP={a['id']:_FK.source_tcp_home(raw,a) for a in scene['assets'] if a['kind']=='robot'}
    verify_out=out/'cold_verification';verify_out.mkdir(exist_ok=True)
    rows_to_verify=[r for r in report['rows'] if r['best']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        fs=[pool.submit(verify_point,r,welds[r['weld_id']],verify_out,out) for r in rows_to_verify]
        rows=[future.result() for future in concurrent.futures.as_completed(fs)]
    for name,value in {**baseline,**report['manifest']['code_sha256']}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Tested source/code changed during verification')
    result=dict(report_sha256=digest(report_path),verification_code_sha256={str(Path(__file__).relative_to(ROOT)):digest(__file__),
                'scripts/audit_contact_release_v17.py':digest(ROOT/'scripts/audit_contact_release_v17.py')},
        verified_rows=rows,summary=dict(selected_paths=len(rows),opening_passed=sum(r['opening_passed'] for r in rows),
                reverified_retreats=sum(r['retreat_reverified'] for r in rows),all_passed=all(r['passed'] for r in rows),wall_s=time.perf_counter()-started),
        scope='Selected grid-minimum successes only. Fresh raw source fixture meshes; source-XML FK/PMI TCP; 0.5mm opening driver and saved q-edge midpoints. '
              'Same contact predicate. Other successful amplitudes not cold-reverified; negatives not infeasibility proofs. '
              'Not a continuous entire-system safety certificate; no positioner transition or inter-weld transfer validation.',
        assembly_exemptions=['DONG vs entire JING','DONG vs entire two source hanging-interface assemblies'],system_safe=False)
    atomic(out/'cold_verification.json',result)
    print('VARIABLE_COLD_SUMMARY',json.dumps(result['summary']),flush=True)
    if not result['summary']['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
