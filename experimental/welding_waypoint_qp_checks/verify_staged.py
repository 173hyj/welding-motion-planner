"""Cold verification of straight release prefixes followed by joint-space transfer."""
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
from variable_world import VariableToolWorld
from variable_core import driver_samples
from run_geometry import atomic,digest,initialize_worker
from run_initial import assert_frozen_geometry_cache
from run_arm import full_probe
from tool_world import ROOT
from arm_path import line_error
from contracts import ContactTracker
import verify_initial as old_verify
_ENGINE=None;_FK=None;_JOINTS=None;_TCP=None


def independent_tcp(model,q):return model.base@_FK.flange_source(q,_JOINTS)@_TCP[model.asset['id']]


def verify_row(row,weld,out,root_out):
    started=time.perf_counter();initial=row['initial'];best=row['best'];engine=_ENGINE;rid=initial['robot_id'];other=initial['other_robot_id']
    model=engine.world.models[rid];q0=np.asarray(initial['q_rad']);park=np.asarray(initial['other_q_rad']);driver=best['driver_mm']
    path=json.loads((root_out/best['path_file']).read_text())
    if digest(root_out/best['path_file'])!=best['path_sha256']:raise RuntimeError('Saved path changed')
    if path['q_start_rad']!=initial['q_rad'] or path['frames'][0]['q_rad']!=initial['q_rad'] or path['driver_mm']!=driver:raise RuntimeError('Archived q/driver changed')
    engine.set_angle(initial['positioner_deg']);moved,_=engine.target(weld,initial);start=independent_tcp(model,q0)
    result=dict(weld_id=weld['id'],planned_path_sha256=best['path_sha256'],passed=True,opening_samples=0,prefix_and_rrt_samples=0,
                maximum_source_fk_matrix_difference=float(np.max(abs(independent_tcp(engine.world.models[other],park)-engine.world.models[other].tcp(park)))))
    for d in driver_samples(driver,.5):
        state=full_probe(engine,rid,q0,other,park,float(d),moved,'closed' if d==0. else 'opening');result['opening_samples']+=1
        result['maximum_source_fk_matrix_difference']=max(result['maximum_source_fk_matrix_difference'],float(np.max(abs(independent_tcp(model,q0)-model.tcp(q0)))))
        caps=[p for p,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
        if not state['passed'] or (d==0. and not all(state.get('certificates',{}).get(p,{}).get('passed',False) for p in caps)):
            result.update(passed=False,failure=dict(phase='opening',driver_mm=float(d),state=state));break
    opened=engine.state(rid,start,driver,moved,engine.world.static,'opening')
    context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver);tracker=ContactTracker(context['initial_touching']);max_line=0.;max_rotation=0.;last_s=0.;direction=np.asarray(path['direction_world']);rrt_seen=False
    if result['passed']:
        for a,b in zip(path['frames'],path['frames'][1:]):
            if a['driver_mm']!=driver or b['driver_mm']!=driver:continue
            qa,qb=np.asarray(a['q_rad']),np.asarray(b['q_rad'])
            for u in (.5,1.):
                q=qa+u*(qb-qa);actual=independent_tcp(model,q);result['prefix_and_rrt_samples']+=1
                result['maximum_source_fk_matrix_difference']=max(result['maximum_source_fk_matrix_difference'],float(np.max(abs(actual-model.tcp(q)))))
                transfer=b['phase']=='rrt_transfer'
                if transfer:phase='separated';rrt_seen=True;s=None
                else:
                    s=float((actual[:3,3]-start[:3,3])@direction)
                    if s<=1e-8:continue
                    lateral=line_error(actual[:3,3],start[:3,3],direction)
                    rotation=float(np.degrees(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@actual[:3,:3]).as_rotvec())))
                    max_line=max(max_line,lateral);max_rotation=max(max_rotation,rotation)
                    if lateral>.05 or rotation>.01 or s<last_s-1e-5:
                        result.update(passed=False,failure=dict(phase='prefix_shape',travel_mm=s,lateral_mm=lateral,rotation_deg=rotation));break
                    phase='release' if s<=5.000001 else 'separated'
                state=full_probe(engine,rid,q,other,park,driver,moved,phase,context if phase=='release' else None)
                if not state['passed'] or (not transfer and not tracker.advance(state.get('touching',[]),s)):
                    result.update(passed=False,failure=dict(phase='rrt' if transfer else 'prefix',state=state,travel_mm=s));break
                if not transfer:last_s=s
            if not result['passed']:break
    end=independent_tcp(model,np.asarray(path['q_end_rad']));clearance=engine.minimum_distance(rid,end,driver,engine.world.static)
    displacement=float(np.linalg.norm(end[:3,3]-start[:3,3]))
    result.update(endpoint_tool_clearance_mm=clearance,endpoint_displacement_mm=displacement,maximum_prefix_line_error_mm=max_line,
           maximum_prefix_rotation_error_deg=max_rotation,straight_prefix_mm=path['straight_prefix_mm'],rrt_transfer_checked=rrt_seen)
    result['passed']=result['passed'] and rrt_seen and last_s>=path['straight_prefix_mm']-.05 and clearance>=5. and displacement>=100. and result['maximum_source_fk_matrix_difference']<=1e-6
    result['elapsed_s']=time.perf_counter()-started;atomic(out/(row['weld_id']+'.json'),result)
    print('STAGED_COLD',row['weld_id'],'PASS',result['passed'],flush=True);return result


def main():
    global _ENGINE,_FK,_JOINTS,_TCP
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'build/welding_waypoint_qp_v4');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('One to4workers')
    out=args.out.resolve();rp=out/'report.json';report=json.loads(rp.read_text())
    if not report['summary']['complete']:raise RuntimeError('Complete staged experiment required')
    report_digest=digest(rp)
    verification_sources={str(Path(__file__).relative_to(ROOT)):digest(__file__),'scripts/audit_contact_release_v17.py':digest(ROOT/'scripts/audit_contact_release_v17.py')}
    assert_frozen_geometry_cache();hashes={}
    for section in ('source_sha256','previous_code_sha256','code_sha256'):hashes.update(report['manifest'][section])
    for n,v in hashes.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Frozen input/code changed')
    for n,v in report['output_sha256'].items():
        if digest(out/n)!=v:raise RuntimeError('Saved output changed')
    started=time.perf_counter();scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());welds={w['id']:w for w in scene['welds']}
    _ENGINE=old_verify.fresh_source_world(scene);_ENGINE.__class__=VariableToolWorld
    spec=importlib.util.spec_from_file_location('raw_fk',ROOT/'scripts/audit_contact_release_v17.py');_FK=importlib.util.module_from_spec(spec);spec.loader.exec_module(_FK)
    raw=json.loads((ROOT/'build/robot.json').read_text());_JOINTS=_FK.joints_from_raw(raw)[:6];_TCP={a['id']:_FK.source_tcp_home(raw,a) for a in scene['assets'] if a['kind']=='robot'}
    selected=[r for r in report['rows'] if r['best']];vo=out/'cold_verification';vo.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        fs=[pool.submit(verify_row,r,welds[r['weld_id']],vo,out) for r in selected];rows=[f.result() for f in concurrent.futures.as_completed(fs)]
    for n,v in hashes.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Frozen input changed during cold verification')
    if digest(rp)!=report_digest:raise RuntimeError('Planned report changed during cold verification')
    for n,v in report['output_sha256'].items():
        if digest(out/n)!=v:raise RuntimeError('Planned output changed during cold verification')
    for n,v in verification_sources.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Verification code changed during cold run')
    result=dict(report_sha256=report_digest,verified_rows=rows,verification_code_sha256=verification_sources,
       summary=dict(selected_paths=len(rows),reverified_staged_paths=sum(r['passed'] for r in rows),all_passed=all(r['passed'] for r in rows),wall_s=time.perf_counter()-started),
       scope='Fresh raw fixture meshes and independent XML FK agreement;0.5mm opening, actual saved q-edge midpoint rechecks; straight prefix geometry/shape only beforeRRT. '
             'Full source both-arm checks at samples. Same contact rules and source assembly exemptions; no entire-system continuous safety certificate.',system_safe=False)
    atomic(out/'cold_verification.json',result);print('STAGED_COLD_SUMMARY',json.dumps(result['summary']),flush=True)
    if not result['summary']['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
