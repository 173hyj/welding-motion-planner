"""Cold recheck direct initial states with source-XML FK and denser probes."""
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tool_world import ToolWorld, ROOT, HERE
from run_initial import OUT, extended_source_hashes, assert_frozen_geometry_cache
from run_arm import full_probe
from run_geometry import atomic, digest, initialize_worker
from arm_path import line_error
from contracts import ContactTracker

_ENGINE=None;_FK=None;_JOINTS=None;_TCP=None


def fresh_source_world(scene):
    """Replace adapted fixture GLBs with independently rebuilt raw source BVHs."""
    from build_scene import flatten
    from collision_audit import fcl_object
    engine=ToolWorld(scene)
    audit=engine.environment.audit
    fixture=engine.environment.positioner.asset
    base=np.asarray(fixture['transform']).reshape(4,4)
    raw=json.loads((ROOT/'build/fixtureL.json').read_text())
    replacements={}
    for name,node in [('DONG',2),('JING',6196)]:
        mesh=flatten(raw,[node]);mesh.apply_transform(base)
        replacements[fixture['id']+':'+name]=fcl_object(mesh,np.eye(4))
    engine.world.static=[(oid,name,replacements.get(oid,obj)) for oid,name,obj in engine.world.static]
    env=engine.environment;env.objects={oid:obj for oid,name,obj in engine.world.static}
    env.relative_env=[item for item in engine.world.static if item[0] in env.relative_ids]
    env.fixed_env=[item for item in engine.world.static if item[0] not in env.relative_ids]
    return engine


def independent_tcp(model,q):
    return model.base@_FK.flange_source(q,_JOINTS)@_TCP[model.asset['id']]


def verify_point(row, weld, out):
    engine=_ENGINE;start_time=time.perf_counter();initial=row['initial'];rid=initial['robot_id']
    q0=np.asarray(initial['q_rad']);other=initial['other_robot_id'];park=np.asarray(initial['other_q_rad'])
    engine.set_angle(initial['positioner_deg']);moved,_=engine.target(weld,initial)
    model=engine.world.models[rid];start=independent_tcp(model,q0)
    errors=[];opening_passed=True;opening_failure=None;checked_opening=0
    for driver in np.linspace(0.,-185.,371):
        state=full_probe(engine,rid,q0,other,park,float(driver),moved,'closed' if driver==0 else 'opening')
        checked_opening+=1
        errors.append(float(np.max(abs(independent_tcp(model,q0)-model.tcp(q0)))))
        if not state['passed']:
            opening_passed=False;opening_failure=dict(driver_mm=float(driver),failure=state);break
    result=dict(weld_id=weld['id'],opening_passed=opening_passed,opening_matches=opening_passed==row['opening_passed'],
                opening_samples=checked_opening,opening_failure=opening_failure,
                planned_retreat_passed=row['retreat_passed'],retreat_reverified=None,retreat_samples=0,
                maximum_source_fk_matrix_difference=0.,remaining_pairs=[])
    if row['retreat_passed'] and opening_passed:
        path=row['path'];frames=path['frames'];direction=np.asarray(path['direction_world'])
        if frames[0]['q_rad']!=initial['q_rad'] or path['q_start_rad']!=initial['q_rad']:
            raise RuntimeError('Initial q differs from archived input')
        open_state=engine.state(rid,start,-185.,moved,engine.world.static,'opening')
        context=engine.release_context(rid,start,moved,dict(open_state=open_state))
        tracker=ContactTracker(context['initial_touching']);passed=True;last_s=0.;max_line=0.;max_rotation=0.
        for a,b in zip(frames,frames[1:]):
            if a['driver_mm']!=-185. or b['driver_mm']!=-185.:continue
            qa,qb=np.asarray(a['q_rad']),np.asarray(b['q_rad'])
            for u in (.5,1.):
                q=qa+u*(qb-qa);actual=independent_tcp(model,q)
                errors.append(float(np.max(abs(actual-model.tcp(q)))))
                s=float((actual[:3,3]-start[:3,3])@direction)
                if s<=1e-8:continue
                lateral=line_error(actual[:3,3],start[:3,3],direction)
                rotation=float(np.degrees(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@actual[:3,:3]).as_rotvec())))
                phase='release' if s<=5.000001 else 'separated'
                state=full_probe(engine,rid,q,other,park,-185.,moved,phase,context)
                result['retreat_samples']+=1;max_line=max(max_line,lateral);max_rotation=max(max_rotation,rotation)
                if (not state['passed'] or lateral>.05 or rotation>.01 or s<last_s-1e-5 or
                    not tracker.advance(state.get('touching',[]),s)):
                    passed=False;result['retreat_failure']=dict(travel_mm=s,lateral_mm=lateral,rotation_deg=rotation,state=state)
                    result['remaining_pairs']=state.get('remaining_pairs',[]);break
                last_s=s
            if not passed:break
        clearance=engine.minimum_distance(rid,independent_tcp(model,np.asarray(path['q_end_rad'])),-185.,engine.world.static)
        result.update(retreat_reverified=passed and last_s>=path['length_mm']-.05 and clearance>=5.,
                      maximum_line_error_mm=max_line,maximum_rotation_error_deg=max_rotation,
                      final_clearance_mm=clearance)
    result['maximum_source_fk_matrix_difference']=max(errors,default=0.)
    result['passed']=result['opening_matches'] and (not row['retreat_passed'] or result['retreat_reverified'])
    result['elapsed_s']=time.perf_counter()-start_time
    atomic(out/(weld['id']+'_verification.json'),result)
    print('COLD_VERIFY',weld['id'],'OPEN',opening_passed,'RETREAT',result['retreat_reverified'],
          'AGREES',result['passed'],flush=True)
    return result


def main():
    global _ENGINE,_FK,_JOINTS,_TCP
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    out=args.out.resolve();report_path=out/'initial_report.json';report=json.loads(report_path.read_text())
    if not report['summary']['complete']:raise RuntimeError('Full initial search required')
    assert_frozen_geometry_cache();baseline=extended_source_hashes()
    if baseline!=report['manifest']['source_sha256']:raise RuntimeError('Source differs from tested initial run')
    for name,value in report['manifest']['code_sha256'].items():
        if digest(HERE/name)!=value:raise RuntimeError('Initial run code changed: '+name)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());welds={w['id']:w for w in scene['welds']}
    begin=time.perf_counter();_ENGINE=fresh_source_world(scene)
    spec=importlib.util.spec_from_file_location('raw_fk',ROOT/'scripts/audit_contact_release_v17.py')
    _FK=importlib.util.module_from_spec(spec);spec.loader.exec_module(_FK)
    raw=json.loads((ROOT/'build/robot.json').read_text());_JOINTS=_FK.joints_from_raw(raw)[:6]
    _TCP={a['id']:_FK.source_tcp_home(raw,a) for a in scene['assets'] if a['kind']=='robot'}
    verify_out=out/'cold_verification';verify_out.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        futures=[pool.submit(verify_point,row,welds[row['weld_id']],verify_out) for row in report['rows']]
        rows=[future.result() for future in concurrent.futures.as_completed(futures)]
    if extended_source_hashes()!=baseline:raise RuntimeError('Source changed during cold verification')
    result=dict(initial_report_sha256=digest(report_path),verified_rows=rows,
        summary=dict(points=len(rows),opening_passed=sum(r['opening_passed'] for r in rows),
            planned_successes=sum(r['planned_retreat_passed'] for r in rows),
            reverified_retreats=sum(r['retreat_reverified'] is True for r in rows),
            all_agree=all(r['passed'] for r in rows),wall_s=time.perf_counter()-begin),
        scope='Fresh raw source fixture BVHs; source-XML FK and raw PMI TCP; 0.5mm opening driver and '
              'midpoints of saved actual joint edges; full both-arm source pairs. '
              'Same contact predicate, so this is an independent FK/geometry rebuild, not an independent contact-rule implementation. '
              'No continuous full-arm or opening collision proof; no negative-path infeasibility proof.',
        assembly_exemptions=['DONG vs entire JING','DONG vs entire two source hanging-interface assemblies'],
        system_safe=False)
    atomic(out/'initial_cold_verification.json',result)
    print('COLD_SUMMARY',json.dumps(result['summary']),flush=True)
    if not result['summary']['all_agree']:raise SystemExit(2)


if __name__=='__main__':main()
