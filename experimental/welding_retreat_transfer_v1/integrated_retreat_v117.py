"""Final sampled retreat gates evaluated during generation, never called cold."""
import hashlib
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation


def edge_union(points):
    """Original saved probes plus each saved edge's audit midpoint, once in order."""
    for a,b in zip(points,points[1:]):
        yield (np.asarray(a)+np.asarray(b))/2.,False
        yield np.asarray(b),True


def pose_metrics(actual,reference,start,direction,lower,upper):
    if not all(np.isfinite(x).all() for x in (actual,reference,start,direction)):
        raise ValueError('Nonfinite source pose')
    difference=float(np.max(abs(actual-reference)))
    if difference>1e-6:raise ValueError('Independent source XML FK disagrees')
    direction=np.asarray(direction);delta=reference[:3,3]-start[:3,3]
    s=float(delta@direction);line=float(np.linalg.norm(delta-s*direction))
    rotation=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@reference[:3,:3]).as_rotvec())))
    if line>.05 or rotation>.01 or s<lower-1e-5 or s>upper+.05:
        raise ValueError('Actual joint interpolation violates straight retreat')
    return s,line,rotation,difference


def check_path_contract(path,initial,required):
    frames=path['frames'];validation=path['online_validation']
    if (not path['passed'] or not frames or path['length_mm']!=required or
        path['fixed_distance_mm']!=required or path['q_start_rad']!=initial['q_rad'] or
        frames[0]['q_rad']!=initial['q_rad'] or path['q_end_rad']!=frames[-1]['q_rad'] or
        path['other_q_rad']!=initial['other_q_rad'] or path['other_robot_id']!=initial['other_robot_id'] or
        path['robot_id']!=initial['robot_id'] or path['positioner_deg']!=initial['positioner_deg']):
        raise ValueError('Path endpoint or source identity mismatch')
    if any(f['driver_mm']!=path['driver_mm'] for f in frames if f['travel_mm']>0.):
        raise ValueError('Retreat driver changed')
    travels=np.asarray([f['travel_mm'] for f in frames],float)
    if not np.isfinite(travels).all() or np.any(np.diff(travels)<-1e-5):raise ValueError('Nonmonotonic saved path')
    direction=np.asarray(path['direction_world']);q=np.array([f['q_rad'] for f in frames])
    if q.shape!=(len(frames),6) or not np.isfinite(q).all() or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1.)>1e-8:
        raise ValueError('Nonfinite or malformed path')
    expected=max(1,int(math.ceil(abs(path['driver_mm'])/.5)))+1 if path['driver_mm'] else 1
    opening=[f for f in frames if f['travel_mm']==0.]
    if (len(opening)!=expected or any(f['q_rad']!=initial['q_rad'] for f in opening) or
        not np.allclose([f['driver_mm'] for f in opening],np.linspace(0.,path['driver_mm'],expected),atol=1e-12,rtol=0) or
        validation.get('opening_samples')!=expected or validation.get('opening_step_mm')!=.5):
        raise ValueError('Incomplete opening prefix validation')
    if (not all(validation.get(k) is True for k in ('passed','raw_source_geometry','independent_xml_fk','saved_edge_midpoints')) or
        not all(np.isfinite(validation[k]) for k in ('maximum_source_fk_matrix_difference','maximum_line_error_mm','maximum_rotation_error_deg','final_travel_mm','endpoint_clearance_mm')) or
        validation['maximum_source_fk_matrix_difference']>1e-6 or validation['maximum_line_error_mm']>.05 or
        validation['maximum_rotation_error_deg']>.01 or abs(validation['final_travel_mm']-required)>.05 or
        validation['endpoint_clearance_mm']<5. or validation['retreat_samples']<2 or
        abs(frames[-1]['travel_mm']-required)>.05 or frames[-1]['driver_mm']!=path['driver_mm']):
        raise ValueError('Final online validation incomplete or failed')
    return True


def online_retreat_ids(report,folder):
    if not report['summary']['complete']:raise ValueError('Incomplete generation report')
    if len(report['rows'])!=report['summary']['points'] or len({r['weld_id'] for r in report['rows']})!=len(report['rows']):raise ValueError('Duplicate/missing expected point identities')
    for n,h in report['output_sha256'].items():
        if hashlib.sha256((folder/n).read_bytes()).hexdigest()!=h:raise ValueError('Output hash changed')
    accepted=set()
    for index,row in enumerate(report['rows']):
        record=f'point_{index:02d}.json'
        if record not in report['output_sha256'] or json.loads((folder/record).read_text())!=row:raise ValueError('Report row differs from sealed point record')
        if not row.get('best'):
            if row.get('passed') or row.get('online_acceptance'):raise ValueError('Rejected point claims acceptance')
            continue
        if row.get('passed') is not True:raise ValueError('Point failed')
        receipt=row.get('online_acceptance')
        if not receipt or receipt.get('passed') is not True:raise ValueError('Missing online acceptance receipt')
        p=folder/row['best']['path_file'];h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h!=row['best']['path_sha256'] or h!=receipt['path_sha256']:raise ValueError('Online path hash changed')
        path=json.loads(p.read_text())
        if path.get('weld_id')!=row['weld_id']:raise ValueError('Path weld identity changed')
        if any(row['best'].get(k)!=path[k] for k in ('driver_mm','length_mm')) or row['best'].get('amplitude_mm')!=abs(path['driver_mm']):raise ValueError('Selected opening/length mismatch')
        check_path_contract(path,row['initial'],report['summary']['required_straight_distance_mm'])
        if path['online_validation']!=receipt['validation']:raise ValueError('Online receipt validation mismatch')
        if row['weld_id'] in accepted:raise ValueError('Duplicate weld identity')
        accepted.add(row['weld_id'])
    return accepted


def lift_path(engine,weld,candidate,selected,exit_path,*,driver_mm,opening_frames=None):
    """Load-free online acceptance. engine.online_tcp is independent raw XML FK."""
    from analytic_trace_motion_v2 import continuous_ik,interpolation_probes,ContactTracker,full_probe,resolve_exit_direction
    from variable_core import driver_samples
    from contracts import ContactTracker
    import time
    started=time.perf_counter();rid=selected['robot_id'];other=selected['other_robot_id']
    model=engine.world.models[rid];q0=np.asarray(selected['q_rad']);park=np.asarray(selected['other_q_rad'])
    driver_samples(driver_mm,.5)
    if not model.limits_valid(q0) or not engine.world.models[other].limits_valid(park):
        return dict(passed=False,reason='initial_source_limits')
    start=model.tcp(q0);ref=engine.online_tcp(rid,q0)
    other_error=float(np.max(abs(engine.online_tcp(other,park)-engine.world.models[other].tcp(park))))
    try:pose_metrics(start,ref,ref,[1.,0.,0.],0.,0.)
    except ValueError as exc:return dict(passed=False,reason=str(exc))
    if other_error>1e-6:return dict(passed=False,reason='other_arm_independent_FK')
    moved,_=engine.target(weld,selected);direction=np.asarray(resolve_exit_direction(exit_path,engine.delta))
    length=float(exit_path['length_mm']);frames=[];maximum_fk=max(other_error,float(np.max(abs(start-ref))))
    validation=dict(passed=False,raw_source_geometry=True,independent_xml_fk=True,saved_edge_midpoints=True,
        opening_step_mm=.5,opening_samples=0,retreat_samples=0,collision_checks=0)
    caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
    if not caps:return dict(passed=False,reason='No source electrodes')
    for driver in driver_samples(driver_mm,.5):
        phase='closed' if driver==0. else 'opening'
        state=full_probe(engine,rid,q0,other,park,float(driver),moved,phase)
        validation['opening_samples']+=1;validation['collision_checks']+=1
        if not state['passed'] or (driver==0. and not all(state.get('certificates',{}).get(pid,{}).get('passed',False) for pid in caps)):
            return dict(passed=False,reason='final_opening_gate_failed',failure=state,online_validation=validation)
        frames.append(dict(q_rad=q0.tolist(),driver_mm=float(driver),phase=phase,tcp_xyz_mm=start[:3,3].tolist(),travel_mm=0.))
    opened=engine.state(rid,start,driver_mm,moved,engine.world.static,'opening')
    context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver_mm)
    if any(np.asarray(s['outward_normal'])@direction<=1e-6 for s in context['supports']):
        return dict(passed=False,reason='release_direction_rejected')
    tracker=ContactTracker(context['initial_touching']);q=q0.copy();travel=0.;step=2.
    max_line=max_rotation=max_joint=0.;last_s=0.
    while travel<length-1e-8:
        requested=min(length,travel+step);target=start.copy();target[:3,3]+=requested*direction
        fit=continuous_ik(model,target,q)
        if not fit['passed']:
            if step>.100001:step=max(.1,step/2.);continue
            return dict(passed=False,reason='continuous_ik_failed',failure=fit,last_accepted_travel_mm=travel)
        next_q=np.asarray(fit['q_rad']);data=[]
        try:
            for qi,save in edge_union(interpolation_probes(q,next_q)):
                actual=model.tcp(qi);reference=engine.online_tcp(rid,qi)
                metrics=pose_metrics(actual,reference,ref,direction,last_s,requested)
                data.append((qi,save,actual,metrics))
        except ValueError as exc:
            if 'FK' in str(exc):return dict(passed=False,reason=str(exc))
            if step>.100001:step=max(.1,step/2.);continue
            return dict(passed=False,reason=str(exc),last_accepted_travel_mm=travel)
        for qi,save,actual,(s,line,rotation,fk_error) in data:
            phase='release' if s<=5.000001 else 'separated'
            state=full_probe(engine,rid,qi,other,park,driver_mm,moved,phase,context)
            validation['retreat_samples']+=1;validation['collision_checks']+=1
            if not state['passed'] or not tracker.advance(state.get('touching',[]),s):
                return dict(passed=False,reason='final_midpoint_or_source_collision_gate_failed',failure=state,last_accepted_travel_mm=last_s)
            last_s=s;max_line=max(max_line,line);max_rotation=max(max_rotation,rotation);maximum_fk=max(maximum_fk,fk_error)
            if save:frames.append(dict(q_rad=qi.tolist(),driver_mm=driver_mm,phase=phase,tcp_xyz_mm=actual[:3,3].tolist(),travel_mm=s))
        max_joint=max(max_joint,float(np.rad2deg(np.max(abs(next_q-q)))))
        q=next_q;travel=requested;step=min(2.,step*2.)
    clearance=float(engine.minimum_distance(rid,engine.online_tcp(rid,q),driver_mm,engine.world.static))
    if not np.isfinite(clearance) or clearance<5. or abs(last_s-length)>.05:
        return dict(passed=False,reason='final_distance_or_clearance_gate_failed',last_accepted_travel_mm=last_s)
    validation.update(passed=True,maximum_source_fk_matrix_difference=maximum_fk,maximum_line_error_mm=max_line,
        maximum_rotation_error_deg=max_rotation,final_travel_mm=last_s,endpoint_clearance_mm=clearance,elapsed_s=time.perf_counter()-started)
    return dict(passed=True,driver_mm=driver_mm,robot_id=rid,other_robot_id=other,other_q_rad=park.tolist(),
        positioner_deg=selected['positioner_deg'],candidate=candidate,q_start_rad=q0.tolist(),q_end_rad=q.tolist(),
        direction_world=direction.tolist(),length_mm=length,fixed_distance_mm=length,endpoint_clearance_mm=clearance,
        frames=frames,maximum_sampled_line_error_mm=max_line,maximum_sampled_rotation_error_deg=max_rotation,
        maximum_ik_step_joint_change_deg=max_joint,actual_joint_interpolation_checked=True,online_validation=validation,
        no_qp=True,scope='Raw original fixture geometry + independent XML FK online. Saved endpoints and midpoints sampled once. Not independent cold or continuous certification; existing source assembly exemptions retained.')
