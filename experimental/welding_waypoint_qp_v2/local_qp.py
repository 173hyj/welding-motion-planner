"""Sequential local QP at actual accepted and failed configurations."""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_waypoint_qp_v1'))
from qp_step import solve_step,tangent_basis
from proximity import collect_pairs,proximity_linearization,polygon_linearization
from refinement import PoseProblem


def local_step(engine,weld,selected,driver,direction,evaluation,trust_direction_deg=15.):
    started=time.perf_counter();rid=selected['robot_id'];model=engine.world.models[rid]
    other=selected['other_robot_id'];park=np.asarray(selected['other_q_rad']);q0=np.asarray(selected['q_rad'])
    moved,_=engine.target(weld,selected);start=model.tcp(q0);basis=tangent_basis(direction)
    failure=evaluation['arm'];fail_s=float(failure.get('failure_travel_mm',failure.get('travel_mm',100.)))
    geometry=evaluation['geometry'];geometry_s=float(geometry.get('failure_mm',100.)) if not geometry['passed'] else 100.
    critical=min(fail_s,geometry_s,100.)
    accepted=failure.get('accepted_retreat_waypoints',failure.get('frames',[]))
    accepted=[f for f in accepted if f['travel_mm']>0.]
    samples=[]
    for s in sorted(set([min(.2,max(.05,critical*.3)),max(.1,critical-2.),max(.1,critical-.2),max(.1,critical),min(100.,critical+2.),min(100.,critical+5.)])):
        near=min(accepted,key=lambda f:abs(f['travel_mm']-s)) if accepted else None
        seed=np.asarray(near['q_rad']) if near else q0.copy()
        if failure.get('failure_q_rad') is not None and abs(fail_s-s)<.01:
            q=np.asarray(failure['failure_q_rad']);source='actual_failure_q'
        elif near and abs(near['travel_mm']-s)<.01:
            q=seed;source='actual_accepted_q'
        else:
            target=start.copy();target[:3,3]+=s*direction;problem=PoseProblem(model,target)
            fit=least_squares(lambda x:problem.evaluate(x)[0],np.clip(seed,model.bounds[0]+1e-9,model.bounds[1]-1e-9),
                jac=lambda x:problem.evaluate(x)[1],bounds=(model.bounds[0],model.bounds[1]),max_nfev=100,
                ftol=1e-11,xtol=1e-11,gtol=1e-11)
            q=fit.x;source='local_fit_from_nearest_actual_q'
        samples.append(dict(travel_mm=s,q_rad=q.tolist(),seed_source=source))
    # Include the exact actual collision/failed point even if not on the regular local grid.
    if failure.get('failure_q_rad') is not None and fail_s<=100. and not any(abs(t['travel_mm']-fail_s)<1e-6 for t in samples):
        samples.append(dict(travel_mm=fail_s,q_rad=failure['failure_q_rad'],seed_source='actual_failure_q'))
    n=6*len(samples)+2;As=[];bs=[];Gs=[];hs=[];types=[];lower=[];upper=[];diagnostics=[];gradient_s=0.
    opened=engine.state(rid,start,driver,moved,engine.world.static,'opening')
    context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
    for i,sample in enumerate(samples):
        s=sample['travel_mm'];q=np.asarray(sample['q_rad']);target=start.copy();target[:3,3]+=s*direction
        error,jac=PoseProblem(model,target).evaluate(q)
        w=np.array([1000.,1000.,1000.,500.,500.,500.]);A=np.zeros((6,n));A[:,6*i:6*i+6]=w[:,None]*jac;A[:3,-2:]=-s*basis
        As.append(A);bs.extend((-w*error).tolist());joint_trust=np.deg2rad(8.)
        lower.extend(np.maximum(model.bounds[0]-q,-joint_trust).tolist());upper.extend(np.minimum(model.bounds[1]-q,joint_trust).tolist())
        position=float(np.linalg.norm(error[:3])*1000.);rotation=float(np.degrees(np.linalg.norm(error[3:])))
        # Distances linearized only near actual pose-matched branch, never at far unreachable points.
        mesh_qualified=position<=.1 and rotation<=.02
        pairs=[];values=np.empty(0);gradients=np.empty((0,6))
        if mesh_qualified:
            phase='release' if s<=5. else 'separated'
            pairs=collect_pairs(engine,rid,q,other,park,driver,moved,phase,context if phase=='release' else None,margin_mm=2.)
            t=time.perf_counter();values,gradients=proximity_linearization(engine,rid,q,other,park,driver,pairs);gradient_s+=time.perf_counter()-t
            for pair,value,g in zip(pairs,values,gradients):
                row=np.zeros(n);row[6*i:6*i+6]=g;Gs.append(row);hs.append(.5-value);types.append('mesh_mm')
        polygons=polygon_linearization(model,q)
        for p in polygons:
            row=np.zeros(n);row[6*i:6*i+6]=1000.*p['gradient'];Gs.append(row);hs.append(1000.*(np.deg2rad(.05)-p['signed_distance_rad']));types.append('polygon_1000rad')
        for j in range(6):
            lo_gap=q[j]-model.bounds[0,j];hi_gap=model.bounds[1,j]-q[j];margin=np.deg2rad(.05)
            if min(lo_gap,hi_gap)<np.deg2rad(2.):
                row=np.zeros(n);row[6*i+j]=1000. if lo_gap<hi_gap else -1000.
                Gs.append(row);hs.append(1000.*(margin-min(lo_gap,hi_gap)));types.append('joint_1000rad')
        diagnostics.append(dict(**sample,position_error_mm=position,rotation_error_deg=rotation,
            mesh_gradient_qualified=mesh_qualified,mesh_pairs=[dict(pair=p,proxy_mm=float(v),gradient_mm_per_rad=g.tolist()) for p,v,g in zip(pairs,values,gradients)],
            polygon_constraints=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()} for p in polygons]))
    # Box is inside the specified circular direction trust region.
    component=np.tan(np.deg2rad(trust_direction_deg))/np.sqrt(2.)
    lower.extend([-component]*2);upper.extend([component]*2)
    result=solve_step(np.vstack(As),np.asarray(bs),np.asarray(Gs) if Gs else np.empty((0,n)),np.asarray(hs),lower,upper,n_joint_variables=n-2)
    if result['solved']:
        x=np.asarray(result['step']);residual=np.maximum(0.,np.asarray(hs)-np.asarray(Gs)@x) if Gs else np.array([])
        result.update(direction_step_tangent=x[-2:].tolist(),
             maximum_mesh_linear_slack_mm=max((float(v) for v,t in zip(residual,types) if t=='mesh_mm'),default=0.),
             maximum_source_limit_linear_slack_rad=max((float(v)/1000. for v,t in zip(residual,types) if t!='mesh_mm'),default=0.))
        # Old shared solver's mixed-unit aggregate is explicitly relabeled.
        result['maximum_scaled_constraint_slack']=result.pop('maximum_collision_slack_mm')
    result.update(diagnostics=diagnostics,critical_travel_mm=critical,waypoint_count=len(samples),
                  trust_direction_deg=trust_direction_deg,proximity_gradient_wall_s=gradient_s,total_wall_s=time.perf_counter()-started)
    return result,basis
