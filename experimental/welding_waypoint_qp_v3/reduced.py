"""Eliminate exact-pose joint increments, optimize two shared direction variables."""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_waypoint_qp_v1'))
from qp_step import tangent_basis,solve_step
from proximity import collect_pairs,proximity_linearization,polygon_linearization,pair_objects,mesh_proxy
from refinement import PoseProblem


def direction_joint_jacobian(pose_jacobian,basis,travel_mm):
    # pose_jacobian uses translation in mm and rotation scaled by500.
    desired=np.vstack((travel_mm*np.asarray(basis),np.zeros((3,2))))
    return np.linalg.pinv(np.asarray(pose_jacobian),rcond=1e-8)@desired


def evaluate_line(engine,weld,selected,driver,direction):
    begin=time.perf_counter();rid=selected['robot_id'];model=engine.world.models[rid]
    other=selected['other_robot_id'];park=np.asarray(selected['other_q_rad']);q=np.asarray(selected['q_rad']);start=model.tcp(q)
    moved,_=engine.target(weld,selected);direction=np.asarray(direction);direction/=np.linalg.norm(direction)
    opened=engine.state(rid,start,driver,moved,engine.world.static,'opening')
    context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
    samples=[];merit=0.;first_invalid=100.;maximum_source_violation=0.;maximum_penetration=0.;residual=0.
    for s in [.2,.5,1.]+list(np.arange(2.,100.0001,2.)):
        target=start.copy();target[:3,3]+=s*direction;problem=PoseProblem(model,target)
        fit=least_squares(lambda x:problem.evaluate(x)[0],q,jac=lambda x:problem.evaluate(x)[1],
              max_nfev=60,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        next_q=fit.x;error,jac=problem.evaluate(next_q);position=float(np.linalg.norm(error[:3])*1000.);rotation=float(np.degrees(np.linalg.norm(error[3:])))
        qualified=position<.02 and rotation<.005 and np.max(abs(next_q-q))<=np.deg2rad(5.)
        residual+=position**2+(500.*np.linalg.norm(error[3:]))**2
        if not qualified:
            merit+=1e4+residual;first_invalid=min(first_invalid,s)
            samples.append(dict(travel_mm=s,q_rad=next_q.tolist(),pose_qualified=False,position_error_mm=position,rotation_error_deg=rotation));break
        q=next_q;phase='release' if s<=5. else 'separated'
        pairs=collect_pairs(engine,rid,q,other,park,driver,moved,phase,context if phase=='release' else None,margin_mm=2.)
        values=np.array([mesh_proxy(*pair_objects(engine,p)) for p in pairs])
        violations=np.maximum(0.,.5-values)
        source_lo=np.maximum(0.,model.bounds[0]-q);source_hi=np.maximum(0.,q-model.bounds[1]);polys=polygon_linearization(model,q)
        source_violation=sum(source_lo**2)+sum(source_hi**2)+sum(max(0.,-p['signed_distance_rad'])**2 for p in polys)
        maximum_source_violation=max(maximum_source_violation,float(max(np.max(source_lo),np.max(source_hi),max((max(0.,-p['signed_distance_rad']) for p in polys),default=0.))))
        maximum_penetration=max(maximum_penetration,float(max((max(0.,-v) for v in values),default=0.)))
        merit+=float(sum(violations**2))+1e6*source_violation
        if source_violation>1e-12 or np.any(values<0.):first_invalid=min(first_invalid,s)
        weights=np.array([1000.,1000.,1000.,500.,500.,500.])
        samples.append(dict(travel_mm=s,q_rad=q.tolist(),pose_qualified=True,position_error_mm=position,rotation_error_deg=rotation,
             source_limits_valid=model.limits_valid(q),math_pose_jacobian=weights[:,None]*jac,
             mesh_pairs=[dict(pair=p,proxy_mm=float(v)) for p,v in zip(pairs,values)],
             source_polygons=polys,source_violation_rad=float(np.sqrt(source_violation))))
    end=model.tcp(q);clearance=engine.minimum_distance(rid,end,driver,engine.world.static) if samples[-1]['travel_mm']==100. and samples[-1]['pose_qualified'] else None
    if clearance is not None:merit+=max(0.,5.-clearance)**2
    return dict(direction_world=direction.tolist(),merit=float(merit),samples=samples,first_invalid_sample_mm=first_invalid,
                maximum_source_violation_rad=maximum_source_violation,maximum_sampled_penetration_mm=maximum_penetration,
                endpoint_tool_clearance_mm=clearance,pose_residual_cost=float(residual),elapsed_s=time.perf_counter()-begin,
                internal_math_branch_may_violate_source_limits=True)


def reduced_step(engine,selected,driver,evaluation,trust_deg=15.):
    started=time.perf_counter();rid=selected['robot_id'];other=selected['other_robot_id'];park=np.asarray(selected['other_q_rad']);model=engine.world.models[rid]
    basis=tangent_basis(evaluation['direction_world']);Gs=[];hs=[];types=[];worst=[];gradient_time=0.
    for sample in evaluation['samples']:
        if not sample['pose_qualified']:continue
        q=np.asarray(sample['q_rad']);mapping=direction_joint_jacobian(sample['math_pose_jacobian'],basis,sample['travel_mm'])
        for j in range(6):
            # All source limits kept. No polygon convex-hull replacement.
            Gs.extend([1000.*mapping[j],-1000.*mapping[j]])
            hs.extend([1000.*(model.bounds[0,j]+1e-6-q[j]),1000.*(q[j]-model.bounds[1,j]+1e-6)])
            types.extend(['source_1000rad','source_1000rad'])
        for poly in sample['source_polygons']:
            Gs.append(1000.*poly['gradient']@mapping);hs.append(1000.*(1e-6-poly['signed_distance_rad']));types.append('source_1000rad')
        for pair in sample['mesh_pairs']:
            worst.append((max(0.,.5-pair['proxy_mm']),sample,pair,mapping))
    # Linearize up to12 worst pairs, spread across critical travel locations.
    worst.sort(key=lambda item:-item[0]);selected_pairs=[];covered={}
    for severity,sample,pair,mapping in worst:
        bucket=int(sample['travel_mm']//8.)
        if covered.get(bucket,0)>=2:continue
        selected_pairs.append((sample,pair,mapping));covered[bucket]=covered.get(bucket,0)+1
        if len(selected_pairs)>=12:break
    diagnostics=[]
    for sample,pair,mapping in selected_pairs:
        q=np.asarray(sample['q_rad']);t=time.perf_counter()
        values,gradients=proximity_linearization(engine,rid,q,other,park,driver,[pair['pair']]);gradient_time+=time.perf_counter()-t
        Gs.append(gradients[0]@mapping);hs.append(.5-values[0]);types.append('mesh_mm')
        diagnostics.append(dict(travel_mm=sample['travel_mm'],pair=pair['pair'],proxy_mm=float(values[0]),direction_gradient_mm=(gradients[0]@mapping).tolist()))
    component=np.tan(np.deg2rad(trust_deg))/np.sqrt(2.)
    fit=solve_step(np.zeros((0,2)),np.zeros(0),np.asarray(Gs) if Gs else np.empty((0,2)),np.asarray(hs),[-component]*2,[component]*2,n_joint_variables=0)
    if fit['solved']:
        fit['direction_step_tangent']=fit['step'];fit['maximum_scaled_constraint_slack']=fit.pop('maximum_collision_slack_mm')
        fit['minimum_scaled_constraint_residual']=fit.pop('collision_linear_residual_min_mm')
    fit.update(diagnostics=diagnostics,constraint_count=len(hs),gradient_wall_s=gradient_time,total_wall_s=time.perf_counter()-started,trust_direction_deg=trust_deg)
    return fit,basis
