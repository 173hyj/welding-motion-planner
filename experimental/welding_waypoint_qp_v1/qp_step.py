"""One convex QP for shared line-direction change and waypoint joint increments."""
import numpy as np
from scipy import sparse
import osqp


def tangent_basis(direction):
    d=np.asarray(direction,float);d=d/np.linalg.norm(d)
    axis=np.eye(3)[np.argmin(abs(d))]
    a=np.cross(d,axis);a/=np.linalg.norm(a)
    return np.column_stack((a,np.cross(d,a)))


def updated_direction(direction,basis,step,alpha=1.):
    d=np.asarray(direction,float)+alpha*np.asarray(basis)@np.asarray(step)
    return d/np.linalg.norm(d)


def polygon_signed_distance(point,polygon):
    p=np.asarray(point,float);poly=np.asarray(polygon,float);minimum=float('inf');inside=False
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        edge=b-a;length=float(edge@edge)
        nearest=a+np.clip((p-a)@edge/length,0.,1.)*edge if length else a
        minimum=min(minimum,float(np.linalg.norm(p-nearest)))
        if (a[1]>p[1])!=(b[1]>p[1]) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return minimum if inside else -minimum


def solve_step(pose_matrix,pose_rhs,collision_matrix,collision_rhs,lower,upper,*,n_joint_variables):
    A=np.asarray(pose_matrix,float);b=np.asarray(pose_rhs,float)
    G=np.asarray(collision_matrix,float);h=np.asarray(collision_rhs,float)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float);n=len(lower);m=len(h)
    if A.shape[1]!=n or G.shape!=(m,n) or np.any(lower>upper):raise ValueError('Invalid QP dimensions or bounds')
    reg=np.r_[np.full(n_joint_variables,.01),np.full(n-n_joint_variables,.1)]
    P=sparse.block_diag((sparse.csc_matrix(200.*A.T@A+np.diag(reg)),sparse.eye(m)*2e5),format='csc')
    c=np.r_[-200.*A.T@b,np.zeros(m)]
    inequalities=sparse.hstack((sparse.csc_matrix(G),sparse.eye(m)),format='csc')
    identity=sparse.eye(n+m,format='csc')
    constraints=sparse.vstack((identity,inequalities),format='csc')
    l=np.r_[lower,np.zeros(m),h];u=np.r_[upper,np.full(m,np.inf),np.full(m,np.inf)]
    solver=osqp.OSQP();solver.setup(P=sparse.triu(P,format='csc'),q=c,A=constraints,l=l,u=u,verbose=False,
                                   eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,polishing=True)
    fit=solver.solve(raise_error=False);solved=fit.info.status_val in (1,2) and fit.x is not None
    result=dict(solved=solved,status=fit.info.status,iterations=int(fit.info.iter),solve_time_s=float(fit.info.run_time))
    if solved:
        x=fit.x[:n];slack=fit.x[n:]
        result.update(step=x.tolist(),maximum_collision_slack_mm=float(np.max(slack)) if m else 0.,
                      linear_pose_residual_norm=float(np.linalg.norm(A@x-b)),
                      collision_linear_residual_min_mm=float(np.min(G@x-h)) if m else None)
    return result
