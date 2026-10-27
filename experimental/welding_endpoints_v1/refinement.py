"""Deterministic multi-branch inverse kinematics for fixed-normal roll targets."""
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


def roll_target(base, angle_deg, sign):
    target=np.asarray(base).copy()
    if sign < 0:
        target[:3,:3]=target[:3,:3]@Rotation.from_euler('x',np.pi).as_matrix()
    target[:3,:3]=target[:3,:3]@Rotation.from_euler('z',angle_deg,degrees=True).as_matrix()
    return target


def refinement_angles(step):
    return [float(a) for a in np.arange(0.,360.,step) if abs(a-round(a))>1e-9]


class PoseProblem:
    def __init__(self, model, target):
        self.model,self.target=model,np.asarray(target)
        self.coefficients=np.eye(6)
        self.axes=[]
        for i,j in enumerate(model.joints[:6]):
            p=np.asarray(j['axis1'],float)
            axis=np.asarray(j['axis2'],float)-p;axis/=np.linalg.norm(axis)
            x,y,z=axis
            cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            self.axes.append((p,axis,cross,cross@cross))
            for key,factor in zip(j['leadingJoints'],j['leadingFactors']):
                self.coefficients[i,model.key_index[key]]+=factor
        self.previous=None

    def evaluate(self,q):
        if self.previous is not None and np.array_equal(q,self.previous):
            return self.error,self.jacobian
        transform=self.model.base.copy();axes=[];points=[]
        for angle,(p,axis,cross,squared) in zip(self.coefficients@q,self.axes):
            axes.append(transform[:3,:3]@axis)
            points.append(transform[:3,:3]@p+transform[:3,3])
            rotation=np.eye(3)+math.sin(angle)*cross+(1.-math.cos(angle))*squared
            motion=np.eye(4);motion[:3,:3]=rotation;motion[:3,3]=p-rotation@p
            transform=transform@motion
        pose=transform@self.model.tcp_home
        phi=Rotation.from_matrix(self.target[:3,:3].T@pose[:3,:3]).as_rotvec()
        spatial=np.asarray(axes).T
        linear=np.cross(np.asarray(axes),pose[:3,3]-np.asarray(points)).T
        x,y,z=phi;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        theta=np.linalg.norm(phi)
        coefficient=1./12. if theta<1e-5 else (1.-.5*theta/math.tan(.5*theta))/theta**2
        inverse_left=np.eye(3)-.5*cross+coefficient*(cross@cross)
        self.error=np.r_[(pose[:3,3]-self.target[:3,3])/1000.,phi]
        self.jacobian=np.vstack((linear/1000.,inverse_left@self.target[:3,:3].T@spatial))@self.coefficients
        self.previous=np.asarray(q).copy()
        return self.error,self.jacobian


def solve_seed_bank(model,target,seeds,random_count=48):
    problem=PoseProblem(model,target)
    starts=[np.asarray(q,dtype=float) for q in seeds]+[model.q_initial,np.zeros(6)]
    if random_count:
        sampler=qmc.Sobol(6,scramble=True,seed=20261008)
        unit=sampler.random_base2(int(np.ceil(np.log2(max(128,random_count*4)))))
        random=model.bounds[0]+unit*(model.bounds[1]-model.bounds[0])
        starts.extend([q for q in random if model.limits_valid(q)][:random_count])
    starts=list({tuple(np.round(q,7)):q for q in starts}.values())
    solutions=[];stats=dict(starts=len(starts),converged_pose_count=0,illegal_solutions=0,best_residual=float('inf'))
    for seed in starts:
        fit=least_squares(lambda q:problem.evaluate(q)[0],np.clip(seed,model.bounds[0]+1e-8,model.bounds[1]-1e-8),
            jac=lambda q:problem.evaluate(q)[1],bounds=(model.bounds[0],model.bounds[1]),max_nfev=220,
            ftol=1e-11,xtol=1e-11,gtol=1e-11)
        error=problem.evaluate(fit.x)[0]
        stats['best_residual']=min(stats['best_residual'],float(np.linalg.norm(error)))
        if np.linalg.norm(error[:3])*1000.>=.05 or np.linalg.norm(error[3:])>=1e-5:continue
        stats['converged_pose_count']+=1
        if not model.limits_valid(fit.x):
            stats['illegal_solutions']+=1;continue
        if any(np.linalg.norm(np.arctan2(np.sin(fit.x-np.asarray(s['q_rad'])),np.cos(fit.x-np.asarray(s['q_rad']))))<1e-4 for s in solutions):continue
        solutions.append(dict(q_rad=fit.x.tolist(),position_error_mm=float(np.linalg.norm(error[:3])*1000.),
            rotation_error_rad=float(np.linalg.norm(error[3:]))))
    return solutions,stats
