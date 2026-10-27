"""Closed-form spherical-wrist IK derived from the verified source screw axes.

The supplied solve_FANUC.cpp is a class fragment without its DH helpers or
parameters. This uses its shoulder/elbow/wrist branch structure, but derives the
geometry directly from the source model instead of assuming an undocumented DH
convention. Corrections include the opposite shoulder, rejection of genuinely
negative discriminants, deterministic singular wrists, Cartesian winding
expansion, and strict full-pose source-FK validation. J3's leading J2 coupling
is already restored by solving for its absolute forearm angle.
"""
import itertools
import math
import numpy as np
from scipy.spatial.transform import Rotation
from welding_kinematics import axis_matrix


class FanucAnalytic:
    def __init__(self, model):
        self.model=model
        self.joints=model.joints[:6]
        if len(self.joints)!=6 or model.tcp_home is None:
            raise ValueError('Analytic FANUC requires six source axes and a verified TCP home')
        points=np.array([j['axis1'] for j in self.joints],dtype=float)
        axes=np.array([np.array(j['axis2'])-j['axis1'] for j in self.joints],dtype=float)
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        expected=np.array([[0,0,1],[0,1,0],[0,-1,0],[-1,0,0],[0,-1,0],[-1,0,0]])
        if not np.allclose(axes,expected,atol=1e-9,rtol=0):
            raise ValueError('Unsupported FANUC source axis directions')
        for i,joint in enumerate(self.joints):
            keys=joint.get('leadingJoints',[]); factors=joint.get('leadingFactors',[])
            expected_keys=[self.joints[1]['key']] if i==2 else []
            if keys!=expected_keys or factors!=([1.] if i==2 else []):
                raise ValueError('Unsupported FANUC source coupling')
        self.offset=float(points[1,0]); self.height=float(points[1,2])
        self.upper=float(points[2,2]-points[1,2])
        self.fore_x=float(points[4,0]-points[2,0])
        self.fore_z=float(points[3,2]-points[2,2])
        self.wrist=np.array([points[4,0],0.,points[3,2]])
        conditions=[points[0,0],points[0,1],points[2,0]-self.offset,
                    points[3,1],points[4,2]-self.wrist[2],points[5,1],points[5,2]-self.wrist[2]]
        if max(abs(x) for x in conditions)>1e-7 or self.upper<=0 or math.hypot(self.fore_x,self.fore_z)<=0:
            raise ValueError('Unsupported FANUC source axis locations')
        self.length=math.hypot(self.fore_x,self.fore_z)
        self.beta=math.atan2(self.fore_z,self.fore_x)
        self.base_inv=np.linalg.inv(model.base)
        self.home_inv=np.linalg.inv(model.tcp_home)
        # One-time proof against the actual source product of exponentials.
        proof=np.random.default_rng(7081).uniform(-1.,1.,(12,6))
        worst_position=worst_rotation=0.
        for q in proof:
            source=self._source_transform(q)
            geometric=self._geometric_transform(q)
            worst_position=max(worst_position,float(np.linalg.norm(source[:3,3]-geometric[:3,3])))
            worst_rotation=max(worst_rotation,float(Rotation.from_matrix(source[:3,:3].T@geometric[:3,:3]).magnitude()))
        if worst_position>1e-7 or worst_rotation>1e-10:
            raise ValueError('Source/analytic frame correspondence failed')
        self.mapping_proof=dict(samples=len(proof),max_position_error_mm=worst_position,max_rotation_error_rad=worst_rotation)
        self.metadata=dict(
            method='source-axis geometric analytic IK',
            source_axes=[dict(point=np.asarray(j['axis1']).tolist(),unit_axis=a.tolist()) for j,a in zip(self.joints,axes)],
            dimensions_mm=dict(shoulder_offset=self.offset,shoulder_height=self.height,upper_arm=self.upper,
                               forearm_x=self.fore_x,forearm_z=self.fore_z,wrist_home=self.wrist.tolist()),
            correspondence='w=Rz(q1)([offset,0,height]+Ry(q2)[0,0,upper]+Ry(-q3)[fore_x,0,fore_z]); R=Rz(q1)Ry(-q3)Rx(-q4)Ry(-q5)Rx(-q6)',
            coupling='Source J3 rotation is -(q3+q2); accumulated shoulder/forearm rotation is -q3. Returned q3 is the source independent joint.',
            frames='S=inv(source_base)@world_tcp@inv(source_tcp_home); wrist=S@[wrist_home,1]',
            cpp_adaptation='solve_FANUC.cpp lacks class DH parameters and helpers. Geometric equivalent derived from source axes; opposite shoulder added, negative discriminants rejected, singular randomness removed, Cartesian winding expansion and full pose FK guards added.',
            guards=dict(position_mm=1e-5,rotation_rad=1e-7,geometry_mapping_mm=1e-7,geometry_mapping_rad=1e-10),
            deduplication='Raw actuator-vector L2 distance <1e-4 rad, with source-limit-valid representatives preferred; distinct 2pi windings preserved.',
            singular_family='Deterministic 5-degree J4 grid plus bounds/source initial J4 and Euler endpoint; J6 derived exactly. At wrist-on-base-axis singularity use a 5-degree J1 grid plus bounds/initial. These sample continua and do not prove exhaustive collision feasibility.',
            mapping_proof=self.mapping_proof)

    def _source_transform(self,q):
        t=np.eye(4)
        for i,j in enumerate(self.joints):
            p=np.asarray(j['axis1'],dtype=float)
            t=t@axis_matrix(p,np.asarray(j['axis2'])-p,q[i]+(q[1] if i==2 else 0.))
        return t

    def _geometric_transform(self,q):
        rz=Rotation.from_rotvec([0,0,q[0]]).as_matrix()
        arm=rz@Rotation.from_rotvec([0,-q[2],0]).as_matrix()
        wrist=rz@(np.array([self.offset,0,self.height])+Rotation.from_rotvec([0,q[1],0]).apply([0,0,self.upper])+Rotation.from_rotvec([0,-q[2],0]).apply([self.fore_x,0,self.fore_z]))
        rotation=arm@Rotation.from_euler('XYX',[-q[3],-q[4],-q[5]]).as_matrix()
        t=np.eye(4); t[:3,:3]=rotation; t[:3,3]=wrist-rotation@self.wrist
        return t

    def _windings(self,q):
        choices=[]
        for i,value in enumerate(q):
            low,high=self.model.bounds[:,i]
            kmin=math.ceil((low-value-1e-9)/(2*math.pi))
            kmax=math.floor((high-value+1e-9)/(2*math.pi))
            choices.append([value+2*math.pi*k for k in range(kmin,kmax+1)] or [value])
        return itertools.product(*choices)

    def _singular_grid(self,index):
        low,high=self.model.bounds[:,index]; step=math.radians(5)
        values=[k*step for k in range(math.ceil(low/step),math.floor(high/step)+1)]
        return sorted(set([*values,float(low),float(high),float(self.model.q_initial[index])]))

    def solve(self,target_world_tcp):
        target=np.asarray(target_world_tcp,dtype=float)
        if target.shape!=(4,4) or not np.all(np.isfinite(target)):
            raise ValueError('Expected finite 4x4 target TCP')
        source_target=self.base_inv@target@self.home_inv
        wrist=source_target[:3,:3]@self.wrist+source_target[:3,3]
        radius=math.hypot(wrist[0],wrist[1])
        first=math.atan2(wrist[1],wrist[0]) if radius>1e-10 else float(self.model.q_initial[0])
        stats=dict(ik_origin='analytic',mapping_proof=self.mapping_proof,arm_branches=0,
                   unreachable_branches=0,singular_wrist_branches=0,rejected_fk=0,
                   singular_family_parameterization='J4 grid 5 degrees plus initial and Euler; dependent J6 exact',
                   singular_family_sampled=False,shoulder_singular_family=bool(radius<1e-8))
        solutions=[]; seen=set()
        shoulders=[(first,radius),(first+math.pi,-radius)]
        if stats['shoulder_singular_family']:
            shoulders=[(q1,0.) for q1 in self._singular_grid(0)]
        for q1,radial in shoulders:
            x=radial-self.offset; z=wrist[2]-self.height
            distance=math.hypot(x,z)
            if distance<1e-12:
                stats['unreachable_branches']+=1; continue
            cosine=(distance*distance+self.upper*self.upper-self.length*self.length)/(2*distance*self.upper)
            if abs(cosine)>1.+1e-10:
                stats['unreachable_branches']+=1; continue
            angle=math.acos(float(np.clip(cosine,-1.,1.)))
            for phi in (math.atan2(z,x)+angle,math.atan2(z,x)-angle):
                q2=math.pi/2-phi
                psi=math.atan2(z-self.upper*math.sin(phi),x-self.upper*math.cos(phi))
                q3=psi-self.beta
                arm=Rotation.from_euler('ZY',[q1,-q3]).as_matrix()
                residual=arm.T@source_target[:3,:3]
                # Euler decomposition may turn a matrix-level exact zero into
                # beta ~2e-8 through inverse-cosine roundoff. Classify from the
                # off-axis matrix entries and reconstruct the canonical family.
                singular=math.hypot(residual[0,1],residual[0,2])<1e-9
                if singular:
                    stats['singular_wrist_branches']+=1
                    euler=np.array([math.atan2(residual[2,1],residual[1,1]),
                                    0. if residual[0,0]>=0 else math.pi,0.])
                else:
                    # Direct XYX extraction retains independent outer angles
                    # below SciPy's generic Euler gimbal-lock threshold.
                    euler=np.array([math.atan2(residual[1,0],-residual[2,0]),
                                    math.atan2(math.hypot(residual[0,1],residual[0,2]),residual[0,0]),
                                    math.atan2(residual[0,1],residual[0,2])])
                stats['arm_branches']+=1
                wrist_branches=[euler,np.array([euler[0]+math.pi,-euler[1],euler[2]+math.pi])]
                if singular:
                    stats['singular_family_sampled']=True
                    for q4 in self._singular_grid(3):
                        q6=-euler[0]-q4 if math.cos(euler[1])>0 else euler[0]+q4
                        wrist_branches.append(np.array([-q4,euler[1],-q6]))
                for angles in wrist_branches:
                    raw=[q1,q2,q3,*(-angles)]
                    for winding in self._windings(raw):
                        q=np.asarray(winding); key=tuple(np.round(q,9))
                        if key in seen: continue
                        seen.add(key)
                        actual=self.model.base@self._source_transform(q)@self.model.tcp_home
                        position=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
                        rotation=float(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude())
                        if position>1e-5 or rotation>1e-7:
                            stats['rejected_fk']+=1; continue
                        solutions.append(dict(q_rad=q.tolist(),source_joint_limits_valid=bool(self.model.limits_valid(q)),
                                              position_error_mm=position,rotation_error_rad=rotation,ik_origin='analytic'))
        # Quantized keys above only avoid exact repeated branch arithmetic.
        # Source limits are rounded decimals, so singular grid endpoints can
        # differ from an interior sample by a few nanoradians. Consolidate these
        # in raw actuator space, never modulo 2pi (windings matter to motion).
        unique=[]; vectors=[]
        for solution in solutions:
            q=np.asarray(solution['q_rad'])
            near=np.flatnonzero(np.linalg.norm(np.asarray(vectors)-q,axis=1)<1e-4) if vectors else []
            if len(near):
                index=int(near[0])
                if solution['source_joint_limits_valid'] and not unique[index]['source_joint_limits_valid']:
                    unique[index]=solution; vectors[index]=q
            else:
                unique.append(solution); vectors.append(q)
        stats['near_duplicate_solutions_removed']=len(solutions)-len(unique)
        solutions=unique
        stats.update(solutions=len(solutions),illegal_solutions=sum(not s['source_joint_limits_valid'] for s in solutions))
        return solutions,stats
