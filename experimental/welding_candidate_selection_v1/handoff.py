"""Reconfigure the entering arm before its first weld branch is fixed."""
from pathlib import Path
import sys,time
import numpy as np


def warp_seed(original,a,b):
    original=np.asarray(original,float);a=np.asarray(a,float);b=np.asarray(b,float)
    if original.ndim!=2 or original.shape[1]!=15 or len(original)<2 or a.shape!=(15,) or b.shape!=(15,):
        raise ValueError('Full15 seed required')
    if not np.allclose(original[0],a,atol=1e-10,rtol=0):raise ValueError('Seed starts elsewhere')
    if not np.allclose(original[-1,6:13],b[6:13],atol=1e-10,rtol=0) or abs(original[-1,14]-b[14])>1e-10:
        raise ValueError('Only entering arm configuration may change')
    result=original.copy()
    progress=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(original[:,:6],axis=0),axis=1))]
    progress=progress/progress[-1] if progress[-1] else np.linspace(0,1,len(original))
    for j in (0,1,2,3,4,5,13):
        span=original[-1,j]-original[0,j]
        u=np.clip((original[:,j]-original[0,j])/span,0,1) if abs(span)>1e-10 else progress
        result[:,j]+=(b[j]-original[-1,j])*u
    result[0]=a;result[-1]=b
    return result


def connect(engine,a15,b15,baseline_states=None,seconds=60,seed=1,mesh_bounds=None,do_trajopt=False):
    from .edges import compress_collinear
    folder=Path(__file__).absolute().parent.parent/'welding_retreat_transfer_v1'
    sys.path.insert(0,str(folder))
    from world_bridge_v2 import SourceBridge
    from shared_gateway import connect_shared
    from signed_wrist_model_v40 import branch_parks
    from mixed_bridge_v95 import SIGNS
    from state_space import interpolate
    a,b=np.asarray(a15),np.asarray(b15)
    baseline=None if baseline_states is None else np.asarray(baseline_states)
    if baseline is not None and np.array_equal(a,baseline[0]) and np.array_equal(b,baseline[-1]):
        return dict(passed=True,states=baseline.copy(),stats=dict(strategy='inherited_handoff',inherited_source_validation=True))
    begun=time.perf_counter();deadline=begun+seconds
    class Expired(RuntimeError):pass
    class Proposal(SourceBridge):
        def valid_scaled(self,x):
            if time.perf_counter()>deadline:raise Expired()
            return super().valid_scaled(x)
    fine=SourceBridge(engine);proposal=Proposal(engine);proposal.resolutions[:12]=np.deg2rad(.5)
    stats=dict(inherited_source_validation=False,trials=[])
    path=None
    def accept(candidate,label):
        if all(fine.edge_scaled(x,y) for x,y in zip(candidate,candidate[1:])):
            stats['strategy']=label;return candidate
        stats['trials'].append(dict(strategy=label,passed=False));return None
    if baseline is not None:
        warped=warp_seed(baseline,a,b)
        candidate=compress_collinear(warped)/fine.scale
        path=accept(candidate,'warped_handoff_source_certified')
    if path is None:path=accept(np.array([a,b])/fine.scale,'direct_handoff')
    if path is None:
        try:
            parks=branch_parks(engine.world.models,SIGNS)
            base=a.copy()
            for i,rid in enumerate(fine.robot_ids):base[6*i:6*i+6]=parks[rid]
            base[13:]=0.
            for q0,q1 in ((-3.,-3.),(-3.,3.),(-2.,3.),(-3.,1.)):
                gate=base.copy();gate[0]=q0;gate[6]=q1
                for order in ((1,0),(0,1)):
                    candidate,detail=connect_shared(proposal,a/fine.scale,b/fine.scale,gate/fine.scale,
                                                    order,seed+len(stats['trials']),500)
                    stats['trials'].append(dict(strategy='shared_handoff',arm_order=order,stages=detail))
                    if candidate is not None:path=accept(np.asarray(candidate),'shared_handoff')
                    if path is not None:break
                if path is not None:break
        except Expired:stats['reason']='finite_handoff_search_budget'
    if path is None:
        return dict(passed=False,states=None,stats=dict(stats,last_failure=fine.last_failure or proposal.last_failure))
    states=[]
    for x,y in zip(path,path[1:]):
        sample=interpolate(x,y,fine.resolutions/fine.scale)
        states.extend(sample if not states else sample[1:])
    states=np.asarray(states)*fine.scale;states[0]=a;states[-1]=b
    stats.update(fine_checks=fine.checks,elapsed_s=time.perf_counter()-begun,
                 positioner_profile='Original profile retained by warped seed; gateway rotations monotone between fixed endpoints')
    return dict(passed=True,states=states,stats=stats,waypoints_scaled=path)
