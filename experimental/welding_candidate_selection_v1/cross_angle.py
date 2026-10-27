"""Try a source-certified branch-continuous positioner transition proposal."""
import numpy as np
from . import edges


def warp_seed(original,a,b):
    old=np.asarray(original,float);a=np.asarray(a,float);b=np.asarray(b,float)
    if old.ndim!=2 or old.shape[1]!=15 or len(old)<2 or a.shape!=(15,) or b.shape!=(15,):
        raise ValueError('Finite full15 seed and endpoints required')
    if not np.isfinite(old).all() or not np.isfinite([a,b]).all():
        raise ValueError('Nonfinite transition')
    fixed=[6,7,8,9,10,11,14]
    if not np.all(old[:,fixed]==a[fixed]) or not np.array_equal(a[fixed],b[fixed]):
        raise ValueError('Inactive arm must remain fixed')
    if old[0,12]!=a[12] or old[-1,12]!=b[12]:
        raise ValueError('Original positioner profile endpoints required')
    progress=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(old[:,:6],axis=0),axis=1))]
    progress=progress/progress[-1] if progress[-1] else np.linspace(0,1,len(old))
    result=old.copy()
    for j in (0,1,2,3,4,5,13):
        span=old[-1,j]-old[0,j]
        u=np.clip((old[:,j]-old[0,j])/span,0,1) if abs(span)>1e-10 else progress
        result[:,j]+=(a[j]-old[0,j])*(1-u)+(b[j]-old[-1,j])*u
    result[0]=a;result[-1]=b
    return result


def connect(engine,a15,b15,baseline_states=None,seconds=60,seed=1,mesh_bounds=None,do_trajopt=False):
    a,b=np.asarray(a15),np.asarray(b15)
    trials=[]
    if baseline_states is not None and a[12]!=b[12]:
        try:
            candidate=warp_seed(baseline_states,a,b)
        except ValueError as exc:
            trials.append(dict(reason=str(exc)))
        else:
            SourceBridge,_,interpolate=edges._dependencies()
            fine=SourceBridge(engine)
            path=edges.compress_collinear(candidate)/fine.scale
            if all(fine.edge_scaled(x,y) for x,y in zip(path,path[1:])):
                states=edges._dense(path,fine,interpolate);states[[0,-1]]=[a,b]
                return dict(passed=True,states=states,waypoints_scaled=path,
                            stats=dict(strategy='warped_cross_angle_source_certified',
                                       inherited_source_validation=False,fine_checks=fine.checks,
                                       positioner_profile='Exact original angular profile proposal; fine source checked'))
            trials.append(dict(reason='warped_profile_failed_fine_source',last_failure=fine.last_failure))
    result=edges.connect(engine,a,b,baseline_states=baseline_states,seconds=seconds,
                         seed=seed,mesh_bounds=mesh_bounds,do_trajopt=do_trajopt)
    result['stats']['cross_angle_trials']=trials
    return result
