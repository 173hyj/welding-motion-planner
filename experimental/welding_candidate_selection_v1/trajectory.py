"""Physical raw-axis metrics, exact seams and conservative adoption gates."""
import numpy as np


def combine(prefix,pieces):
    result={k:[np.asarray(v)] for k,v in prefix.items() if k not in ('time','times')}
    clock=[np.asarray(prefix.get('time',prefix.get('times')),float)]
    if len(clock[0])!=len(prefix['states']):raise ValueError('Prefix time length')
    last=np.asarray(prefix['states'][-1]);offset=float(clock[0][-1])
    for piece in pieces:
        states=np.asarray(piece['states']);times=np.asarray(piece['time'])
        if len(states)!=len(times) or not len(states) or times[0]!=0 or np.any(np.diff(times)<=0):
            raise ValueError('Piece needs zero-based ascending time')
        if not np.allclose(last,states[0],atol=1e-10,rtol=0):raise ValueError('Raw full15 seam mismatch')
        for k in result:
            if k not in piece or len(piece[k])!=len(states):raise ValueError('Missing piece array: '+k)
            result[k].append(np.asarray(piece[k])[1:])
        clock.append(offset+times[1:]);offset+=float(times[-1]);last=states[-1]
    result={k:np.concatenate(v) for k,v in result.items()}
    joined=np.concatenate(clock)
    if not np.isfinite(joined).all() or np.any(np.diff(joined)<=0):raise ValueError('Invalid combined clock')
    for key in ('time','times'):
        if key in prefix:result[key]=joined.copy()
    return result


def motion_metrics(states,times):
    states=np.asarray(states,float);times=np.asarray(times,float)
    if states.ndim!=2 or states.shape[1]!=15 or len(states)!=len(times) or not np.isfinite(states).all():
        raise ValueError('Finite Nx15 states and matching time required')
    delta=abs(np.diff(states,axis=0));angles=np.rad2deg(states[:,12]);dangle=abs(np.diff(angles))
    moving=dangle>1e-7
    starts=moving&~np.r_[False,moving[:-1]]
    return dict(duration_s=float(times[-1]-times[0]),
                joint_travel_deg=float(np.rad2deg(delta[:,:12]).sum()),
                wrist_travel_deg=float(np.rad2deg(delta[:,[3,4,5,9,10,11]]).sum()),
                rotation_episodes=int(starts.sum()),total_positioner_travel_deg=float(dangle.sum()))


def acceptance(before,after):
    reasons=[]
    if after['duration_s']>before['duration_s']+1e-7:reasons.append('display_time_regression')
    if after['joint_travel_deg']>before['joint_travel_deg']+1e-6:reasons.append('joint_motion_regression')
    if after['wrist_travel_deg']>before['wrist_travel_deg']+1e-6:reasons.append('wrist_motion_regression')
    if min(before['joint_travel_deg']-after['joint_travel_deg'],
           before['wrist_travel_deg']-after['wrist_travel_deg'])<=1e-3:
        reasons.append('no_strict_joint_and_wrist_improvement')
    if after['rotation_episodes']>before['rotation_episodes']:reasons.append('extra_positioner_rotation')
    if after['total_positioner_travel_deg']>before['total_positioner_travel_deg']+1e-6:reasons.append('extra_positioner_travel')
    return dict(accepted=not reasons,reasons=reasons)
