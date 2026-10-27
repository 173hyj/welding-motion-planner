"""Actual merged-sample speed and START-phase weld dwell receipt."""
import numpy as np

def audit_timing(data,task_arms,tolerance=1e-7):
    states=np.asarray(data['states']);clock=np.asarray(data.get('time',data.get('times')));tcp=np.asarray(data['tcp'])
    dt=np.diff(clock)
    if len(states)<2 or states.shape!=(len(clock),15) or tcp.shape!=(len(clock),2,3) or not np.isfinite(clock).all() or not np.isfinite(states).all() or not np.isfinite(tcp).all() or np.any(dt<=0):
        return dict(passed=False,reasons=['invalid_finite_monotone_sample_arrays'])
    delta=abs(np.diff(states,axis=0));speed=dict(joint_deg_s=float(np.rad2deg(delta[:,:12]/dt[:,None]).max()),positioner_deg_s=float(np.rad2deg(delta[:,12]/dt).max()),gun_mm_s=float((delta[:,13:]/dt[:,None]).max()),tcp_mm_s=float((np.linalg.norm(np.diff(tcp,axis=0),axis=2)/dt[:,None]).max()))
    limits=dict(joint_deg_s=30.,positioner_deg_s=15.,gun_mm_s=80.,tcp_mm_s=200.)
    reasons=[k+'_limit_exceeded' for k,value in speed.items() if value>limits[k]+tolerance]
    phases=np.asarray(data['phases']);tasks=np.asarray(data['tasks']);dwells=[]
    for task,arm in task_arms.items():
        mask=(phases[:-1,arm]=='weld')&(tasks[:-1,arm]==task)
        runs=int(np.sum(mask&~np.r_[False,mask[:-1]]));duration=float(dt[mask].sum())
        own=np.r_[np.arange(6*arm,6*arm+6),12,13+arm]
        stationary=bool(not mask.any() or np.max(delta[mask][:,own])<=1e-10)
        passed=runs==1 and abs(duration-.6)<=tolerance and stationary
        dwells.append(dict(task=int(task),arm=int(arm),runs=runs,duration_s=duration,stationary=stationary,passed=passed))
        if not passed:reasons.append('weld_dwell_invalid_'+str(task))
    return dict(passed=not reasons,reasons=reasons,speeds=speed,limits=limits,dwells=dwells,phase_convention='START phase governs [time[i],time[i+1])')
