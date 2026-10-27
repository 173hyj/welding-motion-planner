"""Compact exact-source state playback helpers."""
import numpy as np


def append_states(data, section):
    if not len(section):raise ValueError('Empty motion section')
    if data:
        if not np.allclose(data[-1],section[0],atol=1e-12,rtol=0):
            raise ValueError('Full15axis seam mismatch')
        section=section[1:]
    data.extend(np.asarray(s,dtype=float) for s in section)


def local_states(path, robot_ids):
    result=[]
    for frame in path['frames']:
        state=np.zeros(15)
        state[12]=np.deg2rad(path['positioner_deg'])
        for i,rid in enumerate(robot_ids):
            state[6*i:6*i+6]=frame['q_rad'] if rid==path['robot_id'] else path['other_q_rad']
            state[13+i]=frame['driver_mm'] if rid==path['robot_id'] else 0.
        result.append(state)
    return result


def interpolate_state(times,states,time):
    i=max(0,min(len(times)-1,int(np.searchsorted(times,time,side='right'))-1))
    j=min(i+1,len(times)-1)
    alpha=0. if i==j else float(np.clip((time-times[i])/(times[j]-times[i]),0.,1.))
    return states[i]+alpha*(states[j]-states[i])
