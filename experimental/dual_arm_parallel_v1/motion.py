"""Display-time arm chains with explicit action provenance and exact seams."""
import numpy as np

def section_times(states,tcp=None):
    states=np.asarray(states,float)
    if states.ndim!=2 or states.shape[1]!=7 or not np.isfinite(states).all():
        raise ValueError('finite Nx7 states required')
    if len(states)<2:return np.array([0.])
    dt=np.maximum(np.max(abs(np.diff(states[:,:6],axis=0)),axis=1)/np.deg2rad(30.),abs(np.diff(states[:,6]))/80.)
    if tcp is not None:
        tcp=np.asarray(tcp,float)
        if tcp.shape!=(len(states),3) or not np.isfinite(tcp).all():raise ValueError('finite TCP positions required')
        dt=np.maximum(dt,np.linalg.norm(np.diff(tcp,axis=0),axis=1)/200.)
    return np.r_[0.,np.cumsum(np.maximum(dt,1e-5))]

def new_chain(start):
    return dict(states=[np.array(start,float)],times=[0.],pieces=[])

def append_piece(chain,states,phase,task=None,duration=None,tcp=None):
    states=np.asarray(states,float)
    if len(states)<2:return
    if not np.allclose(chain['states'][-1],states[0],atol=1e-12,rtol=0):raise ValueError('arm chain seam mismatch')
    times=section_times(states,tcp)
    if duration is not None:
        if duration<=0 or not np.isfinite(duration):raise ValueError('positive finite duration required')
        times=np.linspace(0.,float(duration),len(states))
    start=float(chain['times'][-1]);end=start+float(times[-1])
    chain['states'].extend(states[1:]);chain['times'].extend(start+times[1:])
    chain['pieces'].append(dict(start=start,end=end,phase=phase,task=task))

def piece_at(chain,time):
    if time<0 or time>chain['times'][-1]+1e-9:return None
    pieces=chain['pieces']
    if not pieces:return None
    # A timeline offset round trip can land a few ULPs before an exact seam.
    # Snap only sub-nanosecond boundary noise to the next action phase.
    starts=np.asarray([p['start'] for p in pieces],float)
    near=np.flatnonzero(abs(starts-time)<=1e-9)
    if len(near):time=float(starts[near[-1]])
    i=min(len(pieces)-1,int(np.searchsorted(starts,time,side='right'))-1)
    return pieces[max(0,i)]
