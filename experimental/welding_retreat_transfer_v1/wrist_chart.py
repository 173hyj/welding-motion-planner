"""Constant source-legal wrist chart proposals; FK/collision still mandatory."""
import numpy as np


def chart_options(q, low, high):
    q,low,high=map(lambda x:np.asarray(x,float),(q,low,high))
    candidates=[]
    if np.all(q[:,5]>0.):candidates.append(('unchanged',q.copy()))
    lifted=q.copy();lifted[:,5]+=2*np.pi
    candidates.append(('q6_plus_360',lifted))
    for sign in (1.,-1.):
        flip=q.copy();flip[:,3]+=sign*np.pi;flip[:,4]*=-1.;flip[:,5]+=np.pi
        candidates.append(('wrist_flip_'+str(int(sign)),flip))
    return [dict(label=label,q=p) for label,p in candidates
            if np.all(p>=low-1e-10) and np.all(p<=high+1e-10) and np.all(p[:,5]>0.)]
