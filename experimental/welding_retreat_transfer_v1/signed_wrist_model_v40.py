"""Consistent per-robot wrist search domains; source FK and limits stay authoritative."""
import numpy as np


class SignedWristModel:
    def __init__(self, source, sign):
        if sign not in (-1, 1):
            raise ValueError('Wrist sign must be -1 or +1')
        self.source = source
        self.sign = sign
        self.bounds = np.array(source.bounds, copy=True)
        if sign > 0:
            self.bounds[0,5] = max(self.bounds[0,5],1e-6)
        else:
            self.bounds[1,5] = min(self.bounds[1,5],-1e-6)

    def __getattr__(self,name):
        return getattr(self.source,name)

    def limits_valid(self,q):
        q = np.asarray(q)
        return (q.shape == (6,) and np.isfinite(q).all() and self.sign*q[5] >= 1e-6
                and self.source.limits_valid(q))


def branch_parks(models,signs):
    if set(models) != set(signs):
        raise ValueError('Exactly one wrist domain required for each source robot')
    parks = {}
    for rid,model in models.items():
        q = np.array([-3.,-.8,.5,0.,0.,signs[rid]*np.pi/2])
        if not SignedWristModel(model,signs[rid]).limits_valid(q):
            raise ValueError('Selected branch park violates source limits')
        parks[rid] = q.tolist()
    return parks


def path_compatibility(path,signs,parks):
    rid,other = path['robot_id'],path['other_robot_id']
    states = [path['q_start_rad']] + [f['q_rad'] for f in path['frames']]
    active = bool(path['frames']) and all(signs[rid]*q[5] >= 1e-6 for q in states)
    parked = np.array_equal(path['other_q_rad'],parks[other])
    return active,bool(parked)
