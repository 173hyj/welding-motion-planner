"""Finite, original-limit-qualified shoulder/elbow proposals and blocked rays."""
import copy
import itertools
import numpy as np


def park_bank(model, sign):
    if sign not in (-1,1):
        raise ValueError('Exact wrist sign required')
    bank=[]
    for first,second,third in itertools.product(
            (-3.,-2.,-1.,0.,1.,2.,3.), (-45.,-15.,15.,45.,75.), (-90.,-45.,0.,45.,75.)):
        q=[first,np.deg2rad(second),np.deg2rad(third),0.,0.,sign*np.pi/2]
        if model.limits_valid(np.asarray(q)):
            bank.append(q)
    return bank


def blocked_rays(report, limit):
    if not 1 <= limit <= 10:
        raise ValueError('Finite ray budget required')
    candidates=[]
    for case in report['seeds']:
        for trial in case.get('trials',[]):
            if (trial.get('reason')=='full_retreat_blocked' and
                    any(p.get('asset_name')=='other_robot' for p in trial.get('pairs',[]))):
                candidates.append(dict(seed=copy.deepcopy(case['seed']),trial=copy.deepcopy(trial)))
    candidates.sort(key=lambda r: -(r['trial'].get('last_accepted_travel_mm') or 0.))
    seen=set();result=[]
    for case in candidates:
        seed,t=case['seed'],case['trial']
        key=(seed['candidate']['robot_id'],seed.get('positioner_deg'),tuple(seed.get('q_rad',[])),
             t['amplitude_mm'],tuple(t['direction_world']))
        if key not in seen:
            result.append(case);seen.add(key)
        if len(result)>=limit:
            break
    return result
