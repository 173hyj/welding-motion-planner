"""Endpoint-preserving folded gateway proposals; every state still needs source checks."""
import itertools
import numpy as np
from mixed_bridge_v52 import ROBOT_IDS,branch_state_valid


def endpoint_folded_gateways(a,b,robot_ids,parks):
    if robot_ids!=ROBOT_IDS or set(parks)!=set(robot_ids):
        raise ValueError('Exact source identities required')
    a,b=np.asarray(a),np.asarray(b)
    if not branch_state_valid(a,robot_ids) or not branch_state_valid(b,robot_ids):
        raise ValueError('Signed physical endpoints required')
    arms=[[np.asarray(parks[rid]),a[6*i:6*i+6],b[6*i:6*i+6],
           np.array([-3.,-np.pi/12,0.,0.,0.,(-1 if i==0 else 1)*np.pi/2])]
          for i,rid in enumerate(robot_ids)]
    bank=[];seen=set()
    for q0,q1 in itertools.product(*arms):
        for first0,first1 in ((q0[0],q1[0]),(-3.,3.),(-3.,1.),(-3.,0.),(-3.,-1.5)):
            for drivers in (np.zeros(2),a[13:],b[13:]):
                full=a.copy();full[:6]=q0;full[6:12]=q1
                full[0]=first0;full[6]=first1;full[13:]=drivers
                key=tuple(full)
                if key not in seen:bank.append(full);seen.add(key)
    return bank
