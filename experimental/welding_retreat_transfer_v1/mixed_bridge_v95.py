"""Expanded first-axis gateways motivated by full-source collision diagnostics."""
import numpy as np
from conservative_search_v30 import SearchBridge
from signed_wrist_model_v40 import branch_parks
from shared_gateway import connect_shared
from endpoint_folded_gateway_v94 import endpoint_folded_gateways

ROBOT_IDS=['F46171CC-24BD-45DD-82E8-21711C110EE9','F7690943-8A45-40E4-9AD3-F92856F982FA']
SIGNS={ROBOT_IDS[0]:-1,ROBOT_IDS[1]:1}


def branch_bounds(low,high,robot_ids):
    if robot_ids!=ROBOT_IDS:raise ValueError('Exact source robot order required')
    low,high=np.array(low,copy=True),np.array(high,copy=True)
    if low.shape!=(15,) or high.shape!=(15,):raise ValueError('15-axis bounds required')
    high[5]=min(high[5],-1e-6);low[11]=max(low[11],1e-6)
    if np.any(low>high):raise ValueError('Selected branch outside source bounds')
    return low,high


def branch_state_valid(state,robot_ids):
    if robot_ids!=ROBOT_IDS:raise ValueError('Exact source robot order required')
    state=np.asarray(state)
    return bool(state.shape==(15,) and np.isfinite(state).all() and state[5]<=-1e-6 and state[11]>=1e-6)


class SignedSearchBridge(SearchBridge):
    def __init__(self,engine):
        super().__init__(engine)
        self.low,self.high=branch_bounds(self.low,self.high,self.robot_ids)

    def probe(self,state):
        if not branch_state_valid(state,self.robot_ids):
            return dict(passed=False,reason='outside_selected_wrist_search_domain')
        # Source bounds, coupling and all mesh collision checks remain authoritative.
        return super().probe(state)


def gateway_states(a,b,robot_ids,parks):
    if robot_ids!=ROBOT_IDS or set(parks)!=set(robot_ids):raise ValueError('Exact source identities required')
    a,b=np.asarray(a),np.asarray(b)
    if not branch_state_valid(a,robot_ids) or not branch_state_valid(b,robot_ids):
        raise ValueError('Compatible full-state endpoints required')
    base=a.copy()
    for i,rid in enumerate(robot_ids):base[6*i:6*i+6]=parks[rid]
    configurations=[base]
    for q0 in (-3.,-2.,-1.,0.):
        for q1 in (0.,1.,2.,3.):
            q=base.copy();q[0]=q0;q[6]=q1;configurations.append(q)
    for q1 in (-2.5,-2.,-1.5):
        for axes in ((0,),(6,),(0,6)):
            q=base.copy();q[list(axes)]=q1;configurations.append(q)
    # Endpoint arm configurations are additional proposals, never static certificates.
    for endpoint in (a,b):
        for i in (0,1):
            q=base.copy();q[6*i:6*i+6]=endpoint[6*i:6*i+6];configurations.append(q)
    bank=[];seen=set()
    for q in configurations:
        for drivers in (np.zeros(2),a[13:],b[13:]):
            full=q.copy();full[13:]=drivers;key=tuple(full)
            if key not in seen:bank.append(full);seen.add(key)
    return bank


def connect_fresh_park_options(bridge,a,b,seed,iterations):
    parks=branch_parks(bridge.world.models,SIGNS)
    actual=endpoint_folded_gateways(a*bridge.scale,b*bridge.scale,bridge.robot_ids,parks)
    bank=[p/bridge.scale for p in actual]
    bank.sort(key=lambda q:np.linalg.norm(q-a)+np.linalg.norm(q-b))
    attempts=[]
    for budget,gateways in ((0,bank),(min(iterations,300),bank[:8])):
        for gateway in gateways:
            for arm_order in ((0,1),(1,0)):
                path,stages=connect_shared(bridge,a,b,gateway,arm_order,
                                           seed+100*len(attempts),budget)
                attempts.append(dict(gateway_state=(gateway*bridge.scale).tolist(),
                    source='fresh_live_source_proposal_not_static_certificate',arm_order=arm_order,
                    rrt_stage_budget=budget,stages=stages,passed=path is not None))
                if path is not None:return path,attempts
    return None,attempts
