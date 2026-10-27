"""Existing folded/gateway transfer proposals using physical joint bounds.

Adapted from dual_arm_parallel_v1.transfer; only the selected wrist-sign
restriction is removed. Every positive path retains original fine source gates.
"""
import sys
from pathlib import Path
import numpy as np


def make_transfer(world, bridge, arm, angle_deg, start7, goal7,
                  existing_edges=(), seed=0, *, park7, max_iterations=1200,
                  step_size=.25,enable_gateways=True):
    """Return physical Nx7 waypoints and stats, or (None, failure stats).

    Existing edges are physical Nx7 arrays or dictionaries with physical
    Nx7/Nx15 ``waypoints``. Certification always invokes the original
    SourceBridge implementation, bypassing any coarse search override.
    The caller may densify accepted corners at ``stats['resolutions7']``.
    No IK, other-arm motion, or positioner motion is introduced.
    """
    legacy=Path(__file__).resolve().parents[1]/'welding_retreat_transfer_v1'
    if str(legacy) not in sys.path: sys.path.insert(0,str(legacy))
    from world_bridge_v2 import SourceBridge
    arm=bridge.robot_ids.index(arm) if isinstance(arm,str) else int(arm)
    if arm not in (0,1): raise ValueError('arm must be 0, 1 or source robot id')
    start,goal,park=[np.asarray(s,float) for s in (start7,goal7,park7)]
    if any(s.shape!=(7,) or not np.isfinite(s).all() for s in (start,goal,park)):
        raise ValueError('finite physical seven-axis endpoints and park required')
    if not np.isfinite(angle_deg) or step_size<=0 or max_iterations<0:
        raise ValueError('finite angle, positive step and bounded iterations required')
    axes=np.r_[np.arange(arm*6,arm*6+6),13+arm]
    other=1-arm
    base=np.zeros(15);base[6*other:6*other+6]=park[:6];base[13+other]=park[6]
    base[12]=np.deg2rad(angle_deg)
    scale=np.asarray(bridge.scale)[axes]
    def full(q):
        s=base.copy();s[axes]=q;return s
    low=np.asarray(bridge.low)[axes]/scale;high=np.asarray(bridge.high)[axes]/scale
    resolutions=np.asarray(bridge.resolutions)[axes]
    started_checks=bridge.checks
    stats=dict(fine_certified=False,method=None,attempts=[],seed=int(seed),
               resolutions7=resolutions.tolist(),max_iterations=int(max_iterations))
    def certify(path,method):
        path=np.asarray(path,float)
        if path.ndim!=2 or path.shape[1]!=7 or not np.isfinite(path).all():return False
        if not np.array_equal(path[0],start) or not np.array_equal(path[-1],goal):return False
        if np.any(path/scale<low-1e-10) or np.any(path/scale>high+1e-10):return False
        scaled=[full(q)/bridge.scale for q in path]
        # Proposal rejection stays coarse; every positive path is still
        # accepted only by the original fine implementation below.
        if len(path)>1 and not all(bridge.edge_scaled(a,b) for a,b in zip(scaled,scaled[1:])):
            stats['attempts'].append(dict(method=method,waypoints=len(path),fine_certified=False,coarse_rejected=True))
            return False
        if hasattr(bridge,'fine_prefill'):bridge.fine_prefill(scaled)
        passed=bridge.valid_scaled(scaled[0]) if len(path)==1 else all(
            SourceBridge.edge_scaled(bridge,a,b) for a,b in zip(scaled,scaled[1:]))
        stats['attempts'].append(dict(method=method,waypoints=len(path),fine_certified=bool(passed)))
        return passed
    def finish(path,method):
        stats.update(method=method,fine_certified=path is not None,
                     source_checks=bridge.checks-started_checks)
        return path,stats
    # Reuse only a genuinely matching fixed-angle, fixed-other-arm route.
    for index,edge in enumerate(existing_edges):
        path=np.asarray(edge['waypoints'] if isinstance(edge,dict) else edge,float)
        if path.ndim!=2 or not len(path) or path.shape[1] not in (7,15):continue
        if path.shape[1]==15:
            fixed=np.setdiff1d(np.arange(15),axes)
            if not np.allclose(path[:,fixed],base[fixed],rtol=0,atol=1e-12):continue
            path=path[:,axes]
        if np.allclose(path[0],goal,rtol=0,atol=1e-12) and np.allclose(path[-1],start,rtol=0,atol=1e-12):path=path[::-1].copy()
        if not (np.allclose(path[0],start,rtol=0,atol=1e-12) and np.allclose(path[-1],goal,rtol=0,atol=1e-12)):continue
        path=path.copy();path[0]=start;path[-1]=goal
        if certify(path,'existing_edge'):
            stats['existing_edge_index']=index
            return finish(path,'existing_edge')
    direct=np.array([start,goal])
    if certify(direct,'direct'):return finish(direct,'direct')
    # Existing parks and route gateway states are transfer waypoints, not new
    # weld IK solutions. Try gun closure separately from robot motion.
    closed_start=start.copy();closed_start[6]=0.
    closed_goal=goal.copy();closed_goal[6]=0.
    proposals=[('close_move_open',np.array([start,closed_start,closed_goal,goal]))]
    folded=np.array([-3.,-np.pi/12,0.,0.,0.,(-1. if arm==0 else 1.)*np.pi/2,0.])
    proposals.append(('existing_folded_park',np.array([start,closed_start,folded,closed_goal,goal])))
    for template in (folded,np.r_[park[:5],(-1. if arm==0 else 1.)*np.pi/2,0.]):
        entry=template.copy();entry[0]=start[0]
        exit=template.copy();exit[0]=goal[0]
        proposals.append(('endpoint_fold_rotate',np.array([start,closed_start,entry,exit,closed_goal,goal])))
        # A separate wrist change can avoid sweeping the long gun during folding.
        entry_keep=entry.copy();entry_keep[5]=start[5]
        exit_keep=exit.copy();exit_keep[5]=goal[5]
        proposals.append(('endpoint_fold_then_wrist',np.array([start,closed_start,entry_keep,entry,exit,exit_keep,closed_goal,goal])))
    fixed=np.setdiff1d(np.arange(15),axes)
    for edge in existing_edges:
        if not isinstance(edge,dict):continue
        waypoints=np.asarray(edge['waypoints'],float)
        if waypoints.ndim!=2 or waypoints.shape[1]!=15:continue
        matches=np.all(np.isclose(waypoints[:,fixed],base[fixed],rtol=0,atol=1e-12),axis=1)
        # Retain a source prefix/suffix when it begins at our endpoint, even
        # when the old edge later rotates the shared positioner.
        for oriented in (waypoints,waypoints[::-1]):
            if not np.allclose(oriented[0,axes],start,atol=1e-12,rtol=0):continue
            n=0
            while n<len(oriented) and np.allclose(oriented[n,fixed],base[fixed],atol=1e-12,rtol=0):n+=1
            if n>1:proposals.append(('existing_route_prefix',np.vstack([oriented[:n,axes],goal])))
        for gateway in waypoints[matches][:,axes]:
            proposals.append(('existing_route_gateway',np.array([start,closed_start,gateway,closed_goal,goal])))
    for method,path in proposals if enable_gateways else ():
        keep=np.r_[True,np.max(abs(np.diff(path,axis=0)),axis=1)>1e-12]
        path=path[keep];path[0]=start;path[-1]=goal
        if certify(path,method):return finish(path,method)
    a,b=start/scale,goal/scale
    def valid(q):
        return bool(np.all(q>=low) and np.all(q<=high) and bridge.valid_scaled(full(q*scale)/bridge.scale))
    if not valid(a) or not valid(b):return finish(None,'invalid_endpoint')
    coarse=np.r_[np.full(6,np.deg2rad(1.)),2.]/scale
    def edge_valid(x,y):
        count=max(1,int(np.ceil(np.max(abs(y-x)/coarse))))
        return all(valid(x+(y-x)*u) for u in np.linspace(0,1,count+1))
    from state_space import connect_rrt
    # Source bidirectional connector uses bound-normalized distances and local
    # samples, avoiding repeated long connections from a single random tree.
    for attempt in range(3 if max_iterations else 0):
        path,search=connect_rrt(a,b,low,high,edge_valid,seed=seed+attempt,
            max_iterations=max(1,int(max_iterations)//3),step=min(.06,step_size))
        stats.setdefault('rrt_attempts',[]).append(search)
        if path is not None:
            path=np.asarray(path)*scale;path[0]=start;path[-1]=goal
            if certify(path,'rrt'):
                stats['iterations']=search['iterations']
                return finish(path,'rrt')
    stats['iterations']=int(max_iterations)
    return finish(None,'search_exhausted')
