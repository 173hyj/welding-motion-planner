"""Local exact-mesh proximity proxies and central-difference joint gradients."""
import sys
from pathlib import Path
import fcl
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
from variable_world import VariableToolWorld
from run_arm import qualify_full_pairs
from r2_safe_park import additional_self_checks
from qp_step import polygon_signed_distance


def pair_objects(engine,pair):
    rid=pair['checked_robot_id'];a=engine.world.robots[rid][pair['link']]
    if pair['asset_name'] in ('self','additional_robot_self'):
        b=engine.world.robots[rid][pair['asset_id']]
    elif pair['asset_name']=='other_robot':
        other,key=pair['asset_id'].split(':',1);b=engine.world.robots[other][key]
    else:
        b=next(obj for oid,_,obj in engine.world.static if oid==pair['asset_id'])
    return a,b


def mesh_proxy(a,b):
    result=fcl.CollisionResult()
    fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=100,enable_contact=True),result)
    if result.is_collision:
        depth=max((float(c.penetration_depth) for c in result.contacts),default=0.)
        return -max(depth,1e-8)
    return float(fcl.distance(a,b,fcl.DistanceRequest(),fcl.DistanceResult()))


def collect_pairs(engine,rid,q,other,park,driver,weld,phase,context,margin_mm=2.):
    world=engine.world;world.update(other,park,0.);world.update(rid,q,driver)
    tool=engine.state(rid,world.models[rid].tcp(q),driver,weld,world.static,phase,context)
    pairs=[]
    for checked,state,d in ((rid,q,driver),(other,park,0.)):
        raw=world.check(checked,state,margin_mm=margin_mm,max_pairs=sys.maxsize,gun_driver_mm=d)
        raw+=additional_self_checks(world,checked)
        pairs.extend(dict(p,checked_robot_id=checked) for p in raw)
    pairs=qualify_full_pairs(pairs,rid,weld['product_id'],tool,phase)
    unique={tuple(p.get(k) for k in ('checked_robot_id','link','asset_id')):p for p in pairs}
    return list(unique.values())


def proximity_linearization(engine,rid,q,other,park,driver,pairs,h=1e-4):
    world=engine.world;q=np.asarray(q);world.update(other,park,0.)
    def values(state):
        world.update(rid,state,driver)
        return np.array([mesh_proxy(*pair_objects(engine,p)) for p in pairs])
    initial=values(q);gradient=np.zeros((len(pairs),6))
    for j in range(6):
        delta=np.zeros(6);delta[j]=h
        gradient[:,j]=(values(q+delta)-values(q-delta))/(2*h)
    world.update(rid,q,driver)
    return initial,gradient


def polygon_linearization(model,q,h=1e-5):
    rows=[]
    for i,(leading,polygon) in model.polygons.items():
        point=np.array([q[leading],q[i]])
        distance=polygon_signed_distance(point,polygon);gradient=np.zeros(6)
        for local,j in enumerate((leading,i)):
            delta=np.zeros(2);delta[local]=h
            gradient[j]=(polygon_signed_distance(point+delta,polygon)-polygon_signed_distance(point-delta,polygon))/(2*h)
        rows.append(dict(dependent_joint=i,leading_joint=leading,signed_distance_rad=distance,gradient=gradient))
    return rows
