"""Source-mesh active proximity constraints; optimization proxies are not certificates."""
from itertools import combinations
import numpy as np

INTERFACES={tuple(sorted(p)) for p in [('239','240'),('240','241'),('241','242'),('242','243'),('243','244'),('244','245'),('240','246'),('241','246'),('240','247'),('241','247'),('246','247')]}

def enumerate_pairs(world):
    """Union of CollisionWorld.check and additional_self_checks, deduplicated."""
    pairs={}
    def add(a,b,oa,ob):
        key=tuple(sorted((a,b)))
        pairs.setdefault(key,(a,b,oa,ob))
    for rid,objects in world.robots.items():
        for k,o in objects.items():
            if k=='239': continue
            for sid,_,s in world.static: add((rid,k),('static',sid),o,s)
            for other,obs in world.robots.items():
                if other!=rid:
                    for b,ob in obs.items(): add((rid,k),(other,b),o,ob)
        keys=[str(i) for i in range(240,246)]
        for i,a in enumerate(keys):
            for b in keys[i+2:]:
                if a in objects and b in objects: add((rid,a),(rid,b),objects[a],objects[b])
        guns=[k for k in objects if k.startswith('gun:') or k in ('gun','electrode')]
        for a in keys[:-1]:
            for b in guns:
                if a in objects: add((rid,a),(rid,b),objects[a],objects[b])
        for a,b in combinations([k for k in objects if k.isdigit()],2):
            if {a,b}&{'239','246','247'} and tuple(sorted((a,b))) not in INTERFACES:
                add((rid,a),(rid,b),objects[a],objects[b])
        for a in ('239','246','247'):
            for b in objects:
                if a in objects and b.startswith('gun:'): add((rid,a),(rid,b),objects[a],objects[b])
    return list(pairs.values())

def sample_weights(n):
    return [((i,1.),) for i in range(n)]+[((i,.5),(i+1,.5)) for i in range(n-1)]

def polygon_signed_distance(point,polygon):
    p=np.asarray(point,float); poly=np.asarray(polygon,float); minimum=float('inf'); inside=False
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        edge=b-a; length=float(edge@edge)
        near=a+np.clip((p-a)@edge/length,0,1)*edge if length else a
        minimum=min(minimum,float(np.linalg.norm(p-near)))
        if (a[1]>p[1])!=(b[1]>p[1]) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]: inside=not inside
    return minimum if inside else -minimum

def mesh_proxy(a,b):
    import fcl
    result=fcl.CollisionResult()
    fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=100,enable_contact=True),result)
    if result.is_collision:
        return -max(max((float(c.penetration_depth) for c in result.contacts),default=0.),1e-8)
    return float(fcl.distance(a,b,fcl.DistanceRequest(),fcl.DistanceResult()))

class SourceProximity:
    """Input normalized (N,15) knots; outputs g>=0, J shaped (len(g),N*15).

    Mesh distances and margins use mm/100. Coupled-limit polygon distances
    use radians. Endpoint margins and mesh gradients are zero because endpoints are fixed.
    Captured exact source mesh bounds conservatively reject distant pairs.
    """
    def __init__(self,bridge,activation_mm=5.,margin_mm=1.,include_coupling=True,h=1e-4,mesh_bounds=None):
        if not np.isfinite([activation_mm,margin_mm,h]).all() or not activation_mm>=margin_mm>=0 or h<=0:
            raise ValueError("Require finite activation_mm >= margin_mm >= 0 and h > 0")
        self.bridge=bridge; self.world=bridge.world; self.engine=bridge.engine
        self.activation_mm=activation_mm; self.margin_mm=margin_mm
        self.include_coupling=include_coupling; self.h=h
        self.mesh_bounds={} if mesh_bounds is None else mesh_bounds
        self.pairs=enumerate_pairs(self.world)
        if hasattr(getattr(self.engine,"environment",None),"relative_ids"): self.pairs+=layout_pairs(self.engine.environment)
        environment=getattr(self.engine,"environment",None)
        self.moving_object_ids={id(environment.objects[k]) for k in environment.relative_ids} if hasattr(environment,"relative_ids") else set()
        self.last_rows=[]; self.mesh_queries=0
    def set_state(self,x):
        state=np.asarray(x)*self.bridge.scale
        self.engine.set_angle(float(np.rad2deg(state[12])))
        for i,rid in enumerate(self.bridge.robot_ids):
            self.world.update(rid,state[6*i:6*i+6],float(state[13+i]))
    def _values(self,pairs):
        self.mesh_queries+=len(pairs)
        return np.array([mesh_proxy(a,b) for _,_,a,b in pairs])/100.
    def oracle(self,path_scaled,linearize=True):
        path=np.asarray(path_scaled,float)
        if path.ndim!=2 or path.shape[1]!=15 or not np.isfinite(path).all(): raise ValueError('finite N by 15 normalized path required')
        values=[]; rows=[]; self.last_rows=[]
        try:
            for weights in sample_weights(len(path)):
                x=sum(path[k]*w for k,w in weights); self.set_state(x)
                candidate_pairs=[]
                boxes={}
                for pair in self.pairs:
                    a,b=pair[2:]
                    if id(a) in self.mesh_bounds and id(b) in self.mesh_bounds:
                        for obj in (a,b):
                            if id(obj) not in boxes: boxes[id(obj)]=transformed_aabb(obj,self.mesh_bounds[id(obj)])
                        if aabb_gap(boxes[id(a)],boxes[id(b)])>self.activation_mm+1e-7: continue
                    candidate_pairs.append(pair)
                distances=self._values(candidate_pairs)
                selected=np.flatnonzero(distances<self.activation_mm/100.)
                active=[candidate_pairs[k] for k in selected]
                margin=0. if len(weights)==1 and weights[0][0] in (0,len(path)-1) else self.margin_mm/100.
                local=distances[selected]-margin
                jac=np.zeros((len(active),15))
                if linearize and active and not (len(weights)==1 and weights[0][0] in (0,len(path)-1)):
                    dependencies=[pair_axes(p,self.bridge.robot_ids,self.moving_object_ids) for p in active]
                    for j in range(15):
                        subset=[i for i,axes in enumerate(dependencies) if j in axes]
                        if not subset: continue
                        subpairs=[active[i] for i in subset]
                        d=np.zeros(15); d[j]=self.h
                        self.set_state(x+d); plus=self._values(subpairs)
                        self.set_state(x-d); minus=self._values(subpairs)
                        jac[subset,j]=(plus-minus)/(2*self.h)
                    self.set_state(x)
                for i,p in enumerate(active):
                    values.append(local[i]); row=np.zeros(path.size)
                    if linearize:
                        for k,w in weights: row[15*k:15*k+15]=w*jac[i]
                        rows.append(row)
                    self.last_rows.append(dict(weights=weights,pair=(p[0],p[1]),kind='mesh'))
                if self.include_coupling:
                    for ri,rid in enumerate(self.bridge.robot_ids):
                        for dependent,(leading,polygon) in self.world.models[rid].polygons.items():
                            ids=[6*ri+leading,6*ri+dependent]; point=x[ids]
                            val=polygon_signed_distance(point,polygon)
                            values.append(val); row=np.zeros(path.size)
                            if linearize:
                                for j,axis in enumerate(ids):
                                    d=np.zeros(2);d[j]=self.h
                                    grad=(polygon_signed_distance(point+d,polygon)-polygon_signed_distance(point-d,polygon))/(2*self.h)
                                    for k,w in weights: row[15*k+axis]=w*grad
                                rows.append(row)
                            self.last_rows.append(dict(weights=weights,kind='coupling',robot_id=rid,dependent=dependent))
        finally:
            if len(path): self.set_state(path[-1])
        return np.asarray(values),np.asarray(rows).reshape((-1,path.size)) if linearize else None
    __call__=oracle
    def sparse_feasibility(self,path_scaled):
        path=np.asarray(path_scaled,float)
        return all(self.bridge.probe(sum(path[k]*w for k,w in weights)*self.bridge.scale)['passed'] for weights in sample_weights(len(path)))

def layout_pairs(environment):
    """Exact PositionerWorld moving-versus-fixed exclusions."""
    pairs=[]
    for moving in sorted(environment.relative_ids):
        for sid,_,obj in environment.fixed_env:
            if moving==environment.moving_id and sid in (environment.fixed_id,environment.interface_id): continue
            pairs.append((('static',moving),('static',sid),environment.objects[moving],obj))
    return pairs


def transformed_aabb(obj,bounds):
    """Enclosing world AABB of the exact source mesh's local AABB."""
    center,half=bounds
    rotation=np.asarray(obj.getRotation(),float); translation=np.asarray(obj.getTranslation(),float)
    world_center=rotation@center+translation
    extent=np.abs(rotation)@half
    return world_center-extent,world_center+extent

def aabb_gap(a,b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0.)))

from contextlib import contextmanager

@contextmanager
def capture_mesh_bounds():
    """Capture factory mesh bounds without replacing or modifying geometry.

    Construct the source engine inside this context, then pass the yielded
    dictionary to SourceProximity. The original source factory is restored
    even if engine construction raises. This context is process-global and
    must not overlap concurrent engine construction in another thread.
    """
    import collision_audit
    original=collision_audit.fcl_object; bounds={}
    def captured(mesh,transform):
        obj=original(mesh,transform)
        vertices=np.asarray(mesh.vertices,float)
        if len(vertices):
            lo=vertices.min(axis=0);hi=vertices.max(axis=0)
            bounds[id(obj)]=((lo+hi)/2,(hi-lo)/2)
        return obj
    collision_audit.fcl_object=captured
    try:
        yield bounds
    finally:
        collision_audit.fcl_object=original



def pair_axes(pair,robot_ids,moving_object_ids):
    """Exact transform dependencies, independent of static/layout label names."""
    axes=set()
    for (rid,key),obj in zip(pair[:2],pair[2:]):
        if rid in robot_ids:
            i=robot_ids.index(rid)
            axes.update(range(6*i,6*i+6))
            if key.startswith('gun:') or key in ('gun','electrode'): axes.add(13+i)
        if id(obj) in moving_object_ids: axes.add(12)
    return axes

