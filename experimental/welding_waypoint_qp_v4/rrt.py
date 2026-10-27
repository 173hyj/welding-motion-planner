"""Small joint-space RRT used only after a certified pre-collision waypoint."""
import math
import numpy as np


def choose_pre_collision(failure_mm,margin_mm=2.):return max(0.,float(failure_mm)-float(margin_mm))

def edge_samples(start,end,max_joint_step_rad=np.deg2rad(.5)):
    start=np.asarray(start,float);end=np.asarray(end,float);count=max(1,int(np.ceil(np.max(abs(end-start))/max_joint_step_rad)))
    return [start+(end-start)*u for u in np.linspace(0.,1.,count+1)]

def path_length(path):
    return float(sum(np.linalg.norm(np.asarray(b)-np.asarray(a)) for a,b in zip(path,path[1:])))

def rrt_connect(start,goal,model,edge_valid,*,seed=20261008,max_iterations=2500,step_rad=np.deg2rad(5.),goal_bias=.2):
    rng=np.random.default_rng(seed);start=np.asarray(start,float);goal=np.asarray(goal,float)
    if not model.limits_valid(start) or not model.limits_valid(goal):return None,dict(reason='endpoint_limits_invalid',iterations=0)
    if not edge_valid(start,start) or not edge_valid(goal,goal):return None,dict(reason='endpoint_collision',iterations=0)
    nodes=[start];parents=[-1]
    def extend(q):
        distances=[np.linalg.norm(x-q) for x in nodes];i=int(np.argmin(distances));near=nodes[i];delta=q-near;norm=np.linalg.norm(delta)
        new=q if norm<=step_rad else near+delta/norm*step_rad
        if not model.limits_valid(new) or not edge_valid(near,new):return None
        nodes.append(new);parents.append(i);return len(nodes)-1
    for iteration in range(max_iterations):
        target=goal if rng.random()<goal_bias else model.bounds[0]+rng.random(6)*(model.bounds[1]-model.bounds[0])
        idx=extend(target)
        if idx is None:continue
        if np.linalg.norm(nodes[idx]-goal)<=step_rad and edge_valid(nodes[idx],goal):
            nodes.append(goal);parents.append(idx);chain=[];j=len(nodes)-1
            while j>=0:chain.append(nodes[j]);j=parents[j]
            return list(reversed(chain)),dict(reason='goal_reached',iterations=iteration+1,nodes=len(nodes))
    return None,dict(reason='iteration_limit',iterations=max_iterations,nodes=len(nodes))
