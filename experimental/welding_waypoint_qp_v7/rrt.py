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


def bidirectional_rrt(start,goal,model,edge_valid,*,seed=20261008,max_iterations=2000,step_rad=np.deg2rad(3.)):
    rng=np.random.default_rng(seed);start=np.asarray(start,float);goal=np.asarray(goal,float)
    if not model.limits_valid(start) or not model.limits_valid(goal) or not edge_valid(start,start) or not edge_valid(goal,goal):
        return None,dict(reason='endpoint_invalid',iterations=0)
    trees=[dict(nodes=[start],parents=[-1]),dict(nodes=[goal],parents=[-1])]
    def extend(tree,target):
        distances=[np.linalg.norm(x-target) for x in tree['nodes']];idx=int(np.argmin(distances));near=tree['nodes'][idx]
        delta=target-near;distance=np.linalg.norm(delta)
        if distance<1e-10:return idx,True
        new=target if distance<=step_rad else near+delta*step_rad/distance
        if not model.limits_valid(new) or not edge_valid(near,new):return None,False
        tree['nodes'].append(new);tree['parents'].append(idx)
        return len(tree['nodes'])-1,distance<=step_rad
    def chain(tree,idx):
        result=[]
        while idx>=0:result.append(tree['nodes'][idx]);idx=tree['parents'][idx]
        return result[::-1]
    for iteration in range(max_iterations):
        side=iteration%2;a=trees[side];b=trees[1-side]
        mode=rng.random()
        if mode<.2:target=b['nodes'][int(rng.integers(len(b['nodes'])))]
        elif mode<.6:
            near=a['nodes'][int(rng.integers(len(a['nodes'])))];target=np.clip(near+rng.normal(size=6)*np.deg2rad(15.),model.bounds[0],model.bounds[1])
        else:target=model.bounds[0]+rng.random(6)*(model.bounds[1]-model.bounds[0])
        ia,_=extend(a,target)
        if ia is None:continue
        for _ in range(100):
            ib,reached=extend(b,a['nodes'][ia])
            if ib is None:break
            if reached:
                pa=chain(a,ia);pb=chain(b,ib)
                result=pa+pb[-2::-1] if side==0 else pb+pa[-2::-1]
                return result,dict(reason='bidirectional_connected',iterations=iteration+1,nodes=sum(len(t['nodes']) for t in trees))
    return None,dict(reason='iteration_limit',iterations=max_iterations,nodes=sum(len(t['nodes']) for t in trees))
