"""Exact full-candidate, directed grouped Hamilton open paths.

Costs are nonnegative lexicographic vectors: duration, joint travel, wrist travel.
No certification is implied by a numerical solution. Infinite first component
forbids an edge. Every candidate supplied participates, with no beam or top-k.
"""
import itertools
import numpy as np


def _array(value,shape,name):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or np.isnan(a).any() or (a<0).any():
        raise ValueError(f'{name}: expected nonnegative {shape} array, allowing infinity')
    if a.ndim and a.shape[-1] and np.any(np.isfinite(a[...,0]) & ~np.isfinite(a).all(axis=-1)):
        raise ValueError(f'{name}: finite primary cost requires finite secondary costs')
    return a


def _best_rows(a):
    return np.lexsort(tuple(a[...,k] for k in reversed(range(a.shape[-1]))),axis=0)[0]


def grouped_open_path(node_costs,edge_costs,start_costs=None,end_costs=None):
    """Exact DP over (visited task mask,last task,last candidate).

    node_costs[t] is N_t x K. Directed edge_costs[a,b] is N_a x N_b x K;
    omitted edges are forbidden. Endpoint arrays match node arrays. Returns an
    order, candidate indices in that order, and lexicographic cost, or None.
    Ties are deterministic; no lexicographic path tie guarantee is made.
    Complexity O(2**T T**2 C**2 K), memory O(2**T T C K).
    """
    if not node_costs:return dict(order=[],indices=[],cost=[])
    raw=[np.asarray(x,dtype=float) for x in node_costs]
    if any(x.ndim!=2 for x in raw):raise ValueError('node costs must be matrices')
    k=raw[0].shape[1]
    if k<1:raise ValueError('at least one objective required')
    sizes=[len(x) for x in raw];n=len(sizes)
    nodes=[_array(x,(s,k),'node') for x,s in zip(raw,sizes)]
    if not all(sizes):return None
    if start_costs is not None and len(start_costs)!=n:raise ValueError('one start array per task required')
    if end_costs is not None and len(end_costs)!=n:raise ValueError('one end array per task required')
    starts=[np.zeros_like(x) for x in nodes] if start_costs is None else [_array(x,(s,k),'start') for x,s in zip(start_costs,sizes)]
    ends=[np.zeros_like(x) for x in nodes] if end_costs is None else [_array(x,(s,k),'end') for x,s in zip(end_costs,sizes)]
    if len(starts)!=n or len(ends)!=n:raise ValueError('one endpoint array per task required')
    edges={(a,b):_array(x,(sizes[a],sizes[b],k),'edge') for (a,b),x in edge_costs.items()}
    costs={};parents={}
    for t in range(n):
        costs[1<<t,t]=starts[t]+nodes[t]
        parents[1<<t,t]=(np.full(sizes[t],-1),np.full(sizes[t],-1))
    for mask in range(1,1<<n):
        for last in range(n):
            if (mask,last) not in costs:continue
            source=costs[mask,last]
            for nxt in range(n):
                if mask&(1<<nxt) or (last,nxt) not in edges:continue
                proposals=source[:,None,:]+edges[last,nxt]+nodes[nxt][None,:,:]
                take=_best_rows(proposals)
                candidate=proposals[take,np.arange(sizes[nxt])]
                key=(mask|(1<<nxt),nxt)
                if key not in costs:
                    costs[key]=candidate.copy()
                    parents[key]=(np.full(sizes[nxt],last),take.copy())
                else:
                    current=costs[key]
                    # first unequal component decides, so duration dominates continuity.
                    unequal=candidate!=current
                    first=np.argmax(unequal,axis=1)
                    improve=unequal.any(axis=1)&(candidate[np.arange(len(first)),first]<current[np.arange(len(first)),first])
                    current[improve]=candidate[improve]
                    parents[key][0][improve]=last
                    parents[key][1][improve]=take[improve]
    full=(1<<n)-1;options=[]
    for t in range(n):
        if (full,t) not in costs:continue
        totals=costs[full,t]+ends[t]
        i=int(_best_rows(totals[:,None,:])[0])
        if np.isfinite(totals[i]).all():options.append((tuple(totals[i]),t,i))
    if not options:return None
    total,t,i=min(options);mask=full;order=[];indices=[]
    while t>=0:
        order.append(t);indices.append(i)
        pt,pi=parents[mask,t];old_t=t;t,i=int(pt[i]),int(pi[i]);mask^=1<<old_t
    return dict(order=order[::-1],indices=indices[::-1],cost=list(total))


def angle_routes(angles,start_angle,final_angle=None):
    """All single-batch angle orders ranked by rotations then physical travel.

    Angles are degrees in the physical open 0..120 interval; no circular wrap.
    The initial state angle and optional final angle count as motion endpoints.
    """
    values=list(angles)
    if len(set(values))!=len(values):raise ValueError('unique angles required')
    if any(not 0<=a<=120 for a in values+[start_angle]+([] if final_angle is None else [final_angle])):raise ValueError('angle outside open physical interval')
    result=[]
    for order in itertools.permutations(values):
        chain=[start_angle]+list(order)+([] if final_angle is None else [final_angle])
        result.append(dict(angles=list(order),rotation_episodes=sum(a!=b for a,b in zip(chain,chain[1:])),travel_deg=sum(abs(a-b) for a,b in zip(chain,chain[1:]))))
    return sorted(result,key=lambda r:(r['rotation_episodes'],r['travel_deg'],r['angles']))
