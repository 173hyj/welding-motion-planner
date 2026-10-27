"""Fixed-order scheduling and physical-unit dual-arm timeline utilities."""
from functools import lru_cache
import numpy as np


def align_batches(tasks_by_arm):
    """Align consecutive same-angle blocks, minimizing barriers then travel."""
    if len(tasks_by_arm) != 2:
        raise ValueError('exactly two arms required')
    blocks = []
    for tasks in tasks_by_arm:
        groups = []
        for task in tasks:
            if groups and groups[-1][0] == task['angle']:
                groups[-1][1].append(task)
            else:
                groups.append((task['angle'], [task]))
        blocks.append(groups)

    @lru_cache(None)
    def solve(i, j, previous):
        if i == len(blocks[0]) and j == len(blocks[1]):
            return (0, 0., (), ())
        options = []
        angles = sorted({blocks[a][k][0] for a,k in [(0,i),(1,j)] if k < len(blocks[a])})
        for angle in angles:
            take0 = i < len(blocks[0]) and blocks[0][i][0] == angle
            take1 = j < len(blocks[1]) and blocks[1][j][0] == angle
            rest = solve(i+take0,j+take1,angle)
            cost = 0. if previous is None else abs(float(angle)-float(previous))
            stage = dict(angle=angle,tasks=[blocks[0][i][1] if take0 else [],blocks[1][j][1] if take1 else []])
            options.append((1+rest[0],cost+rest[1],(angle,)+rest[2],(stage,)+rest[3]))
        return min(options,key=lambda x:x[:3])
    return list(solve(0,0,None)[3])


def _knots(times, states):
    times = np.asarray(times,dtype=float)
    states = np.asarray(states,dtype=float)
    if times.ndim != 1 or states.ndim != 2 or len(times) != len(states) or not len(times):
        raise ValueError('nonempty matching times and states required')
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(states)) or np.any(np.diff(times)<0):
        raise ValueError('finite ascending samples required')
    duplicate = np.diff(times)==0
    if np.any(np.any(np.abs(np.diff(states,axis=0)[duplicate])>1e-12,axis=1)):
        raise ValueError('inconsistent duplicate timestamp states')
    keep = np.r_[True,np.diff(times)>0]
    return times[keep],states[keep]


def state_at(times, states, t):
    """Piecewise-linear state, holding the true first/last sample outside time."""
    times,states = _knots(times,states)
    return np.array([np.interp(t,times,states[:,a]) for a in range(states.shape[1])])


def merge_samples(path0, path1, offsets=(0.,0.), park_poses=None, angle_deg=0., resolutions=None):
    """Return timestamp union and dense physical 15-axis states.

    Paths are (times, Nx7 states) or dictionaries with times/states. Joint
    angles and positioner are radians; drivers remain millimeters. Before
    delayed starts hold each chain's first state (optional park override).
    """
    paths=[]
    for path in (path0,path1):
        times,states = (path['times'],path['states']) if isinstance(path,dict) else path
        times,states = _knots(times,states)
        if states.shape[1] != 7 or times[0] != 0:
            raise ValueError('chains require local time zero and seven axes')
        paths.append((times,states))
    offsets=np.asarray(offsets,dtype=float)
    if offsets.shape != (2,) or np.any(offsets<0) or not np.all(np.isfinite(offsets)):
        raise ValueError('two finite nonnegative offsets required')
    knots=np.unique(np.r_[0.,paths[0][0]+offsets[0],paths[1][0]+offsets[1]])
    def sample(t):
        arms=[]
        for a,(times,states) in enumerate(paths):
            arms.append(np.asarray(park_poses[a]) if park_poses is not None and t<offsets[a]
                        else np.array([np.interp(t-offsets[a],times,states[:,axis]) for axis in range(7)]))
        return np.r_[arms[0][:6],arms[1][:6],np.deg2rad(angle_deg),arms[0][6],arms[1][6]]
    limits=np.r_[np.full(12,np.deg2rad(.05)),np.deg2rad(.25),.5,.5] if resolutions is None else np.asarray(resolutions,dtype=float)
    if limits.shape != (15,) or np.any(limits<=0):
        raise ValueError('15 positive axis resolutions required')
    dense=[knots[0]]
    for start,end in zip(knots[:-1],knots[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(sample(end)-sample(start))/limits))))
        dense.extend(np.linspace(start,end,n+1)[1:])
    times=np.asarray(dense)
    return times,np.asarray([sample(t) for t in times])


def choose_offset(duration0, duration1, validate, step=.5, candidates=None):
    """Validate shifts in either direction, including both serial fallbacks.

    Return the safe candidate with minimum makespan, or None. The callback
    receives an offset pair and returns bool or (bool, diagnostic).
    """
    durations=np.asarray([duration0,duration1],dtype=float)
    if np.any(durations<0) or not np.all(np.isfinite(durations)) or step<=0:
        raise ValueError('finite nonnegative durations and positive step required')
    if candidates is None:
        candidates=[(0.,0.)]+[(float(x),0.) for x in np.arange(step,duration1,step)]+[(0.,float(x)) for x in np.arange(step,duration0,step)]
    candidates=list(candidates)+[(0.,float(duration0)),(float(duration1),0.)]
    tried=[]; safe=[]; seen=set()
    for candidate in candidates:
        offsets=tuple(map(float,candidate))
        if len(offsets)!=2 or min(offsets)<0 or not np.all(np.isfinite(offsets)):
            raise ValueError('invalid candidate offsets')
        if offsets in seen:
            continue
        seen.add(offsets)
        verdict=validate(offsets)
        accepted=bool(verdict[0]) if isinstance(verdict,tuple) else bool(verdict)
        record=dict(offsets=offsets,makespan=max(offsets[0]+duration0,offsets[1]+duration1),safe=accepted)
        if isinstance(verdict,tuple):
            record['validation']=verdict[1]
        tried.append(record)
        if accepted:
            safe.append(record)
    if not safe:
        return None
    best=min(safe,key=lambda r:(r['makespan'],sum(r['offsets']),r['offsets']))
    return dict(best,tried=tried)
