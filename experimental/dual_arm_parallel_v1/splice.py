"""Pure splice operations that preserve sealed serial-baseline samples and timing.

Rows label the incoming interval; chain pieces start at its predecessor knot.
Index slices use inclusive end indexes, so adjacent parts share a seam row.
"""
import numpy as np
from .motion import new_chain


def _arrays(data, columns=1):
    times=np.asarray(data['times'] if 'times' in data else data['time'],float).reshape(-1)
    states=np.asarray(data['states'],float)
    phases=np.asarray(data['phases'])
    tasks=np.asarray(data['tasks'])
    if columns==1:
        phases=phases.reshape(-1);tasks=tasks.reshape(-1)
    if not len(times) or states.shape!=(len(times),15) or phases.shape!=((len(times),) if columns==1 else (len(times),2)) or tasks.shape!=phases.shape:
        raise ValueError('matching nonempty baseline arrays required')
    if not np.isfinite(times).all() or not np.isfinite(states).all() or np.any(np.diff(times)<=0):
        raise ValueError('finite states and strictly ascending times required')
    if not np.issubdtype(tasks.dtype,np.number) or not np.isfinite(tasks).all() or np.any(tasks!=tasks.astype(int)):
        raise ValueError('finite integer task labels required')
    return times,states,phases,tasks.astype(int)


def _span_indices(phases,tasks,task,phase):
    mask=(tasks==task)&((phases!='transfer') if phase is None else (phases==phase))
    indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('requested baseline span absent')
    if np.any(np.diff(indices)!=1):raise ValueError('requested baseline span must be contiguous')
    return max(0,int(indices[0])-1),int(indices[-1])


def baseline_span(data,task,phase=None):
    """Return a contiguous action span including its predecessor state."""
    times,states,phases,tasks=_arrays(data)
    first,last=_span_indices(phases,tasks,task,phase)
    sl=slice(first,last+1)
    return dict(times=times[sl]-times[first],states=states[sl].copy(),phases=phases[sl].copy(),tasks=tasks[sl].copy())


def _assigned(assignment,task):
    try:arm=assignment[task] if isinstance(assignment,dict) else assignment[task-1]
    except (KeyError,IndexError,TypeError) as exc:raise ValueError('missing task assignment') from exc
    if arm not in (0,1):raise ValueError('assignment must use arm indexes 0 or 1')
    return int(arm)


def arm_chain_from_baseline(data,arm,assignment,start_task,end_task):
    """Project a contiguous same-arm run without retiming any interval.

    Intermediate transfers remain; the final task's outgoing transfer is excluded.
    Phase changes belong to the next piece at its shared predecessor timestamp.
    """
    if arm not in (0,1) or start_task<1 or end_task<start_task:raise ValueError('valid arm and ordered task range required')
    times,states,phases,tasks=_arrays(data)
    for task in range(start_task,end_task+1):
        if _assigned(assignment,task)!=arm:raise ValueError('task assignment differs from requested arm')
        _span_indices(phases,tasks,task,None)
    first,_=_span_indices(phases,tasks,start_task,None)
    _,last=_span_indices(phases,tasks,end_task,None)
    active_start=first+1 if first else 0
    if np.any((tasks[active_start:last+1]<start_task)|(tasks[active_start:last+1]>end_task)):
        raise ValueError('task run must be contiguous')
    own=states[first:last+1][:,np.r_[np.arange(arm*6,arm*6+6),13+arm]].copy()
    relative=times[first:last+1]-times[first]
    chain=new_chain(own[0]);chain['states']=own;chain['times']=relative
    # Every interval is identified by the label on its destination row.
    begin=first+1
    while begin<=last:
        end=begin
        while end<last and phases[end+1]==phases[begin] and tasks[end+1]==tasks[begin]:end+=1
        chain['pieces'].append(dict(start=float(times[begin-1]-times[first]),end=float(times[end]-times[first]),phase=str(phases[begin]),task=None if phases[begin]=='transfer' else int(tasks[begin])))
        begin=end+1
    return chain


def serial_part(data,assignment,start_index=0,end_index=None):
    """Copy an inclusive baseline slice with two-arm serial provenance."""
    times,states,phases,tasks=_arrays(data)
    end_index=len(times)-1 if end_index is None else end_index
    if not 0<=start_index<=end_index<len(times):raise ValueError('valid inclusive index range required')
    sl=slice(start_index,end_index+1);count=end_index-start_index+1
    labels=np.full((count,2),'idle',dtype='<U64');owners=np.zeros((count,2),int)
    for j,i in enumerate(range(start_index,end_index+1)):
        if phases[i]=='transfer':labels[j,:]='transfer'
        else:
            arm=_assigned(assignment,int(tasks[i]));labels[j,arm]=phases[i];owners[j,arm]=tasks[i]
    return dict(times=times[sl]-times[start_index],states=states[sl].copy(),phases=labels,tasks=owners)


def stitch_parts(parts):
    """Join exact seams, dropping shared first rows and adding no time knots."""
    if not parts:raise ValueError('at least one part required')
    output={key:[] for key in ('time','states','phases','tasks','stage')}
    offset=0.;previous=None
    for number,part in enumerate(parts):
        times,states,phases,tasks=_arrays(part,2)
        if times[0]!=0.:raise ValueError('part time must start at zero')
        if previous is not None and not np.allclose(previous,states[0],atol=1e-12,rtol=0):raise ValueError('splice seam mismatch')
        start=0 if previous is None else 1
        stage=np.asarray(part.get('stage',np.full(len(times),number)))
        if stage.ndim==0:stage=np.full(len(times),stage.item())
        if stage.shape!=(len(times),):raise ValueError('matching stage rows required')
        for key,value in (('time',offset+times),('states',states),('phases',phases),('tasks',tasks),('stage',stage)):
            output[key].append(value[start:].copy())
        previous=states[-1];offset+=float(times[-1])
    result={key:np.concatenate(value) for key,value in output.items()}
    if np.any(np.diff(result['time'])<=0):raise ValueError('stitched times must strictly ascend')
    return result


