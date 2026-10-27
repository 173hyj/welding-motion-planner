"""Conservative display-clock correction for fresh single-active-arm anchored stages."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .anchored import literal_slice,assemble_segments,stage_gate
from experimental.welding_candidate_selection_v1.pool import atomic,digest
from experimental.welding_candidate_selection_v1.run import verify_bindings
from experimental.welding_candidate_selection_v1.trajectory import motion_metrics,acceptance


def retime_piece(piece,active_arm):
    if active_arm not in (0,1):raise ValueError('One active arm required')
    result={k:np.asarray(v).copy() for k,v in piece.items()};states=result['states'];clock=result.get('time',result.get('times'));tcp=result['tcp']
    if states.shape!=(len(clock),15) or tcp.shape!=(len(clock),2,3) or not np.isfinite(states).all() or not np.isfinite(clock).all() or not np.isfinite(tcp).all() or np.any(np.diff(clock)<=0):raise ValueError('Valid full15, TCP and ascending clock required')
    passive=1-active_arm;passive_axes=np.r_[np.arange(passive*6,passive*6+6),13+passive]
    if not np.all(states[:,passive_axes]==states[0,passive_axes]) or not np.all(tcp[:,passive]==tcp[0,passive]):raise ValueError('Passive arm must remain stationary for clock-only source reuse')
    if not np.all(states[:,12]==states[0,12]):raise ValueError('Positioner must remain stationary')
    original_dt=np.diff(clock);delta=abs(np.diff(states,axis=0));required_axis=np.max(delta/np.r_[np.full(12,np.deg2rad(30)),np.deg2rad(15),80.,80.],axis=1)
    tcp_distance=np.linalg.norm(np.diff(tcp,axis=0),axis=2).max(axis=1);required_tcp=tcp_distance/200.
    protected=np.asarray(result['phases'])[:-1,active_arm]=='weld';tasks=sorted(int(t) for t in np.unique(result['tasks'][:-1,active_arm][protected]))
    if np.any(delta[protected]>1e-9) or np.any(tcp_distance[protected]>1e-8):raise ValueError('Weld interval must be stationary')
    for task in tasks:
        mask=protected&(result['tasks'][:-1,active_arm]==task)
        if task<=0 or abs(float(original_dt[mask].sum())-.6)>1e-7:raise ValueError('START-labeled weld must retain0.6s')
    dt=np.maximum.reduce([original_dt,required_axis,required_tcp])
    if np.any(dt[protected]>original_dt[protected]+1e-12):raise ValueError('Cannot lengthen protected weld interval')
    dt[protected]=original_dt[protected]
    joined=np.r_[0.,np.cumsum(dt)]
    for key in ('time','times'):
        if key in result:result[key]=joined.copy()
    for key,value in piece.items():
        if key not in ('time','times') and not np.array_equal(result[key],value):raise ValueError('Clock correction changed geometry or labels')
    stats=dict(active_arm=active_arm,passive_arm_stationary=True,positioner_stationary=True,modified_intervals=int(np.sum(dt>original_dt+1e-12)),duration_before_s=float(clock[-1]-clock[0]),duration_after_s=float(joined[-1]),added_seconds=float(joined[-1]-(clock[-1]-clock[0])),protected_tasks=tasks,protected_weld_dt_unchanged=True,original_max_tcp_mm_s=float(np.max(tcp_distance/original_dt)),retimed_max_tcp_mm_s=float(np.max(tcp_distance/np.diff(joined))),identical_ordered_full15_states=True,identical_tcp_phases_tasks=True,source_basis='Identical ordered static-geometry source samples; only non-weld clock intervals lengthened. Single active arm, passive arm and positioner held. No hardware timing claim.')
    return result,stats


def process(result_folder,pool_folder,out):
    result_folder=Path(result_folder);out=Path(out).resolve();report_file=result_folder/'report.json';pool_file=Path(pool_folder)/'report.json'
    if out.exists():raise FileExistsError('Fresh retimed result directory required')
    report=json.loads(report_file.read_text());pool=json.loads(pool_file.read_text())
    if report['pool_sha256']!=digest(pool_file):raise ValueError('Pool differs from original anchored result')
    bindings=dict(report['manifest']['bindings']);bindings[str(report_file.resolve())]=digest(report_file);own=Path(__file__).absolute();bindings[str(own)]=digest(own)
    old=dict(np.load(report['comparison_route'],allow_pickle=False));baseline_metrics=motion_metrics(old['states'],old.get('time',old.get('times')))
    original_trial=result_folder/'trial_route.npz';bindings[str(original_trial.resolve())]=digest(original_trial)
    stages=copy.deepcopy(report['stages']);original_pieces=[]
    for stage in stages:
        verify_bindings({stage['file']:stage['sha256']});bindings[stage['file']]=stage['sha256'];original_pieces.append(dict(np.load(stage['file'],allow_pickle=False)))
    verify_bindings(bindings);verify_bindings({r['file']:r['sha256'] for r in report['edges'] if r.get('passed')})
    reconstructed=assemble_segments(old,report['stage_bounds'],original_pieces);original=dict(np.load(original_trial,allow_pickle=False))
    if set(original)!=set(reconstructed) or any(not np.array_equal(original[k],v) for k,v in reconstructed.items()):raise ValueError('Original stage artifacts do not reproduce sealed trial')
    selected={row['task']:copy.deepcopy(row) for row in report['selected_actions']};inc_report_path=next((Path(p) for p in bindings if Path(p).name=='report.json' and Path(p).parent.resolve()==Path(report['comparison_route']).parent.resolve()),None)
    if inc_report_path is None:raise ValueError('Bound incumbent report missing')
    incumbent=json.loads(inc_report_path.read_text());inc_actions={row['task']:row for row in incumbent['selected_actions']};pieces=[];corrections=[]
    out.mkdir()
    for index,(stage,piece) in enumerate(zip(stages,original_pieces)):
        if stage['source']=='new_source_checked':
            active=[arm for arm,order in enumerate(stage['task_ids']) if order]
            if len(active)!=1:raise ValueError('Concurrent fresh stages require a new source schedule; clock-only repair refused')
            if not np.allclose(piece['tcp'][[0,-1]],old['tcp'][stage['bounds']],atol=1e-9,rtol=0):raise ValueError('Stage TCP anchors differ from sealed incumbent boundary')
            candidate,stats=retime_piece(piece,active[0]);metrics=motion_metrics(candidate['states'],candidate['time']);gate=stage_gate(stage['before'],metrics)
            stats.update(stage=index,stage_gate=gate,original_file=stage['file'],original_sha256=stage['sha256']);corrections.append(stats)
            if gate['accepted']:
                piece=candidate;stage['after']=metrics;stage['timing_correction']=stats;stage['validation']=dict(stage['validation'],time_only_retimed=True,ordered_source_states_unchanged=True,static_geometry_sequence_basis=True)
            else:
                piece=literal_slice(old,*stage['bounds']);stage['source']='incumbent';stage['after']=copy.deepcopy(stage['before']);stage['validation']=dict(passed=True,inherited_incumbent_sha256=report['comparison_sha256']);stage['timing_fallback']=stats
                for task in [t for order in stage['task_ids'] for t in order]:
                    old_action=inc_actions[task]['action'];matches=[(i,a) for i,a in enumerate(pool['layers'][task-1]['actions']) if a['path_sha256']==old_action['path_sha256']]
                    if not matches:raise ValueError('Incumbent fallback action missing')
                    choice,action=matches[0];stage['choices'][str(task) if str(task) in stage['choices'] else task]=choice;selected[task]=dict(inc_actions[task],action=action)
        new_file=out/f'stage_{index:02d}.npz';np.savez_compressed(new_file,**piece);stage.update(file=str(new_file),sha256=digest(new_file));atomic(out/f'stage_{index:02d}.json',stage);pieces.append(piece)
    trial=assemble_segments(old,report['stage_bounds'],pieces);after=motion_metrics(trial['states'],trial.get('time',trial.get('times')));verdict=acceptance(baseline_metrics,after)
    verify_bindings(bindings);verify_bindings({r['file']:r['sha256'] for r in report['edges'] if r.get('passed')})
    verify_bindings(report['manifest']['bindings']);verify_bindings({stage['file']:stage['sha256'] for stage in stages})
    np.savez_compressed(out/'trial_route.npz',**trial)
    if verdict['accepted']:np.savez_compressed(out/'route.npz',**trial)
    final=copy.deepcopy(report);final.update(**verdict,adopted=verdict['accepted'],before=baseline_metrics,after=after,stages=stages,selected_actions=[selected[t] for t in range(1,22)],new_source_checked_stages=sum(stage['source']=='new_source_checked' for stage in stages),baseline_retained=None if verdict['accepted'] else report['comparison_route'],timing_corrections=corrections,timing_parent_report=str(report_file.resolve()),timing_parent_report_sha256=digest(report_file),independent_cold_validation=False)
    final['manifest']['bindings']=bindings;final['manifest']['settings']['timing_postprocess']='Single active arm static source-sequence clock extension; protected0.6s welds unchanged; source stage fallback if any stage gate regresses.'
    atomic(out/'report.json',final);return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('result','pool','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps({k:v for k,v in process(args.result,args.pool,args.out).items() if k in ('accepted','before','after','timing_corrections')},indent=2))

if __name__=='__main__':main()
