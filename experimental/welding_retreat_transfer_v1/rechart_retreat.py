"""Test fixed wrist chart proposals against complete source motion geometry."""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'welding_variable_open_v1')]
from run_transfer_v5 import VariableToolWorld,assert_frozen_geometry_cache
from run_geometry import atomic,digest,initialize_worker
from run_arm import full_probe
from wrist_chart import chart_options

ENGINE=None


def job(index,row,source,out):
    original=json.loads((source/row['best']['path_file']).read_text())
    rid=row['initial']['robot_id'];model=ENGINE.world.models[rid]
    ENGINE.set_angle(row['initial']['positioner_deg'])
    weld=next(w for w in ENGINE.scene['welds'] if w['id']==row['weld_id'])
    moved,_=ENGINE.target(weld,row['initial'])
    old_q=np.asarray([f['q_rad'] for f in original['frames']])
    trials=[];accepted=None;selected=None
    for option in chart_options(old_q,*model.bounds):
        trial=dict(label=option['label'],passed=True,source_samples=0,maximum_fk_difference=0.)
        trials.append(trial)
        path=copy.deepcopy(original)
        for i,(frame,q) in enumerate(zip(path['frames'],option['q'])):
            actual=model.tcp(q);old=model.tcp(old_q[i])
            trial['maximum_fk_difference']=max(trial['maximum_fk_difference'],float(np.max(abs(actual-old))))
            if not model.limits_valid(q) or trial['maximum_fk_difference']>1e-6:
                trial.update(passed=False,reason='source_limits_or_fk_mismatch');break
            frame['q_rad']=q.tolist();frame['tcp_xyz_mm']=actual[:3,3].tolist()
        if not trial['passed']:continue
        initial=dict(row['initial'],q_rad=path['frames'][0]['q_rad'],
                     actual_tcp_matrix=model.tcp(option['q'][0]).ravel().tolist())
        path['q_start_rad']=initial['q_rad'];path['q_end_rad']=path['frames'][-1]['q_rad'];path['candidate']=initial
        start=model.tcp(option['q'][0]);driver=path['driver_mm']
        opened=ENGINE.state(rid,start,driver,moved,ENGINE.world.static,'opening')
        context=ENGINE.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
        previous=None
        for frame in path['frames']:
            q=np.asarray(frame['q_rad'])
            samples=[q] if previous is None else [(previous+q)/2.,q]
            for sample in samples:
                actual=model.tcp(sample);travel=float((actual[:3,3]-start[:3,3])@path['direction_world'])
                phase='closed' if frame['driver_mm']==0. else 'opening' if frame['travel_mm']==0. else 'release' if travel<=5.000001 else 'separated'
                check=full_probe(ENGINE,rid,sample,initial['other_robot_id'],np.asarray(initial['other_q_rad']),
                    frame['driver_mm'],moved,phase,context if phase=='release' else None)
                trial['source_samples']+=1
                if not check['passed']:
                    trial.update(passed=False,reason='full_source_rejected',failure=check);break
            if not trial['passed']:break
            previous=q
        if trial['passed']:
            accepted=path;selected=initial;break
    result=copy.deepcopy(row)
    if accepted:
        name=f'paths/retreat_{index:02d}.json';atomic(out/name,accepted)
        result.update(initial=selected,best=dict(row['best'],path_file=name,path_sha256=digest(out/name)),
                      original_initial=row['initial'],wrist_chart_reselected=True,chart_trials=trials)
    else:
        result.update(best=None,passed=False,chart_trials=trials)
    atomic(out/f'point_{index:02d}.json',result)
    print('RECHART',row['weld_id'],bool(accepted),[(t['label'],t['passed']) for t in trials],flush=True)
    return index,result


def main():
    global ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--retreat-out',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    source,out=args.retreat_out.resolve(),args.out.resolve()
    if out.exists():raise FileExistsError('Use new output')
    parent=json.loads((source/'report.json').read_text());assert_frozen_geometry_cache()
    baseline={**parent['manifest']['source_sha256'],**parent['manifest']['code_sha256'],
              str((source/'report.json').relative_to(ROOT)):digest(source/'report.json')}
    for name,value in parent['output_sha256'].items():baseline[str((source/name).relative_to(ROOT))]=value
    codes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),HERE/'wrist_chart.py']}
    for name,value in {**baseline,**codes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Inputs changed')
    out.mkdir();manifest=dict(source_sha256=baseline,code_sha256=codes,settings=dict(
        distance_mm=parent['summary']['required_straight_distance_mm'],constant_chart_per_retreat=True,
        proposal_only=True,complete_source_checks_required=True))
    atomic(out/'manifest.json',manifest);start=time.perf_counter()
    ENGINE=VariableToolWorld(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()))
    rows={i:copy.deepcopy(r) for i,r in enumerate(parent['rows']) if not r['best']}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,
        mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        futures=[pool.submit(job,i,r,source,out) for i,r in enumerate(parent['rows']) if r['best']]
        for f in concurrent.futures.as_completed(futures):i,r=f.result();rows[i]=r
    for name,value in {**baseline,**codes}.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Evidence changed')
    outputs={str(p.relative_to(out)):digest(p) for p in list(out.glob('point_*.json'))+list((out/'paths').glob('*.json'))}
    report=dict(manifest=manifest,summary=dict(complete=True,points=21,passed=sum(bool(r['best']) for r in rows.values()),
        required_straight_distance_mm=parent['summary']['required_straight_distance_mm'],wall_s=time.perf_counter()-start),
        rows=[rows[i] for i in sorted(rows)],output_sha256=outputs,
        scope='Same Cartesian process pose and ray, constant alternative source wrist chart; only fully checked paths retained. Cold source/FK audit still required.')
    atomic(out/'report.json',report);print('RECHART_SUMMARY',report['summary'],flush=True)


if __name__=='__main__':main()
