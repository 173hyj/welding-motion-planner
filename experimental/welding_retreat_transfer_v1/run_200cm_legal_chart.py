"""Retry 200cm GW4 using a legal equivalent whole-stroke joint chart."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import run_retreat_v2 as runner
from run_geometry import atomic,digest
from wrist_chart import chart_options


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Use new output')
    runner.assert_frozen_geometry_cache()
    parent_path=ROOT/'build/welding_variable_open_v1/report.json'
    prior_path=ROOT/'build/welding_retreat_200cm_v2/report.json'
    parent=json.loads(parent_path.read_text());prior=json.loads(prior_path.read_text())
    baseline={**prior['manifest']['source_sha256'],**prior['manifest']['code_sha256'],
              str(prior_path.relative_to(ROOT)):digest(prior_path)}
    for n,v in prior['output_sha256'].items():baseline[str((prior_path.parent/n).relative_to(ROOT))]=v
    codes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),HERE/'wrist_chart.py']}
    for n,v in {**baseline,**codes}.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Frozen input changed')
    out.mkdir();manifest=dict(source_sha256=baseline,code_sha256=codes,settings=dict(
        distance_mm=2000.,arm_candidates_per_amplitude=48,legal_equivalent_chart=True,
        original_roll_normal_positioner_other_park_unchanged=True))
    atomic(out/'manifest.json',manifest);t=time.perf_counter()
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    engine=runner.VariableToolWorld(scene);runner._ENGINE=engine
    parent_rows=parent['rows'];index=next(i for i,r in enumerate(parent_rows) if r['weld_id']=='OP10L-SUB3-POINT-GW-4')
    row=copy.deepcopy(parent_rows[index]);initial=row['initial'];model=engine.world.models[initial['robot_id']]
    q=np.asarray(initial['q_rad']);opts=chart_options(q[None,:],*model.bounds)
    option=next(c for c in opts if c['label']=='q6_plus_360')
    new_q=option['q'][0]
    if np.max(abs(model.tcp(q)-model.tcp(new_q)))>1e-6:raise RuntimeError('Chart changes weld target')
    row['initial']=dict(initial,q_rad=new_q.tolist(),actual_tcp_matrix=model.tcp(new_q).ravel().tolist())
    weld=next(w for w in scene['welds'] if w['id']==row['weld_id'])
    _,result=runner.point_job(index,row,weld,parent_path.parent,out,2000.,48)
    result.update(original_initial=initial,legal_wrist_chart='q6_plus_360',welding_posture_reselected=True)
    atomic(out/f'point_{index:02d}.json',result)
    rows=copy.deepcopy(prior['rows'])
    for i,r in enumerate(rows):
        if r['best']:
            data=json.loads((prior_path.parent/r['best']['path_file']).read_text())
            name=f'paths/retreat_{i:02d}.json';atomic(out/name,data)
            r['best'].update(path_file=name,path_sha256=digest(out/name))
    rows[index]=result
    outputs={str(p.relative_to(out)):digest(p) for p in list(out.glob('point_*.json'))+list((out/'paths').glob('*.json'))}
    for n,v in {**baseline,**codes}.items():
        if digest(ROOT/n)!=v:raise RuntimeError('Frozen input changed during run')
    report=dict(manifest=manifest,summary=dict(complete=True,points=21,passed=sum(bool(r['best']) for r in rows),
        required_straight_distance_mm=2000.,wall_s=time.perf_counter()-t),rows=rows,output_sha256=outputs,
        scope='Legal alternate initial wrist chart, identical weld target. Whole new retreat recomputed and source checked. No instantaneous wrap in an existing motion; cold check required.')
    atomic(out/'report.json',report);print('LEGAL_CHART_200CM_SUMMARY',report['summary'],flush=True)


if __name__=='__main__':main()
