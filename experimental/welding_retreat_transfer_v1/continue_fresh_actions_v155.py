"""Continue unresolved points of this experiment without historical good poses."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import run_joint_weld_retreat_v154 as search
from integrated_retreat_v117 import online_retreat_ids
from run_geometry import atomic,digest

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--point-seconds',type=float,default=300.)
    args=parser.parse_args();started=time.perf_counter()
    source=args.source.resolve();out=args.out.resolve()
    original=json.loads((source/'report.json').read_text())
    if original['manifest']['settings'].get('prior_pose_hints'):
        raise ValueError('Continue only a no-history-hint experiment')
    accepted=online_retreat_ids(original,source)
    for group in ('source_sha256','code_sha256'):
        for name,h in original['manifest'][group].items():
            if digest(ROOT/name)!=h:raise ValueError('Changed input '+name)
    out.mkdir(parents=True,exist_ok=False)
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    search.ENGINE=search.load_raw_world(scene,search.SIGNS)
    search.OUT=out;search.HINTS={};search.EXCLUDED={}
    rows=copy.deepcopy(original['rows']); continuation_angles={}
    for i,row in enumerate(rows):
        if row['weld_id'] in accepted:
            atomic(out/row['best']['path_file'],json.loads((source/row['best']['path_file']).read_text()))
            assert digest(out/row['best']['path_file'])==row['best']['path_sha256']
            continue
        tried=[a['positioner_deg'] for a in row['angle_trials'] if a['positioner_deg']!=0]
        last=tried[-1] if tried else None
        # The corrected opening-first evaluator may visit previously capped
        # angles fairly. No prior good-pose hints are used.
        angles=search.positioner_angles()
        continuation_angles[row['weld_id']]=angles
        search.SETTINGS=dict(directions=64,poses=96,configurations=4,point_seconds=args.point_seconds,
            angle_seconds=20.,amplitudes=[10.,30.,60.,80.,120.,140.,180.],feedback_angles=angles)
        row['positioner_done']=False
        weld=next(w for w in scene['welds'] if w['id']==row['weld_id'])
        _,rows[i]=search.search_point(i,row,weld,'positioner')
    for i,row in enumerate(rows):atomic(out/f'point_{i:02d}.json',row)
    manifest=copy.deepcopy(original['manifest'])
    manifest['source_sha256'][str((source/'report.json').relative_to(ROOT))]=digest(source/'report.json')
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            f=Path(filename).resolve()
            if f.suffix=='.py' and f.is_relative_to(ROOT) and '.venv' not in f.parts:
                manifest['code_sha256'][str(f.relative_to(ROOT))]=digest(f)
    manifest['settings']['continuation_settings']=dict(point_seconds=args.point_seconds,angle_seconds=20.,directions=64,poses=96,configurations=4,per_weld_angle_order=continuation_angles)
    manifest['settings']['same_experiment_continuation']=True
    manifest['settings']['historical_good_pose_candidates_injected']=False
    summary=dict(original['summary'],passed=sum(r['passed'] for r in rows),
        online_validated_retreats=sum(r['passed'] for r in rows),wall_s=time.perf_counter()-started,
        prior_stage_wall_s=original['summary']['wall_s'],timing_scope='Continuation only; original fresh action generation separately recorded',
        full_inventory_action_planned=all(r['passed'] for r in rows))
    outputs={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    report=dict(manifest=manifest,summary=summary,rows=rows,output_sha256=outputs,
        scope='Same fresh experiment continuation: unchanged accepted actions retained; failed points explore unfinished angles. No historical hints or good-pose candidates, no extra cold acceptance.')
    online_retreat_ids(report,out);atomic(out/'manifest.json',manifest);atomic(out/'report.json',report)
    print('CONTINUATION_SUMMARY',summary,flush=True)
    if not summary['full_inventory_action_planned']:raise SystemExit(2)


if __name__=='__main__':main()
