"""Freeze completed branch-compatible positives, keeping every requested weld."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from signed_wrist_model_v40 import path_compatibility
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_tool_exit_v1'))
from run_geometry import atomic, digest

ROOT=Path(__file__).resolve().parents[2]


def select_row(row,path,signs,parks,name,sha):
    initial=row['initial']
    if (not path.get('passed') or path.get('weld_id')!=row['weld_id'] or
        path.get('fixed_distance_mm')!=2000. or path.get('length_mm')!=2000. or
        path.get('robot_id')!=initial['robot_id'] or path.get('other_robot_id')!=initial['other_robot_id'] or
        path.get('q_start_rad')!=initial['q_rad'] or path.get('positioner_deg')!=initial['positioner_deg'] or
        not path.get('frames') or path['q_end_rad']!=path['frames'][-1]['q_rad'] or
        path['driver_mm']!=row['best']['driver_mm'] or
        path_compatibility(path,signs,parks)!=(True,True)):
        raise ValueError('Unfinished, mismatched, short or incompatible selected retreat')
    selected=copy.deepcopy(row)
    selected.setdefault('original_initial',copy.deepcopy(initial))
    selected['initial']['other_q_rad']=copy.deepcopy(path['other_q_rad'])
    selected['best'].update(path_file=name,path_sha256=sha,length_mm=2000.)
    selected.update(mixed_wrist_branch_compatible=True,fresh_cold_required=True,
                    selection_status='completed_candidate_not_yet_cold_verified')
    selected.pop('opening_profile',None)  # A changed park needs fresh opening checks.
    return selected


def defer_row(row):
    result=copy.deepcopy(row)
    if result['best']:
        result['retained_existing_candidate']=result['best']
    result.update(best=None,mixed_wrist_branch_compatible=False,
                  selection_status='awaiting_branch_reselection_not_failure')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--running-out',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--extra-code',type=Path,nargs='*',default=[])
    args=parser.parse_args();started=time.perf_counter()
    baseline,running,out=(p.resolve() for p in (args.baseline,args.running_out,args.out))
    if out.exists():raise FileExistsError('Fresh output required')
    report=json.loads((baseline/'report.json').read_text())
    live=json.loads((running/'manifest.json').read_text())
    if (len(report['rows'])!=21 or len({r['weld_id'] for r in report['rows']})!=21 or
        not report['summary']['complete'] or report['summary']['required_straight_distance_mm']!=2000.):
        raise ValueError('Complete full-21 2000mm baseline required')
    signs=live['settings']['per_robot_wrist_sign'];parks=live['settings']['per_robot_inactive_park']
    sources={**report['manifest']['source_sha256'],**report['manifest']['code_sha256'],
             **live['source_sha256'],**live['code_sha256']}
    for p in (baseline/'report.json',running/'manifest.json'):
        sources[str(p.relative_to(ROOT))]=digest(p)
    for name,sha in report['output_sha256'].items():
        sources[str((baseline/name).relative_to(ROOT))]=sha
    candidates={}
    originals={r['weld_id']:r for r in report['rows']}
    for row in report['rows']:
        if not row['best']:continue
        for p in (baseline/row['best']['path_file'],running/'paths'/('mixed_retrace_'+row['weld_id']+'.json')):
            if not p.exists():continue
            path=json.loads(p.read_text())
            if path_compatibility(path,signs,parks)==(True,True):
                select_row(row,path,signs,parks,'selected.json',digest(p))
                candidates[row['weld_id']]=(row,p)
    for p in sorted(running.glob('OP10L*_angle_*.json')):
        if p.name.endswith('_progress.json'):continue
        row=json.loads(p.read_text())
        if not row.get('best'):continue
        if row['weld_id'] not in originals:raise ValueError('Unknown weld identity')
        pathfile=running/row['best']['path_file'];path=json.loads(pathfile.read_text())
        if digest(pathfile)!=row['best']['path_sha256']:raise ValueError('Changed completed path')
        select_row(row,path,signs,parks,'selected.json',digest(pathfile))
        candidates[row['weld_id']]=(row,pathfile)
        sources[str(p.relative_to(ROOT))]=digest(p)
    for row,p in candidates.values():sources[str(p.relative_to(ROOT))]=digest(p)
    codes={str(p.relative_to(ROOT)):digest(p) for p in
           [Path(__file__),Path(__file__).with_name('signed_wrist_model_v40.py'),*[p.resolve() for p in args.extra_code]]}
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed input '+name)
    out.mkdir(parents=True);rows=[]
    for i,row in enumerate(report['rows']):
        retained=copy.deepcopy(row)
        if row['best']:
            old=baseline/row['best']['path_file'];name=f'paths/retained_{i:02d}.json'
            (out/name).parent.mkdir(parents=True,exist_ok=True);(out/name).write_bytes(old.read_bytes())
            retained['best'].update(path_file=name,path_sha256=digest(out/name))
        if row['weld_id'] not in candidates:
            rows.append(defer_row(retained));continue
        selected,p=candidates[row['weld_id']];name=f'paths/selected_{i:02d}.json'
        (out/name).parent.mkdir(parents=True,exist_ok=True);(out/name).write_bytes(p.read_bytes())
        r=select_row(selected,json.loads(p.read_text()),signs,parks,name,digest(out/name))
        if retained['best']:r['retained_existing_candidate']=retained['best']
        rows.append(r)
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise ValueError('Changed input while freezing '+name)
    outputs={str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')}
    summary=dict(complete=True,snapshot_complete=True,live_search_complete=False,points=21,
        passed=sum(bool(r['best']) for r in rows),required_straight_distance_mm=2000.,
        retained_existing_candidates=report['summary']['passed'],historical_chain_intact=False,
        fresh_cold_required=True,wall_s=time.perf_counter()-started)
    atomic(out/'report.json',dict(manifest=dict(source_sha256=sources,code_sha256=codes,settings=live['settings']),
        summary=summary,rows=rows,output_sha256=outputs,
        scope='Completed compatible paths only; deferred welds remain among all21 tasks. '
              'No negative or old cold certificate reused; fresh full-source cold required. '
              'Historical missing dependencies explicitly disclosed in sealed v43 baseline.'))
    print('MIXED_SNAPSHOT_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
