"""Merge sealed same-distance positive paths without losing existing nodes."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_tool_exit_v1'))
from run_geometry import atomic,digest

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',action='append',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():raise FileExistsError('Use fresh merged directory')
    rows=None; inventory=None; sources={}; reports=[]; positives={}
    for directory in args.input:
        directory=directory.resolve();path=directory/'report.json'
        report=json.loads(path.read_text())
        if not report['summary']['complete'] or report['summary']['required_straight_distance_mm'] != 2000.:raise RuntimeError('Complete2000mm report required')
        ids=[r['weld_id'] for r in report['rows']]
        if len(ids)!=21 or len(set(ids))!=21:raise RuntimeError('Exactly21 unique welds required')
        if inventory is None:
            inventory=set(ids);rows=copy.deepcopy(report['rows'])
        elif set(ids)!=inventory:raise RuntimeError('Inventory changed')
        seal={**report['manifest']['source_sha256'],**report['manifest']['code_sha256'],str(path.relative_to(ROOT)):digest(path)}
        for name,sha in report['output_sha256'].items():
            p=directory/name
            if digest(p)!=sha:raise RuntimeError('Output changed '+name)
            seal[str(p.relative_to(ROOT))]=sha
        for name,sha in seal.items():
            if digest(ROOT/name)!=sha:raise RuntimeError('Input changed '+name)
            if name in sources and sources[name]!=sha:raise RuntimeError('Conflicting source seal')
            sources[name]=sha
        reports.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),passed=report['summary']['passed']))
        for row in report['rows']:
            if not row['best'] or row['weld_id'] in positives:continue
            p=directory/row['best']['path_file'];trajectory=json.loads(p.read_text())
            if digest(p)!=row['best']['path_sha256'] or not trajectory['passed'] or trajectory['length_mm']!=2000.:raise RuntimeError('Positive identity/distance mismatch')
            positives[row['weld_id']]=(copy.deepcopy(row),p)
    out.mkdir(parents=True)
    for i,row in enumerate(rows):
        if row['weld_id'] not in positives:continue
        selected,p=positives[row['weld_id']];name=f'paths/merged_retreat_{i:02d}.json'
        (out/name).parent.mkdir(parents=True,exist_ok=True);(out/name).write_bytes(p.read_bytes())
        if digest(out/name)!=digest(p):raise RuntimeError('Copy mismatch')
        selected['best'].update(path_file=name,path_sha256=digest(out/name))
        rows[i]=selected
    codes={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(distance_mm=2000.,input_reports=reports,prefer_first_positive=True))
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Input changed during merge')
    atomic(out/'manifest.json',manifest)
    summary=dict(complete=True,points=21,passed=len(positives),required_straight_distance_mm=2000.,input_batches=len(reports),wall_s=0.)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,output_sha256={str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')},scope='Byte-exact merged complete-distance candidate evidence; cold audit and actual full-state transfers still required. Source manifests remain conditional on original archived implementation.'))
    print('MERGED_RETREAT_INVENTORY',summary,flush=True)


if __name__=='__main__':main()
