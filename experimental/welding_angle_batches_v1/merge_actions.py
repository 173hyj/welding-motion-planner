"""Minimum-stop cover of already certified complete local actions."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import shutil
import sys
from .actions import atomic,digest
from .grouping import minimal_angle_covers
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def cover_order(covers,start,end):
    candidates=[]
    for cover in covers:
        for route in itertools.permutations(cover):
            full=(start,)+route+(end,)
            travel=sum(abs(b-a) for a,b in zip(full,full[1:]))
            movements=sum(abs(b-a)>1e-7 for a,b in zip(full,full[1:]))
            candidates.append((movements,travel,route))
    return min(candidates)[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actions',type=Path,action='append',required=True)
    p.add_argument('--start-angle',type=float,default=30)
    p.add_argument('--end-angle',type=float,default=90)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    out=args.out.resolve()
    if out.exists(): raise FileExistsError('Fresh output required')
    sys.path[:0]=[str(ROOT),str(HERE.parent/'welding_retreat_transfer_v1')]
    from integrated_retreat_v117 import online_retreat_ids
    bank={}; available={}; inputs={}; bound={'source_sha256':{},'code_sha256':{}}
    for directory in args.actions:
        directory=directory.resolve()
        report=json.loads((directory/'report.json').read_text())
        accepted=online_retreat_ids(report,directory)
        for section in bound:
            for name,h in report['manifest'].get(section,{}).items():
                if digest(ROOT/name)!=h: raise ValueError('Changed action input: '+name)
                if name in bound[section] and bound[section][name]!=h:
                    raise ValueError('Conflicting source identities')
                bound[section][name]=h
        inputs[str(directory/'report.json')]=digest(directory/'report.json')
        for row in report['rows']:
            if row['weld_id'] not in accepted: continue
            angle=float(row['initial']['positioner_deg'])
            available.setdefault(angle,set()).add(row['weld_id'])
            bank.setdefault(row['weld_id'],[]).append((angle,directory,row))
    required=set(bank)
    if len(required)!=21: raise ValueError('Exactly 21 supported welds required')
    covers=minimal_angle_covers(available,required)
    angles=cover_order(covers,args.start_angle,args.end_angle)
    rank={angle:i for i,angle in enumerate(angles)}
    out.mkdir(parents=True)
    rows=[]
    for i,wid in enumerate(sorted(required)):
        choices=[x for x in bank[wid] if x[0] in angles]
        # Prefer lower opening and raw arm movement within the chosen stop cover.
        angle,directory,row=min(choices,key=lambda x:(abs(x[2]['best']['driver_mm']),rank[x[0]],str(x[1])))
        row=deepcopy(row)
        source=directory/row['best']['path_file']
        path=json.loads(source.read_text())
        name=f'paths/retreat_{i:02d}.json'
        (out/'paths').mkdir(exist_ok=True)
        shutil.copyfile(source,out/name)
        if digest(out/name)!=row['best']['path_sha256']: raise ValueError('Copied path changed')
        row['best']['path_file']=name
        atomic(out/f'point_{i:02d}.json',row)
        rows.append(row)
    bound['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    bound['code_sha256'][str((HERE/'grouping.py').relative_to(ROOT))]=digest(HERE/'grouping.py')
    bound.update(required_weld_ids=sorted(required),input_reports_sha256=inputs,
                 selected_angle_order_deg=list(angles),physical_source_limits=True)
    for section in ('source_sha256','code_sha256'):
        for name,h in bound[section].items():
            if digest(ROOT/name)!=h: raise ValueError('Input changed during merge')
    outputs={str(f.relative_to(out)):digest(f) for f in out.rglob('*.json')}
    result=dict(manifest=bound,summary=dict(complete=True,points=21,passed=21,
        required_straight_distance_mm=1000.,all21_actions_passed=True,transfers_planned=False),
        rows=rows,output_sha256=outputs,minimum_validated_action_angle_covers=covers,
        selected_angle_order_deg=list(angles),scope='Minimum stop cover of supplied validated local actions, not a full transfer route. Original action paths copied unchanged.')
    online_retreat_ids(result,out)
    atomic(out/'manifest.json',bound);atomic(out/'report.json',result)
    print(json.dumps(dict(angles=angles,points=21)),flush=True)


if __name__=='__main__':main()
