"""Frozen selected rays, freshly source-generated at 200 and 300 mm."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experimental/welding_retreat_transfer_v1'
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'experimental/welding_variable_open_v1'),str(ROOT/'experimental/welding_tool_exit_v1')]
ENGINE=None

def frozen_ray(source,length):
    if length not in (200.,300.) or not np.isclose(np.linalg.norm(source['direction_world']),1.,atol=1e-8):
        raise ValueError('Approved distance and normalized original ray required')
    return dict(direction_world=source['direction_world'],direction_frame='current_world',length_mm=float(length)),source['driver_mm']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def generate(job):
    item,length,out=job;started=time.perf_counter();action=item['action'];source_file=Path(action['path_file'])
    if digest(source_file)!=action['path_sha256']:raise ValueError('Frozen source hash changed')
    source=json.loads(source_file.read_text());selected=action['initial']
    ray,driver=frozen_ray(source,length)
    from integrated_retreat_v117 import lift_path,check_path_contract
    ENGINE.set_angle(selected['positioner_deg'])
    weld=next(w for w in ENGINE.scene['welds'] if w['id']==item['weld_id'])
    path=lift_path(ENGINE,weld,selected,selected,ray,driver_mm=driver)
    row=dict(task=item['task'],weld_id=item['weld_id'],arm=item['arm'],angle=item['angle'],initial=selected,
             source_path=str(source_file),source_sha256=action['path_sha256'],passed=bool(path.get('passed')),length_mm=length,
             elapsed_s=time.perf_counter()-started,reason=path.get('reason'))
    path['weld_id']=item['weld_id']
    if row['passed']:check_path_contract(path,selected,length)
    target=Path(out)/f'paths/task_{item["task"]:02d}.json';atomic(target,path)
    row.update(path_file=str(target),path_sha256=digest(target))
    atomic(Path(out)/f'point_{item["task"]:02d}.json',row)
    print('FRESH_SOURCE_ACTION',length,item['task'],row['passed'],row['elapsed_s'],row['reason'],flush=True)
    return row

def main():
    global ENGINE
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    if (args.out/'progress.json').exists():raise FileExistsError('Fresh experimental output required')
    config=json.loads(args.config.read_text(encoding='utf-8'));items=config['selected_actions']
    if [r['task'] for r in items]!=[7,*range(1,7),*range(8,22)]:raise ValueError('Fixed order changed')
    bindings={str(args.config.resolve()):digest(args.config)}
    for item in items:
        a=item['action'];bindings[a['path_file']]=a['path_sha256']
    for name,sha in bindings.items():
        if digest(name)!=sha:raise ValueError('Frozen input changed '+name)
    for folder in [HERE,ROOT/'experimental/welding_variable_open_v1',ROOT/'experimental/welding_tool_exit_v1',Path(__file__).parent]:
        for file in folder.glob('*.py'):bindings[str(file)]=digest(file)
    for file in [ROOT/'build/robot.json',ROOT/'build/scene_placement_corrected_v1.json',ROOT/'scripts/audit_contact_release_v17.py']:
        bindings[str(file)]=digest(file)
    args.out.mkdir(parents=True,exist_ok=True);atomic(args.out/'manifest.json',dict(bindings=bindings,no_hardware=True,source_geometry='Original raw geometry',no_crop=True))
    from run_geometry import initialize_worker
    from run_integrated_retreat_v118 import load_raw_world
    from mixed_bridge_v95 import SIGNS
    initialize_worker();print('RAW_WORLD_LOADING',flush=True)
    ENGINE=load_raw_world(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()),SIGNS)
    # Preserve source raw joint convention used by the selected candidate pool.
    for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
    print('RAW_WORLD_READY',flush=True)
    rows=[]
    # Run the first 200 mm action immediately, before forking the rest.
    rows.append(generate((items[0],200.,args.out/'200')))
    atomic(args.out/'progress.json',dict(rows=rows))
    jobs=[(item,length,args.out/str(int(length))) for length in (200.,300.) for item in items if not(length==200. and item is items[0])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
        for row in pool.map(generate,jobs):
            rows.append(row);atomic(args.out/'progress.json',dict(rows=rows))
    for name,sha in bindings.items():
        if digest(name)!=sha:raise ValueError('Input changed during run '+name)
    atomic(args.out/'generation_report.json',dict(complete=True,rows=rows,passed=sum(r['passed'] for r in rows),manifest=dict(bindings=bindings),no_hardware=True))

if __name__=='__main__':main()
