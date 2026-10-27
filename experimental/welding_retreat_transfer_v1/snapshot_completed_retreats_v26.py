"""Snapshot only finalized positive point-angle outputs from a live sealed search."""
import argparse,copy,json,time
from pathlib import Path
from merge_retreat_inventory_v19 import atomic,digest,ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--running-out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();started=time.perf_counter()
    if out.exists():raise FileExistsError('Fresh snapshot')
    baseline=args.baseline.resolve();running=args.running_out.resolve()
    previous=json.loads((baseline/'report.json').read_text());live_manifest=json.loads((running/'manifest.json').read_text())
    if not previous['summary']['complete'] or previous['summary']['required_straight_distance_mm']!=2000.:raise RuntimeError('Completed2m baseline')
    sources={**previous['manifest']['source_sha256'],**previous['manifest']['code_sha256'],**live_manifest['source_sha256'],**live_manifest['code_sha256']}
    for p in [baseline/'report.json',running/'manifest.json']:sources[str(p.relative_to(ROOT))]=digest(p)
    for name,sha in previous['output_sha256'].items():
        if digest(baseline/name)!=sha:raise RuntimeError('Old evidence changed')
        sources[str((baseline/name).relative_to(ROOT))]=sha
    rows=copy.deepcopy(previous['rows']);inventory={r['weld_id'] for r in rows}
    if len(rows)!=21 or len(inventory)!=21:raise RuntimeError('Expected21unique')
    positives={r['weld_id']:(copy.deepcopy(r),baseline/r['best']['path_file']) for r in rows if r['best']}
    picked=[]
    for p in sorted(running.glob('OP10L*_angle_*.json')):
        if p.name.endswith('_progress.json'):continue
        row=json.loads(p.read_text())
        if not row.get('best') or row['weld_id'] in positives:continue
        if row['weld_id'] not in inventory or row['required_straight_distance_mm']!=2000.:raise RuntimeError('Changed inventory/distance')
        path=running/row['best']['path_file'];trajectory=json.loads(path.read_text())
        if digest(path)!=row['best']['path_sha256'] or not trajectory['passed'] or trajectory['length_mm']!=2000. or trajectory['fixed_distance_mm']!=2000.:raise RuntimeError('Unfinished/invalid positive')
        if trajectory['q_start_rad']!=row['initial']['q_rad'] or trajectory['other_q_rad']!=row['initial']['other_q_rad'] or trajectory['q_end_rad']!=trajectory['frames'][-1]['q_rad']:raise RuntimeError('Positive identity mismatch')
        sources[str(p.relative_to(ROOT))]=digest(p);sources[str(path.relative_to(ROOT))]=digest(path)
        picked.append(dict(weld_id=row['weld_id'],completed_point_report=str(p.relative_to(ROOT)),sha256=digest(p)))
        positives[row['weld_id']]=(copy.deepcopy(row),path)
    for name,sha in sources.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Inputs changed '+name)
    out.mkdir(parents=True)
    for i,row in enumerate(rows):
        if row['weld_id'] not in positives:continue
        selected,path=positives[row['weld_id']];name=f'paths/snapshot_retreat_{i:02d}.json'
        (out/name).parent.mkdir(parents=True,exist_ok=True);(out/name).write_bytes(path.read_bytes())
        selected['best'].update(path_file=name,path_sha256=digest(out/name));rows[i]=selected
    codes={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)), 'experimental/welding_retreat_transfer_v1/merge_retreat_inventory_v19.py':digest(ROOT/'experimental/welding_retreat_transfer_v1/merge_retreat_inventory_v19.py')}
    for name,sha in {**sources,**codes}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Source changed during snapshot')
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(distance_mm=2000.,snapshot_only=True,live_search_complete=False,finalized_positive_reports=picked))
    summary=dict(complete=True,inventory_snapshot_complete=True,live_search_complete=False,points=21,passed=len(positives),newly_snapshotted_points=len(picked),required_straight_distance_mm=2000.,wall_s=time.perf_counter()-started)
    atomic(out/'manifest.json',manifest);atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,output_sha256={str(p.relative_to(out)):digest(p) for p in (out/'paths').glob('*.json')},scope='Finished inventory snapshot of finalized positive paths; ongoing multi-angle search not declared complete. Remaining negative rows retain sealed earlier finite evidence. New positives require independent cold audit and actual15axis transfers.'))
    print('EARLY_COMPLETED_POSITIVE_SNAPSHOT',summary,flush=True)


if __name__=='__main__':main()
