"""Seal known route fragments without asserting completion of the live run."""
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'welding_tool_exit_v1'))
from run_geometry import atomic, digest

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source,out=args.source.resolve(),args.out.resolve()
    if out.exists():raise FileExistsError('Use a fresh snapshot')
    nodes=json.loads((source/'nodes.json').read_text())
    retreat=ROOT/'build/welding_retreat_200mm_diagnostic_v1'
    order=[1,3,0,7,2,5,8,6,4]
    out.mkdir();(out/'paths').mkdir()
    shutil.copy2(source/'nodes.json',out/'nodes.json')
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['code_sha256'][str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    manifest['source_sha256'][str((source/'manifest.json').relative_to(ROOT))]=digest(source/'manifest.json')
    edges=[];selected=[];missing=[]
    for a,b in zip(order,order[1:]):
        file=source/f'edge_{a:02d}_{b:02d}.json'
        if not file.exists():missing.append([a,b]);continue
        edge=json.loads(file.read_text());edges.append(edge)
        shutil.copy2(file,out/file.name)
        if edge['passed']:
            if digest(source/edge['path_file'])!=edge['path_sha256']:
                raise RuntimeError('Source path changed')
            shutil.copy2(source/edge['path_file'],out/edge['path_file'])
            selected.append(edge)
        else:missing.append([a,b])
    outputs={str(p.relative_to(out)):digest(p) for p in list(out.glob('edge_*.json'))+
             list((out/'paths').glob('*.json'))+[out/'nodes.json']}
    report=dict(manifest=manifest,summary=dict(complete=True,experiment_finished=False,
        full_route_planned=False,all_retained_nodes_route_planned=False,all_required_points_route_planned=False,
        required_retreat_points=21,retained_retreat_nodes=9,expected_edges=8,
        selected_passed_edges=len(selected),wall_s=0.,snapshot_only=True),
        **nodes,order=order,order_weld_ids=[nodes['nodes'][i]['weld_id'] for i in order],
        selected_edges=selected,all_edges=edges,missing_route_edges=missing,output_sha256=outputs,
        scope='Frozen diagnostic fragments only. Live planning may continue; no complete route claimed.')
    atomic(out/'report.json',report)
    print('PARTIAL_SNAPSHOT',len(selected),'MISSING',missing)


if __name__=='__main__':main()
