"""Seal finalized actual route edges for cold checking while search continues."""
import argparse
import json
import time
from pathlib import Path
from merge_retreat_inventory_v19 import atomic, digest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--retreat-out', type=Path, required=True)
    parser.add_argument('--order', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, retreat, out = args.source.resolve(), args.retreat_out.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh fragment output required')
    started = time.perf_counter()
    node_path = source/'nodes.json'
    nodes = json.loads(node_path.read_text())
    n = len(nodes['nodes'])
    if len(args.order) != n or set(args.order) != set(range(n)):
        raise ValueError('Complete inventory order required, including unconnected nodes')
    parent = json.loads((retreat/'report.json').read_text())
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    seals = {**manifest['source_sha256'], **manifest['code_sha256']}
    for path in [manifest_path, node_path, retreat/'report.json']:
        seals[str(path.relative_to(ROOT))] = digest(path)
    edges, selected, missing = [], [], []
    payloads = {'nodes.json': node_path.read_bytes()}
    for a, b in zip(args.order, args.order[1:]):
        edge_path = source/f'edge_{a:02d}_{b:02d}.json'
        if not edge_path.exists():
            missing.append(dict(from_index=a, to_index=b, reason='not_finalized'))
            continue
        edge = json.loads(edge_path.read_text())
        if (edge['from_index'] != a or edge['to_index'] != b or
                edge['from_weld_id'] != nodes['nodes'][a]['weld_id'] or
                edge['to_weld_id'] != nodes['nodes'][b]['weld_id']):
            raise RuntimeError('Finalized edge identity differs from actual nodes')
        edges.append(edge)
        seals[str(edge_path.relative_to(ROOT))] = digest(edge_path)
        payloads[edge_path.name] = edge_path.read_bytes()
        if not edge['passed']:
            missing.append(dict(from_index=a, to_index=b, reason=edge['reason']))
            continue
        path = source/edge['path_file']
        if digest(path) != edge['path_sha256']:
            raise RuntimeError('Finalized path hash differs')
        data = json.loads(path.read_text())
        if (data['robot_ids'] != nodes['robot_ids'] or data['start_state'] != nodes['nodes'][a]['state'] or
                data['end_state'] != nodes['nodes'][b]['state']):
            raise RuntimeError('Finalized path physical endpoints differ')
        seals[str(path.relative_to(ROOT))] = edge['path_sha256']
        payloads[edge['path_file']] = path.read_bytes()
        selected.append(edge)
    if not selected:
        raise RuntimeError('No finalized actual path to cold-check')
    for name, sha in seals.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Source changed before snapshot '+name)
    manifest['source_sha256'] = seals
    manifest['code_sha256'] = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    manifest['settings'] = dict(manifest['settings'], snapshot_only=True, live_search_complete=False,
                                 complete_order_preserved=args.order)
    out.mkdir(parents=True)
    for name, data in payloads.items():
        path = out/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name, sha in seals.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Evidence changed during snapshot '+name)
    summary = dict(complete=True, inventory_snapshot_complete=True, experiment_finished=False,
                   live_search_complete=False, snapshot_only=True, full_route_planned=False,
                   all_retained_nodes_route_planned=False, all_required_points_route_planned=False,
                   required_retreat_points=parent['summary']['points'], retained_retreat_nodes=n,
                   cold_verified_retreat_nodes=n, expected_edges=max(0, n-1), selected_passed_edges=len(selected),
                   tested_edges=len(edges), passed_edges=len(selected), wall_s=time.perf_counter()-started)
    outputs = {name: digest(out/name) for name in payloads}
    atomic(out/'manifest.json', manifest)
    atomic(out/'report.json', dict(manifest=manifest, summary=summary, **nodes, order=args.order,
           order_weld_ids=[nodes['nodes'][i]['weld_id'] for i in args.order], selected_edges=selected,
           all_edges=edges, missing_route_edges=missing, output_sha256=outputs,
           scope='Frozen byte-identical finalized actual transfer fragments with all retained nodes and missing edges preserved. '
           'Live search continues; no connected subset/full-route completion claimed. Every selected edge requires independent cold check.'))
    print('SEALED_ACTUAL_TRANSFER_FRAGMENTS', summary, flush=True)


if __name__ == '__main__':
    main()
