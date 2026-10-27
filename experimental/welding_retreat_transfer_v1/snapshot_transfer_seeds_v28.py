"""Reuse only independently certified edges whose full endpoints are unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap_certified_edges(target_nodes, report, cold):
    by_id = {row['weld_id']: i for i, row in enumerate(target_nodes)}
    if len(by_id) != len(target_nodes):
        raise ValueError('Ambiguous target weld identities')
    source_nodes = report['nodes']
    if len({row['weld_id'] for row in source_nodes}) != len(source_nodes):
        raise ValueError('Ambiguous source weld identities')
    checks = {(row['from_weld_id'], row['to_weld_id']): row for row in cold['transfer_rows']}
    if len(checks) != len(cold['transfer_rows']):
        raise ValueError('Ambiguous cold edge identities')
    retained = []
    for edge in report['selected_edges']:
        a, b = edge['from_weld_id'], edge['to_weld_id']
        check = checks.get((a, b), {})
        if not edge['passed'] or not check.get('passed') or check.get('planned_path_sha256') != edge['path_sha256']:
            continue
        if a not in by_id or b not in by_id:
            continue
        old_a, old_b = source_nodes[edge['from_index']], source_nodes[edge['to_index']]
        if old_a['weld_id'] != a or old_b['weld_id'] != b:
            raise ValueError('Source edge index differs from weld identity')
        i, j = by_id[a], by_id[b]
        if target_nodes[i]['state'] != old_a['state'] or target_nodes[j]['state'] != old_b['state']:
            continue
        retained.append(dict(from_index=i, to_index=j, source_path=edge['path_file'],
                             source_path_sha256=edge['path_sha256']))
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retreat-out', type=Path, required=True)
    parser.add_argument('--retreat-cold', type=Path, required=True)
    parser.add_argument('--source', nargs=2, action='append', metavar=('TRANSFER_DIR', 'COLD_JSON'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from run_transfer_v6 import cold_retreat_ids
    from world_bridge_v2 import node_state
    target = args.retreat_out.resolve()
    report_path = target/'report.json'
    retreat = json.loads(report_path.read_text())
    retreat_cold_path = args.retreat_cold.resolve()
    retreat_cold = json.loads(retreat_cold_path.read_text())
    accepted = cold_retreat_ids(retreat['rows'], retreat_cold, digest(report_path))
    sources = {}

    def seal(path, expected=None):
        path = Path(path).resolve()
        name = str(path.relative_to(ROOT))
        value = digest(path)
        if expected is not None and value != expected:
            raise RuntimeError('Evidence seal changed: '+name)
        if name in sources and sources[name] != value:
            raise RuntimeError('Conflicting evidence identity: '+name)
        sources[name] = value

    def seal_report(directory, value):
        seal(directory/'report.json')
        for name, sha in {**value['manifest']['source_sha256'], **value['manifest']['code_sha256']}.items():
            seal(ROOT/name, sha)
        for name, sha in value['output_sha256'].items():
            seal(directory/name, sha)

    def seal_cold(path, value):
        seal(path)
        for name, sha in value['verification_code_sha256'].items():
            seal(ROOT/name, sha)

    seal_report(target, retreat)
    seal_cold(retreat_cold_path, retreat_cold)
    seal(Path(__file__))
    robot_ids = sorted({r['initial']['robot_id'] for r in retreat['rows']} |
                       {r['initial']['other_robot_id'] for r in retreat['rows']})
    nodes = []
    for row in retreat['rows']:
        if not row['best'] or row['weld_id'] not in accepted:
            continue
        path = target/row['best']['path_file']
        data = json.loads(path.read_text())
        nodes.append(dict(weld_id=row['weld_id'], retreat_path_file=str(path),
                          retreat_path_sha256=row['best']['path_sha256'],
                          state=node_state(data, robot_ids).tolist()))
    edges = []
    provenance = []
    seen = set()
    for directory_arg, cold_arg in args.source:
        directory, cold_path = Path(directory_arg).resolve(), Path(cold_arg).resolve()
        transfer_path = directory/'report.json'
        transfer = json.loads(transfer_path.read_text())
        cold = json.loads(cold_path.read_text())
        if cold['transfer_report_sha256'] != digest(transfer_path):
            raise RuntimeError('Cold proof belongs to a different transfer report')
        if transfer['robot_ids'] != robot_ids:
            raise RuntimeError('Robot state axis order changed')
        seal_report(directory, transfer)
        seal_cold(cold_path, cold)
        for entry in remap_certified_edges(nodes, transfer, cold):
            pair = (entry['from_index'], entry['to_index'])
            if pair in seen or pair[::-1] in seen:
                continue
            path = directory/entry['source_path']
            value = json.loads(path.read_text())
            if value['robot_ids'] != robot_ids or value['start_state'] != nodes[pair[0]]['state'] or value['end_state'] != nodes[pair[1]]['state']:
                raise RuntimeError('Certified path full endpoint identity differs')
            if (not np.allclose(value['frames'][0]['state'], value['start_state'], atol=1e-12, rtol=0) or
                not np.allclose(value['frames'][-1]['state'], value['end_state'], atol=1e-12, rtol=0)):
                raise RuntimeError('Certified path saved endpoints differ')
            seal(path, entry['source_path_sha256'])
            entry['source_path'] = str(path.relative_to(ROOT))
            edges.append(entry)
            seen.add(pair)
            provenance.append(dict(from_weld_id=nodes[pair[0]]['weld_id'], to_weld_id=nodes[pair[1]]['weld_id'],
                                   cold_path=str(cold_path.relative_to(ROOT)), source_report=str(transfer_path.relative_to(ROOT))))
    for name, sha in sources.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError('Evidence changed while snapshotting: '+name)
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError('Use fresh seed output')
    result = dict(nodes=nodes, edges=edges, source_sha256=sources, provenance=provenance,
                  scope='Only cold-passed edges with byte-identical physical endpoint states are reused; remapped by weld identity. '
                        'Transfer planner may reuse static paths in reverse, then independently cold-check the final selected route.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    temporary.replace(output)
    print('COLD_CERTIFIED_TRANSFER_SEED', dict(nodes=len(nodes), edges=len(edges), seals=len(sources)), flush=True)


if __name__ == '__main__':
    main()
