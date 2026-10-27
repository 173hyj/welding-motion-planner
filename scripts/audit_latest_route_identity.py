"""Read-only file/route identity audit; does not run collision checks."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experimental/welding_retreat_transfer_v1'))
import numpy as np
from integrated_retreat_v117 import online_retreat_ids
from world_bridge_v2 import node_state


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / 'build/welding_rrt_reorder_v158_fresh'
    outer = read(folder / 'pipeline_report.json')
    assert outer['complete'] and not outer['prior_pose_hints']
    assert not outer['independent_cold_performed']
    checked = 0
    for stage in outer['stages']:
        assert stage['exit_code'] == 0
        assert digest(folder / stage['stage'] / 'report.json') == stage['report_sha256']
    assert digest(ROOT / 'experimental/welding_retreat_transfer_v1/run_fresh_pipeline_v158.py') == outer['wrapper_sha256']
    transfer_dir = folder / 'transfers'
    transfer = read(transfer_dir / 'report.json')
    action_dir = ROOT / transfer['final_retreat_out']
    actions = read(action_dir / 'report.json')
    for directory, report in ((action_dir, actions), (transfer_dir, transfer)):
        for group in ('source_sha256', 'code_sha256'):
            for name, value in report['manifest'][group].items():
                assert digest(ROOT / name) == value, name
                checked += 1
        for name, value in report['output_sha256'].items():
            assert digest(directory / name) == value, name
            checked += 1
    accepted = online_retreat_ids(actions, action_dir)
    nodes = transfer['nodes']
    assert len(accepted) == len(nodes) == 21
    assert {n['weld_id'] for n in nodes} == accepted
    assert transfer['summary']['all_required_points_route_planned']
    order = transfer['order']
    assert len(order) == len(set(order)) == len(nodes)
    assert set(order) == set(range(len(nodes)))
    edges = transfer['selected_edges']
    assert len(edges) == 20
    assert [(e['from_index'], e['to_index']) for e in edges] == list(zip(order, order[1:]))
    max_error = 0.
    frames = 0
    for node in nodes:
        path = Path(node['retreat_path_file'])
        if path.is_absolute():
            if 'build' not in path.parts:
                raise ValueError('Unexpected external node path')
            path = ROOT.joinpath(*path.parts[path.parts.index('build'):])
        else:
            path = ROOT / path
        assert digest(path) == node['retreat_path_sha256']
        state = node_state(read(path), transfer['robot_ids'])
        assert np.max(np.abs(state - node['state'])) <= 1e-12
    for edge in edges:
        assert edge['passed']
        path = transfer_dir / edge['path_file']
        assert digest(path) == edge['path_sha256']
        data = read(path)
        a, b = nodes[edge['from_index']], nodes[edge['to_index']]
        assert data['from_weld_id'] == a['weld_id']
        assert data['to_weld_id'] == b['weld_id']
        for x, y in ((data['start_state'], a['state']), (data['end_state'], b['state']),
                     (data['frames'][0]['state'], a['state']), (data['frames'][-1]['state'], b['state'])):
            error = float(np.max(np.abs(np.asarray(x) - y)))
            assert error <= 1e-12
            max_error = max(max_error, error)
        frames += len(data['frames'])
    result = dict(passed=True, nodes=len(nodes), selected_edges=len(edges),
                  selected_frames=frames, checked_hashes=checked,
                  maximum_endpoint_identity_error=max_error,
                  feedback_attempts=len(transfer['feedback_trials']),
                  adopted_replacements=sum(bool(t.get('adopted')) for t in transfer['feedback_trials']),
                  extra_collision_checks_performed=False,
                  pipeline_report_sha256=digest(folder / 'pipeline_report.json'),
                  audit_script_sha256=digest(Path(__file__)))
    (ROOT / 'local_results').mkdir(exist_ok=True)
    (ROOT / 'local_results/latest_route_identity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
