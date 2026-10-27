"""Cold source geometry and independent XML-FK audit of retreat/transfer evidence."""
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'welding_variable_open_v1')]
from variable_world import VariableToolWorld
from run_initial import assert_frozen_geometry_cache
from run_geometry import atomic, digest, initialize_worker
from world_bridge import SourceBridge, node_state
import verify_initial

_ENGINE = None
_RETREAT_VERIFY = None
_FK = None
_JOINTS = None
_TCP = None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_tcp(rid, q):
    return _ENGINE.world.models[rid].base@_FK.flange_source(q, _JOINTS)@_TCP[rid]


def verify_retreat(row, weld, out, source_out, required):
    path = json.loads((source_out/row['best']['path_file']).read_text())
    initial = row['initial']
    if (path.get('fixed_distance_mm') != required or path['length_mm'] != required or
        path['other_robot_id'] != initial['other_robot_id'] or
        path['q_end_rad'] != path['frames'][-1]['q_rad']):
        raise RuntimeError('Fixed-distance retreat identity mismatch')
    result = _RETREAT_VERIFY.verify_point(row, weld, out, source_out)
    result['required_distance_mm'] = required
    result['passed'] = bool(result['passed'] and result['final_travel_mm'] >= required-.05)
    other = initial['other_robot_id']
    park = np.asarray(initial['other_q_rad'])
    result['other_arm_source_fk_error'] = float(np.max(abs(independent_tcp(other, park)-_ENGINE.world.models[other].tcp(park))))
    result['passed'] = result['passed'] and result['other_arm_source_fk_error'] <= 1e-6
    atomic(out/(row['weld_id']+'.json'), result)
    return result


def verify_transfer(edge, report, out, source_out):
    started = time.perf_counter()
    path = json.loads((source_out/edge['path_file']).read_text())
    if digest(source_out/edge['path_file']) != edge['path_sha256']:
        raise RuntimeError('Transfer path changed')
    a = report['nodes'][edge['from_index']]['state']
    b = report['nodes'][edge['to_index']]['state']
    if (path['start_state'] != a or path['end_state'] != b or
        path['frames'][0]['state'] != a or not np.allclose(path['frames'][-1]['state'], b, atol=1e-10, rtol=0)):
        raise RuntimeError('Transfer endpoint or stitching identity mismatch')
    bridge = SourceBridge(_ENGINE)
    if path['robot_ids'] != bridge.robot_ids:
        raise RuntimeError('Robot identity mismatch')
    result = dict(from_weld_id=edge['from_weld_id'], to_weld_id=edge['to_weld_id'],
                  planned_path_sha256=edge['path_sha256'], passed=True,
                  samples=0, maximum_source_fk_error=0., maximum_saved_tcp_error_mm=0.)
    frames = path['frames']
    previous = None
    for frame in frames:
        state = np.asarray(frame['state'])
        if previous is not None and np.any(abs(state-previous) > bridge.resolutions+1e-8):
            raise RuntimeError('Saved transfer edge exceeds interpolation resolution')
        for i, rid in enumerate(bridge.robot_ids):
            saved = frame['robots'][rid]
            if not np.allclose(saved['q_rad'], state[6*i:6*i+6], atol=1e-12, rtol=0) or saved['driver_mm'] != state[13+i]:
                raise RuntimeError('Frame state metadata mismatch')
            actual = independent_tcp(rid, state[6*i:6*i+6])
            difference = np.max(abs(actual-bridge.world.models[rid].tcp(state[6*i:6*i+6])))
            result['maximum_source_fk_error'] = max(result['maximum_source_fk_error'], float(difference))
            saved_error = np.linalg.norm(actual[:3, 3]-saved['tcp_xyz_mm'])
            result['maximum_saved_tcp_error_mm'] = max(result['maximum_saved_tcp_error_mm'], float(saved_error))
        if abs(frame['positioner_deg']-np.rad2deg(state[12])) > 1e-8:
            raise RuntimeError('External-axis metadata mismatch')
        samples = [state] if previous is None else [(previous+state)/2, state]
        for sample in samples:
            result['samples'] += 1
            for i, rid in enumerate(bridge.robot_ids):
                q = sample[6*i:6*i+6]
                difference = float(np.max(abs(independent_tcp(rid, q)-bridge.world.models[rid].tcp(q))))
                result['maximum_source_fk_error'] = max(result['maximum_source_fk_error'], difference)
            probe = bridge.probe(sample)
            if not probe['passed']:
                result.update(passed=False, failure=dict(state=sample.tolist(), **probe))
                break
        if not result['passed']:
            break
        previous = state
    result['passed'] = bool(result['passed'] and result['maximum_source_fk_error'] <= 1e-6 and
                            result['maximum_saved_tcp_error_mm'] <= .000001)
    result['elapsed_s'] = time.perf_counter()-started
    atomic(out/f'edge_{edge["from_index"]:02d}_{edge["to_index"]:02d}.json', result)
    print('TRANSFER_COLD', edge['from_index'], edge['to_index'], 'PASS', result['passed'], flush=True)
    return result


def check_report(source_out):
    report_path = source_out/'report.json'
    report = json.loads(report_path.read_text())
    if not report['summary']['complete']:
        raise RuntimeError('Finished experiment required')
    for section in ('source_sha256', 'code_sha256'):
        for name, value in report['manifest'][section].items():
            if digest(ROOT/name) != value:
                raise RuntimeError('Frozen inputs changed: '+name)
    for name, value in report['output_sha256'].items():
        if digest(source_out/name) != value:
            raise RuntimeError('Output changed: '+name)
    return report, digest(report_path)


def main():
    global _ENGINE, _RETREAT_VERIFY, _FK, _JOINTS, _TCP
    parser = argparse.ArgumentParser()
    parser.add_argument('--retreat-out', type=Path, required=True)
    parser.add_argument('--transfer-out', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1–4 workers')
    retreat_out, out = args.retreat_out.resolve(), args.out.resolve()
    retreat, retreat_sha = check_report(retreat_out)
    transfer, transfer_sha = None, None
    if args.transfer_out:
        transfer_out = args.transfer_out.resolve()
        transfer, transfer_sha = check_report(transfer_out)
        input_key = str((retreat_out/'report.json').relative_to(ROOT))
        if transfer['manifest']['source_sha256'][input_key] != retreat_sha:
            raise RuntimeError('Transfer is based on another retreat report')
    if transfer:
        parent_rows = {r['weld_id']: r for r in retreat['rows']}
        n = len(transfer['nodes'])
        order = transfer['order']
        if order is not None and sorted(order) != list(range(n)):
            raise RuntimeError('Visit order must contain each retained node exactly once')
        for node in transfer['nodes']:
            row = parent_rows[node['weld_id']]
            if not row['best'] or node['retreat_path_sha256'] != row['best']['path_sha256']:
                raise RuntimeError('Transfer node uses another retreat path')
            path = json.loads((retreat_out/row['best']['path_file']).read_text())
            expected = node_state(path, transfer['robot_ids'])
            if not np.allclose(node['state'], expected, atol=1e-12, rtol=0):
                raise RuntimeError('Transfer node differs from actual retreat endpoint')
        route_pairs = list(zip(order or [], (order or [])[1:]))
        selected_pairs = [(e['from_index'],e['to_index']) for e in transfer['selected_edges']]
        if len(set(selected_pairs)) != len(selected_pairs) or any(p not in route_pairs for p in selected_pairs):
            raise RuntimeError('Selected transfers do not belong to the claimed visit route')
        if transfer['summary']['all_retained_nodes_route_planned'] and selected_pairs != route_pairs:
            raise RuntimeError('Claimed full retained route has missing or reordered transfers')
    selected_retreats = [r for r in retreat['rows'] if r['best']]
    selected_edges = transfer['selected_edges'] if transfer else []
    code_names = [str(Path(__file__).relative_to(ROOT)),
                  'experimental/welding_retreat_transfer_v1/world_bridge.py',
                  'experimental/welding_retreat_transfer_v1/state_space.py',
                  'experimental/welding_variable_open_v1/verify.py',
                  'experimental/welding_tool_exit_v1/verify_initial.py',
                  'scripts/audit_contact_release_v17.py']
    seal = {n: digest(ROOT/n) for n in code_names}
    out.mkdir(parents=True, exist_ok=True)
    vo = out/'retreats'
    vo.mkdir(exist_ok=True)
    eo = out/'transfers'
    eo.mkdir(exist_ok=True)
    started = time.perf_counter()
    retreat_rows, transfer_rows = [], []
    if selected_retreats or selected_edges:
        assert_frozen_geometry_cache()
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        welds = {w['id']: w for w in scene['welds']}
        _ENGINE = verify_initial.fresh_source_world(scene)
        _ENGINE.__class__ = VariableToolWorld
        _FK = load_module('raw_fk_transfer', ROOT/'scripts/audit_contact_release_v17.py')
        raw = json.loads((ROOT/'build/robot.json').read_text())
        _JOINTS = _FK.joints_from_raw(raw)[:6]
        _TCP = {a['id']: _FK.source_tcp_home(raw, a) for a in scene['assets'] if a['kind']=='robot'}
        _RETREAT_VERIFY = load_module('retreat_source_verify', ROOT/'experimental/welding_variable_open_v1/verify.py')
        _RETREAT_VERIFY._ENGINE, _RETREAT_VERIFY._FK = _ENGINE, _FK
        _RETREAT_VERIFY._JOINTS, _RETREAT_VERIFY._TCP = _JOINTS, _TCP
        required = retreat['summary']['required_straight_distance_mm']
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
            fs = [pool.submit(verify_retreat, r, welds[r['weld_id']], vo, retreat_out, required)
                  for r in selected_retreats]
            retreat_rows = [f.result() for f in concurrent.futures.as_completed(fs)]
            fs = [pool.submit(verify_transfer, e, transfer, eo, transfer_out) for e in selected_edges]
            transfer_rows = [f.result() for f in concurrent.futures.as_completed(fs)]
    if check_report(retreat_out)[1] != retreat_sha or (transfer and check_report(transfer_out)[1] != transfer_sha):
        raise RuntimeError('Report changed during cold audit')
    for name, value in seal.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Verification code changed during audit')
    all_passed = bool(retreat_rows) and all(r['passed'] for r in retreat_rows+transfer_rows)
    retained_route = bool(transfer and transfer['summary']['all_retained_nodes_route_planned'] and all_passed and
                          len(transfer_rows) == max(0,len(retreat_rows)-1) and
                          transfer['summary']['retained_retreat_nodes'] == len(retreat_rows))
    full_route = bool(retained_route and len(retreat_rows) == retreat['summary']['points'])
    result = dict(retreat_report_sha256=retreat_sha, transfer_report_sha256=transfer_sha,
                  verification_code_sha256=seal, retreat_rows=retreat_rows, transfer_rows=transfer_rows,
                  summary=dict(selected_retreats=len(retreat_rows),
                    passed_retreats=sum(r['passed'] for r in retreat_rows),
                    selected_transfers=len(transfer_rows), passed_transfers=sum(r['passed'] for r in transfer_rows),
                    all_selected_passed=all_passed, retained_subset_route_verified=retained_route,
                    full_route_verified=full_route,
                    wall_s=time.perf_counter()-started),
                  scope='Fresh raw fixture triangles, independent source-XML TCP FK and saved-edge midpoint collision rechecks. Assembly interfaces and source mounting exemptions retained. Sampled, not continuous all-system certification.')
    atomic(out/'cold_verification.json', result)
    print('RETREAT_TRANSFER_COLD_SUMMARY', result['summary'], flush=True)
    if (retreat_rows or transfer_rows) and not all_passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
