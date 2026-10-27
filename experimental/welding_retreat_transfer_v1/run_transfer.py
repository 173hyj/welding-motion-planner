"""Order retreat nodes and connect full physical cell states with RRT-Connect."""
import argparse
import concurrent.futures
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
from ordering import visit_order, route_cost
from state_space import connect_rrt
from world_bridge import SourceBridge, node_state

_ENGINE = None


def cold_retreat_ids(rows, cold, report_sha):
    if cold.get('retreat_report_sha256') != report_sha:
        raise RuntimeError('Retreat cold audit is for another report')
    audited = {r['weld_id']: r for r in cold.get('retreat_rows', [])}
    accepted = set()
    for row in rows:
        check = audited.get(row['weld_id'])
        if not check:
            continue
        if not row['best'] or check.get('planned_path_sha256') != row['best']['path_sha256']:
            raise RuntimeError('Retreat cold path identity mismatch: '+row['weld_id'])
        if check.get('passed'):
            accepted.add(row['weld_id'])
    return accepted


def solve_segment(bridge, a, b, free, seed, iterations):
    if bridge.edge_scaled(a, b):
        return [a, b], dict(reason='direct_edge', iterations=0)
    fixed = sorted(set(range(15))-set(free))
    if not np.array_equal(a[fixed], b[fixed]):
        return None, dict(reason='fixed_axes_differ', iterations=0)
    low, high = bridge.low/bridge.scale, bridge.high/bridge.scale
    def unpack(part):
        full = a.copy()
        full[free] = part
        return full
    path, stats = connect_rrt(a[free], b[free], low[free], high[free],
                             lambda x,y: bridge.edge_scaled(unpack(x), unpack(y)),
                             seed=seed, max_iterations=iterations)
    return ([unpack(p) for p in path] if path is not None else None), stats


def connect_via_park(bridge, a, b, seed, iterations):
    # A shared source-qualified park avoids changing the external axis next to
    # the product. Every gateway and motion is checked at the actual angle.
    parent = json.loads((ROOT/'build/welding_variable_open_v1/report.json').read_text())
    parks = {}
    for row in parent['rows']:
        s = row['initial']
        parks.setdefault(s['other_robot_id'], np.asarray(s['other_q_rad']))
    current = a.copy()
    path, trials = [a], []
    for phase in ('park_arms', 'close_guns', 'rotate_positioner', 'set_next_openings', 'deploy_arms'):
        actions = []
        if phase in ('park_arms', 'deploy_arms'):
            for i, rid in enumerate(bridge.robot_ids):
                target = current.copy()
                target[6*i:6*i+6] = parks[rid] if phase == 'park_arms' else b[6*i:6*i+6]
                actions.append((target, list(range(6*i, 6*i+6))))
                # Build actions sequentially below, avoiding a stale other-arm state.
        elif phase == 'rotate_positioner':
            target = current.copy(); target[12] = b[12]
            actions = [(target, [12])]
        else:
            target = current.copy(); target[13:] = 0. if phase == 'close_guns' else b[13:]
            actions = [(target, [13, 14])]
        for target, free in actions:
            target = target.copy()
            frozen = sorted(set(range(15))-set(free))
            target[frozen] = current[frozen]
            if np.array_equal(current, target):
                continue
            segment, stats = solve_segment(bridge, current, target, free, seed+len(trials), iterations)
            trials.append(dict(phase=phase, **stats))
            if segment is None:
                return None, trials
            path += segment[1:]
            current = target
    if not np.allclose(current, b, atol=1e-10, rtol=0):
        raise RuntimeError('Park gateway failed to restore exact target axes')
    return path, trials


def transition_job(i, j, nodes, out, seed, iterations):
    started = time.perf_counter()
    bridge = SourceBridge(_ENGINE)
    a = np.asarray(nodes[i]['state'])/bridge.scale
    b = np.asarray(nodes[j]['state'])/bridge.scale
    trials, path = [], None
    if not bridge.valid_scaled(a) or not bridge.valid_scaled(b):
        reason = 'retreat_endpoint_invalid'
    else:
        if bridge.edge_scaled(a, b):
            path = [a, b]
            trials.append(dict(reason='full_state_direct_edge'))
        else:
            path, park_trials = connect_via_park(bridge, a, b, seed, min(iterations, 700))
            trials.append(dict(strategy='sequential_arms_safe_park_rotation', stages=park_trials,
                               passed=path is not None))
            # RRT in joint/angle space while opening values follow a deterministic
            # waypoint schedule; use a closed-gun gateway when those edges are valid.
            closed_a, closed_b = a.copy(), b.copy()
            closed_a[13:] = 0.
            closed_b[13:] = 0.
            if path is None and bridge.edge_scaled(a, closed_a) and bridge.edge_scaled(closed_b, b):
                free = [axis for i in range(2) if not np.array_equal(a[6*i:6*i+6], b[6*i:6*i+6])
                        for axis in range(6*i, 6*i+6)] + ([12] if abs(a[12]-b[12]) > 1e-10 else [])
                middle, stats = solve_segment(bridge, closed_a, closed_b, free, seed, iterations)
                trials.append(dict(strategy='closed_gun_joint_and_positioner_rrt', **stats))
                if middle is not None:
                    path = [a] + middle + [b]
            if path is None:
                # Both openings and external-axis motion are explicit sampled axes.
                path, stats = solve_segment(bridge, a, b, list(range(15)), seed+1, iterations)
                trials.append(dict(strategy='joint_external_axis_and_opening_rrt', **stats))
        reason = 'connected' if path is not None else 'finite_rrt_budget_exhausted'
    result = dict(from_index=i, to_index=j, from_weld_id=nodes[i]['weld_id'],
                  to_weld_id=nodes[j]['weld_id'], passed=path is not None,
                  reason=reason, trials=trials, full_source_checks=bridge.checks,
                  last_failure=bridge.last_failure)
    if path is not None:
        frames = bridge.dense_frames(path)
        states = [p*bridge.scale for p in path]
        metric_length = float(sum(np.linalg.norm(b-a) for a,b in zip(path, path[1:])))
        data = dict(from_weld_id=nodes[i]['weld_id'], to_weld_id=nodes[j]['weld_id'],
                    robot_ids=bridge.robot_ids, state_axes='q_robot0[6],q_robot1[6],positioner_rad,driver0_mm,driver1_mm',
                    start_state=nodes[i]['state'], end_state=nodes[j]['state'],
                    waypoints=[s.tolist() for s in states], frames=frames,
                    interpolation_resolution=bridge.resolutions.tolist(),
                    metric_length=metric_length, source_checks=bridge.checks)
        name = f'paths/transfer_{i:02d}_{j:02d}.json'
        atomic(out/name, data)
        result.update(path_file=name, path_sha256=digest(out/name), metric_length=metric_length,
                      saved_frames=len(frames))
    result['elapsed_s'] = time.perf_counter()-started
    atomic(out/f'edge_{i:02d}_{j:02d}.json', result)
    print('TRANSFER_EDGE', i, j, 'PASS', result['passed'], 'CHECKS', bridge.checks,
          'SECONDS', round(result['elapsed_s'], 2), flush=True)
    return result


def main():
    global _ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--retreat-out', type=Path, required=True)
    parser.add_argument('--retreat-cold', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--ordering-rounds', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.iterations < 1 or args.ordering_rounds < 1:
        raise ValueError('Invalid worker/iteration/round budget')
    source_out, out = args.retreat_out.resolve(), args.out.resolve()
    retreat = json.loads((source_out/'report.json').read_text())
    if not retreat['summary']['complete']:
        raise RuntimeError('Finished straight retreat report required')
    for name, value in retreat['output_sha256'].items():
        if digest(source_out/name) != value:
            raise RuntimeError('Retreat evidence changed')
    retreat_cold_path = args.retreat_cold.resolve()
    retreat_cold = json.loads(retreat_cold_path.read_text())
    accepted_ids = cold_retreat_ids(retreat['rows'], retreat_cold, digest(source_out/'report.json'))
    baseline = {**retreat['manifest']['source_sha256'], **retreat['manifest']['code_sha256']}
    baseline[str((source_out/'report.json').relative_to(ROOT))] = digest(source_out/'report.json')
    baseline[str(retreat_cold_path.relative_to(ROOT))] = digest(retreat_cold_path)
    codes = {str((HERE/n).relative_to(ROOT)): digest(HERE/n) for n in
             ('run_transfer.py', 'state_space.py', 'world_bridge.py', 'ordering.py')}
    for name, value in {**baseline, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Source/code changed: '+name)
    manifest = dict(source_sha256=baseline, code_sha256=codes, settings=dict(
                    iterations=args.iterations, ordering_rounds=args.ordering_rounds,
                    ordering='multistart_nearest_neighbour_plus_open_2opt',
                    q_edge_step_deg=.05, angle_edge_step_deg=.25, driver_step_mm=.5))
    out.mkdir(parents=True, exist_ok=True)
    mp = out/'manifest.json'
    if mp.exists():
        if not args.resume or json.loads(mp.read_text()) != manifest:
            raise RuntimeError('Use new output or unchanged --resume')
    else:
        atomic(mp, manifest)
    robot_ids = sorted({r['initial']['robot_id'] for r in retreat['rows']} |
                       {r['initial']['other_robot_id'] for r in retreat['rows']})
    nodes = []
    for row in retreat['rows']:
        if not row['best'] or row['weld_id'] not in accepted_ids:
            continue
        path = json.loads((source_out/row['best']['path_file']).read_text())
        nodes.append(dict(weld_id=row['weld_id'], retreat_path_file=str(source_out/row['best']['path_file']),
                          retreat_path_sha256=row['best']['path_sha256'],
                          state=node_state(path, robot_ids).tolist()))
    atomic(out/'nodes.json', dict(robot_ids=robot_ids, nodes=nodes))
    n = len(nodes)
    scale = np.r_[np.ones(13), 100., 100.]
    states = np.asarray([p['state'] for p in nodes]).reshape(n, 15)/scale
    costs = np.linalg.norm(states[:, None]-states[None, :], axis=2)
    np.fill_diagonal(costs, np.inf)
    original_costs = costs.copy()
    known, rounds, order = {}, [], []
    if args.resume:
        for p in out.glob('edge_*.json'):
            edge = json.loads(p.read_text())
            if edge['passed'] and digest(out/edge['path_file']) != edge['path_sha256']:
                raise RuntimeError('Saved transfer path changed')
            known[(edge['from_index'], edge['to_index'])] = edge
            costs[edge['from_index'], edge['to_index']] = edge.get('metric_length', np.inf)
    started = time.perf_counter()
    load = 0.
    if n > 1:
        assert_frozen_geometry_cache()
        scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        t = time.perf_counter()
        _ENGINE = VariableToolWorld(scene)
        load = time.perf_counter()-t
        print('TRANSFER_MODEL_LOADED', round(load, 2), flush=True)
    route_complete = n == 1
    for ri in range(args.ordering_rounds):
        order = visit_order(costs)
        if order is None:
            rounds.append(dict(round=ri, status='no_route_in_finite_edge_graph'))
            break
        edges = list(zip(order, order[1:]))
        missing = [(i,j) for i,j in edges if (i,j) not in known]
        rounds.append(dict(round=ri, order=order, estimated_cost=route_cost(order, costs),
                           newly_requested_edges=missing))
        print('TRANSFER_ORDER', ri, [nodes[i]['weld_id'] for i in order], flush=True)
        if missing:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                    mp_context=multiprocessing.get_context('fork'), initializer=initialize_worker) as pool:
                fs = [pool.submit(transition_job, i, j, nodes, out, 20261009+100*i+j,
                                   args.iterations) for i,j in missing]
                for f in concurrent.futures.as_completed(fs):
                    edge = f.result()
                    i,j = edge['from_index'], edge['to_index']
                    known[(i,j)] = edge
                    costs[i,j] = edge.get('metric_length', np.inf)
                    atomic(out/'progress.json', dict(round=ri, requested_edges=len(missing),
                          known_edges=len(known), passed_edges=sum(e['passed'] for e in known.values()),
                          elapsed_s=time.perf_counter()-started))
        if all(known[(i,j)]['passed'] for i,j in edges):
            route_complete = bool(n)
            break
    selected = [known[(i,j)] for i,j in zip(order or [], (order or [])[1:])
                if (i,j) in known and known[(i,j)]['passed']]
    output = {str(p.relative_to(out)): digest(p) for p in list(out.glob('edge_*.json'))+
              list((out/'paths').glob('*.json'))+[out/'nodes.json']}
    report = dict(manifest=manifest, summary=dict(complete=True,
                  experiment_finished=True,
                  full_route_planned=bool(route_complete and n == retreat['summary']['points']),
                  all_retained_nodes_route_planned=route_complete,
                  all_required_points_route_planned=bool(route_complete and n == retreat['summary']['points']),
                  required_retreat_points=retreat['summary']['points'],
                  retained_retreat_nodes=n, cold_verified_retreat_nodes=n, expected_edges=max(0,n-1),
                  selected_passed_edges=len(selected), tested_edges=len(known),
                  passed_edges=sum(e['passed'] for e in known.values()), model_load_s=load,
                  wall_s=time.perf_counter()-started), robot_ids=robot_ids, nodes=nodes,
                  order=order, order_weld_ids=[nodes[i]['weld_id'] for i in order or []],
                  initial_natural_order_estimated_cost=route_cost(list(range(n)), original_costs),
                  selected_edges=selected, all_edges=list(known.values()), rounds=rounds,
                  output_sha256=output,
                  scope='Source-mesh sampled both-arm and gun checks with actual external-axis motion. Open visit heuristic, not globally optimal. Only passing fixed-distance retreat nodes included; source assembly interface exemptions retained.')
    for name, value in {**baseline, **codes}.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Source/code changed during transfer planning')
    atomic(out/'report.json', report)
    print('TRANSFER_SUMMARY', report['summary'], flush=True)


if __name__ == '__main__':
    main()
