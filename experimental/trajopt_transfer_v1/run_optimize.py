"""Independent, fail-closed classic TrajOpt postprocessor for sealed transfers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def refine_seed(path, count):
    """Split longest edges without shortcutting any certified polyline corner."""
    points = [np.array(p, float, copy=True) for p in path]
    while len(points) < count:
        lengths = [np.linalg.norm(b-a) for a, b in zip(points, points[1:])]
        if not lengths or max(lengths) < 1e-12:
            break
        i = int(np.argmax(lengths))
        points.insert(i+1, (points[i]+points[i+1])/2)
    return np.array(points)


def accept_candidate(original, candidate, initial_objective, final_objective, fine_passed):
    if not np.isfinite(candidate).all():
        return False, 'nonfinite_candidate'
    if not np.array_equal(original[[0, -1]], candidate[[0, -1]]):
        return False, 'endpoint_changed'
    if not fine_passed:
        return False, 'original_fine_source_check_failed'
    if path_length(candidate) > path_length(original)+1e-8:
        return False, 'normalized_length_regression'
    if final_objective >= initial_objective-1e-8:
        return False, 'no_objective_improvement'
    return True, 'objective_improved_and_original_fine_source_certified'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def load_source(report_path):
    report = json.loads(report_path.read_text())
    for group in ('source_sha256', 'code_sha256'):
        for name, expected in report['manifest'].get(group, {}).items():
            if digest(ROOT/name) != expected:
                raise RuntimeError('Source identity mismatch: '+name)
    for edge in report['selected_edges']:
        if not edge['passed'] or digest(report_path.parent/edge['path_file']) != edge['path_sha256']:
            raise RuntimeError('Uncertified or changed source edge')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'build/welding_rrt_reorder_v158_fresh/transfers/report.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--edge', help='from_index,to_index; omit for all selected edges')
    parser.add_argument('--knots', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=18)
    parser.add_argument('--margin-mm', type=float, default=1.)
    parser.add_argument('--activation-mm', type=float, default=8.)
    args = parser.parse_args()
    if args.knots < 3 or args.iterations < 1:
        parser.error('knots >= 3 and iterations >= 1 required')
    if args.out.resolve() == args.source.parent.resolve():
        parser.error('output must be independent of original route')
    sys.path[:0] = [str(HERE), str(ROOT), str(HERE.parent/'welding_retreat_transfer_v1'), str(HERE.parent/'welding_variable_open_v1')]
    from solver import optimize, Options, objective
    from proximity import SourceProximity, capture_mesh_bounds
    from variable_world import VariableToolWorld
    from mixed_bridge_v95 import SignedSearchBridge
    from world_bridge_v2 import SourceBridge
    started = time.perf_counter()
    report = load_source(args.source)
    selected = report['selected_edges']
    if args.edge:
        key = tuple(map(int, args.edge.split(',')))
        selected = [e for e in selected if (e['from_index'], e['to_index']) == key]
        if not selected:
            raise ValueError('Requested edge not in selected route')
    engine = None
    mesh_bounds = {}
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for edge in selected:
        tick = time.perf_counter()
        path = args.source.parent/edge['path_file']
        data = json.loads(path.read_text())
        physical = np.array(data['waypoints'], float)
        row = {k: edge[k] for k in ('from_index', 'to_index', 'from_weld_id', 'to_weld_id')}
        row.update(source_path=str(path.resolve()), source_sha256=digest(path), accepted=False, fine_checks=0)
        scale = np.r_[np.ones(13), 100., 100.]
        original = physical/scale
        row['before_length'] = path_length(original)
        print('TRAJOPT_START', edge['from_index'], edge['to_index'], 'CORNERS', len(original), flush=True)
        if len(original) == 2:
            row.update(reason='direct_seed_already_minimum_normalized_length', after_length=row['before_length'], path_file=str(path.resolve()), path_sha256=row['source_sha256'])
        else:
            if engine is None:
                scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
                with capture_mesh_bounds() as mesh_bounds:
                    engine = VariableToolWorld(scene)
                print('TRAJOPT_WORLD_READY', round(time.perf_counter()-tick, 3), flush=True)
            bridge = SignedSearchBridge(engine)
            seed = refine_seed(original, args.knots)
            oracle = SourceProximity(bridge, activation_mm=args.activation_mm, margin_mm=args.margin_mm, mesh_bounds=mesh_bounds)
            def progress(step):
                print('TRAJOPT_ITERATION', edge['from_index'], edge['to_index'], step['iteration'], 'ACCEPTED', step['accepted'], 'RADIUS', round(step['trust_radius'], 6), 'ACTUAL', round(step['actual_reduction'], 6), flush=True)
            options = Options(max_iterations=args.iterations, feasibility_callback=oracle.sparse_feasibility, progress_callback=progress)
            initial_objective = objective(seed, options)
            result = optimize(seed, bridge.low/scale, bridge.high/scale, oracle.oracle, options)
            candidate = result.trajectory
            final_objective = objective(candidate, options)
            row.update(initial_objective=float(initial_objective), final_objective=float(final_objective), solver_status=result.status, solver_history=result.history)
            # Backtrack toward the certified seed if optimization's sparse samples
            # miss a fine-sampled obstruction. Every accepted trial is rechecked.
            raw_candidate = candidate.copy()
            attempts = []
            accepted = False
            for alpha in (1., .5, .25, .125, .0625):
                candidate = seed+alpha*(raw_candidate-seed)
                candidate[[0, -1]] = seed[[0, -1]]
                final_objective = objective(candidate, options)
                print('TRAJOPT_FINE_CHECK', edge['from_index'], edge['to_index'], 'ALPHA', alpha, flush=True)
                # Never dispatch to SearchBridge's coarse proposal sampler.
                fine = all(SourceBridge.edge_scaled(bridge, a, b) for a, b in zip(candidate, candidate[1:]))
                accepted, reason = accept_candidate(original, candidate, initial_objective, final_objective, fine)
                attempts.append(dict(alpha=alpha, fine_passed=fine, reason=reason, candidate_length=path_length(candidate), objective=final_objective, failure=bridge.last_failure if not fine else None))
                if accepted or reason == 'no_objective_improvement':
                    break
            row.update(accepted=accepted, reason=reason, final_objective=final_objective, fine_backtrack_alpha=alpha, fine_attempts=attempts, fine_checks=bridge.checks, fine_passed=fine, after_length=path_length(candidate) if accepted else row['before_length'], candidate_length=path_length(candidate), last_failure=bridge.last_failure if not fine else None, mesh_queries=oracle.mesh_queries)
            if accepted:
                frames = bridge.dense_frames(candidate, phase='transfer')
                actual = candidate*scale
                # Physical endpoint bytes are preserved separately from normalized arithmetic.
                actual[0] = physical[0]; actual[-1] = physical[-1]
                replacement = dict(data, waypoints=actual.tolist(), frames=frames, metric_length=path_length(candidate), source_checks=bridge.checks, acceptance='classic_trajopt_original_fine_source_checked', trajopt_source_sha256=row['source_sha256'])
                outpath = args.out/edge['path_file']
                save(outpath, replacement)
                row.update(path_file=str(outpath.resolve()), path_sha256=digest(outpath), saved_frames=len(frames))
            else:
                row.update(path_file=str(path.resolve()), path_sha256=row['source_sha256'])
        row['elapsed_s'] = time.perf_counter()-tick
        rows.append(row)
        save(args.out/f"edge_{edge['from_index']:02d}_{edge['to_index']:02d}.json", row)
        save(args.out/'optimization_report.json', dict(source_report=str(args.source.resolve()), source_report_sha256=digest(args.source), rows=rows, elapsed_s=time.perf_counter()-started, settings=vars(args) | {'source':str(args.source), 'out':str(args.out)}, scope='offline_geometric_transfer_optimization_no_time_parameterization', continuous_collision_certified=False))
        print('TRAJOPT_END', edge['from_index'], edge['to_index'], row['reason'], 'LENGTH', round(row['before_length'], 6), round(row['after_length'], 6), 'SECONDS', round(row['elapsed_s'], 3), flush=True)
    if len(rows) == len(report['selected_edges']):
        edges = [dict(original_edge, path_file=row['path_file'], path_sha256=row['path_sha256'], metric_length=row['after_length'], trajopt_accepted=row['accepted'], trajopt_reason=row['reason']) for original_edge, row in zip(report['selected_edges'], rows)]
        save(args.out/'optimized_route.json', dict(nodes=report['nodes'], order=report['order'], order_weld_ids=report['order_weld_ids'], robot_ids=report['robot_ids'], final_retreat_out=report['final_retreat_out'], selected_edges=edges, source_report=str(args.source.resolve()), source_report_sha256=digest(args.source), optimization_report_sha256=digest(args.out/'optimization_report.json'), complete=True, acceptance='accepted replacements fine checked; retained edges inherit sealed source acceptance', continuous_collision_certified=False))
    print('TRAJOPT_FINISHED', len(rows), 'ACCEPTED', sum(r['accepted'] for r in rows), flush=True)


if __name__ == '__main__':
    main()
