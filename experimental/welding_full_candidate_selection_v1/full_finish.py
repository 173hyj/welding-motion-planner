"""Reserve final stage rounds for scheduling already certified transfer graphs.

All candidate vertices remain present. Earlier rounds explore optimistic edges;
final rounds exclude unknown edges using a private shadow of the record map.
This is bounded search and carries no claim of global optimality.
"""
import argparse
import json
from pathlib import Path
import sys


def known_graph_builder(build_graph, iterations=6, known_rounds=2):
    """Wrap the frozen builder; call 1 per stage is its all-node prebuild.

    Actual stage iteration r corresponds to call r+1. With iterations=6 and
    known_rounds=2, calls 1..5 retain optimistic edges and calls >=6 exclude
    unknown edges. The original records and candidate layers are never mutated.
    Exposes calls and first_known_round maps for report provenance.
    """
    if (isinstance(iterations, bool) or not isinstance(iterations, int) or
            isinstance(known_rounds, bool) or not isinstance(known_rounds, int) or
            not 1 <= known_rounds < iterations):
        raise ValueError('require integer 1 <= known_rounds < iterations; reserve exploration')
    calls, first_known_round = {}, {}

    def wrapped(task_layers, nodes, records, arm_orders, starts, goals, stage):
        calls[stage] = calls.get(stage, 0) + 1
        call = calls[stage]
        if call < iterations - known_rounds + 2:
            return build_graph(task_layers, nodes, records, arm_orders, starts, goals, stage)
        # Obtain the complete potential key set from the original builder; the
        # all-node topology does not depend on which edges have been attempted.
        result = build_graph(task_layers, nodes, records, arm_orders, starts, goals, stage)
        shadow = dict(records)
        for key in result[-1]:
            if key not in shadow:
                shadow[key] = {'passed': False}
        first_known_round.setdefault(stage, call - 1)
        return build_graph(task_layers, nodes, shadow, arm_orders, starts, goals, stage)

    wrapped.calls = calls
    wrapped.first_known_round = first_known_round
    return wrapped


def main():
    from . import upgraded
    from . import optimize as base
    from experimental.welding_candidate_selection_v1.pool import atomic
    from experimental.welding_candidate_selection_v1.run import verify_bindings

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--known-rounds', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--out', type=Path)
    args, _ = parser.parse_known_args()
    if args.out is None:
        # Let the underlying entrypoint explain its complete required arguments.
        if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
            upgraded.main()
            return
        parser.error('--out is required')
    original_builder = base.build_graph
    wrapped = known_graph_builder(original_builder, args.iterations, args.known_rounds)
    # Remove only this wrapper's option; preserve the upgraded entrypoint's args.
    forwarded = []
    iterator = iter(sys.argv[1:])
    for value in iterator:
        if value == '--known-rounds':
            next(iterator)
        elif value.startswith('--known-rounds='):
            continue
        else:
            forwarded.append(value)
    if not any(value == '--iterations' or value.startswith('--iterations=') for value in forwarded):
        forwarded.extend(['--iterations', str(args.iterations)])
    original_argv = sys.argv
    try:
        base.build_graph = wrapped
        sys.argv = [original_argv[0], *forwarded]
        upgraded.main()
    finally:
        base.build_graph = original_builder
        sys.argv = original_argv
    report_file = args.out / 'report.json'
    report = json.loads(report_file.read_text())
    verify_bindings(report['manifest']['bindings'])
    report['known_graph_finish'] = dict(
        iterations=args.iterations, known_rounds=args.known_rounds,
        exploration_rounds=args.iterations - args.known_rounds,
        first_known_round_by_stage=wrapped.first_known_round,
        build_calls_by_stage=wrapped.calls,
        all_vertices_preserved=True, actual_records_unchanged_by_wrapper=True,
        policy='Final rounds exclude only unknown edges; failed schedules ban complete sequences in the frozen optimizer.')
    report['manifest']['settings']['known_rounds'] = args.known_rounds
    atomic(report_file, report)


if __name__ == '__main__':
    main()
