"""Run complete-action posture search and fresh RRT transfers under one timer."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--directions', type=int, default=64)
    parser.add_argument('--poses', type=int, default=96)
    parser.add_argument('--configurations', type=int, default=4)
    parser.add_argument('--point-seconds', type=float, default=240.)
    parser.add_argument('--angle-seconds',type=float,default=20.)
    parser.add_argument('--transfer-iterations', type=int, default=200)
    parser.add_argument('--edge-seconds',type=float,default=12.)
    parser.add_argument('--edge-attempts',type=int,default=2)
    parser.add_argument('--reselect-seconds',type=float,default=45.)
    parser.add_argument('--reselect-per-edge',type=int,default=2)
    parser.add_argument('--reorder-first-rounds',type=int,default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        parser.error('--out must be inside the welding project root')
    out.mkdir(parents=True, exist_ok=False)
    retreat, transfer = out/'joint_actions', out/'transfers'
    joint_cmd = [sys.executable, '-u', str(HERE/'run_joint_weld_retreat_cold300.py'),
                 '--out', str(retreat)]
    for key in ('workers', 'directions', 'poses', 'configurations', 'point_seconds', 'angle_seconds'):
        joint_cmd.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
    transfer_cmd = [sys.executable, '-u', str(HERE/'run_transfer_limited_cold300.py'),
                    '--retreat-out', str(retreat), '--retreat-online',
                    '--out', str(transfer), '--workers', str(args.workers),
                    '--iterations', str(args.transfer_iterations),
                    '--edge-seconds',str(args.edge_seconds),'--edge-attempts',str(args.edge_attempts),
                    '--reselect-seconds',str(args.reselect_seconds),'--reselect-per-edge',str(args.reselect_per_edge),'--reorder-first-rounds',str(args.reorder_first_rounds)]
    stages = []
    complete = False
    for name, cmd, folder in [('joint_actions', joint_cmd, retreat),
                              ('transfers', transfer_cmd, transfer)]:
        stage_start = time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        stages.append(dict(stage=name, command=cmd, exit_code=result.returncode,
                           command_wall_s=time.perf_counter()-stage_start))
        report_path = folder/'report.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        if report:
            stages[-1].update(summary=report['summary'], report_sha256=digest(report_path))
        if result.returncode:
            break
        if report is None:
            stages[-1]['failure'] = 'successful_process_missing_report'
            break
        if name == 'transfers':
            complete = bool(report['summary']['all_required_points_route_planned'])
    report = dict(complete=complete, command_wall_s=time.perf_counter()-started,
                  stages=stages, prior_pose_hints=False,
                  independent_cold_performed=False, wrapper_sha256=digest(Path(__file__)),
                  scope='Outer elapsed time includes both subprocesses, model loads, input checks, '
                        'geometry screen, failed posture/retreat attempts, ordering and fresh '
                        'RRT transfers and failed-edge fresh posture reselection. No historical pose hints/configuration/path seeds. Finite cooperative budgets. No extra cold acceptance.')
    (out/'pipeline_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
