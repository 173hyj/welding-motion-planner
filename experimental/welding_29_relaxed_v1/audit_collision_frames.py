"""Fresh actual-triangle FCL checks at every saved replay configuration.

This records sampled contact witnesses, not a continuous safety certificate.
It deliberately includes all exported local and transfer frames rather than
reusing the planner's sparse endpoint/edge evidence.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from planner import Planner, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT/'build/welding_29_staged_v2')
    args = parser.parse_args()
    out = args.folder
    path = out/'welding_plan_29_staged_v2.json.gz'
    raw = gzip.decompress(path.read_bytes())
    sha = hashlib.sha256(raw).hexdigest()
    registry = json.loads((out/'audit.json').read_text())
    assert registry['plans'][path.name]['sha256'] == sha
    plan = json.loads(raw)
    planner = Planner(out, plan['summary']['reference_interference_mm'])
    phases, tasks, local_tasks, transfers = {}, {}, {}, {}
    worst = None
    caps = 0
    for i, frame in enumerate(plan['frames']):
        for rid, robot in frame['robots'].items():
            if rid != planner.rid:
                model = planner.world.models[rid]
                planner.world.update(rid, np.array(robot['q_rad']), frame['guns'][model.gun_id]['driver_mm'])
        q = np.array(frame['robots'][planner.rid]['q_rad'])
        driver = frame['guns'][planner.model.gun_id]['driver_mm']
        check = planner.cost.check(q, driver)
        depth = check['observed_max_depth_mm']
        caps += check['contact_caps_reached']
        witness = dict(frame_index=i, time_s=frame['time_s'], phase=frame['phase'],
                       task_index=frame['task_index'], weld_id=frame['weld_id'],
                       observed_max_depth_mm=depth, pairs=check['pairs'])
        if worst is None or depth > worst['observed_max_depth_mm']:
            worst = witness
        classification=(transfers if frame['phase'] in ('transfer','loop_return') else local_tasks)
        for key, mapping in [(frame['phase'], phases), (str(frame['task_index']), tasks),
                             (str(frame['task_index']),classification)]:
            row = mapping.setdefault(key, dict(frame_count=0, frames_above_reference=0,
                                              maximum_observed_depth_mm=0., worst=None))
            row['frame_count'] += 1
            row['frames_above_reference'] += int(depth > planner.cost.reference)
            if row['worst'] is None or depth > row['maximum_observed_depth_mm']:
                row['maximum_observed_depth_mm'] = depth
                row['worst'] = witness
        if i % 500 == 0:
            print(f'FRAME FCL {i+1}/{len(plan["frames"])} max {worst["observed_max_depth_mm"]:.3f} mm', flush=True)
    report = dict(decoded_plan_sha256=sha, saved_frames_checked=len(plan['frames']),
                  unique_fcl_states_checked=planner.cost.count, reference_mm=planner.cost.reference,
                  maximum_observed_depth_mm=worst['observed_max_depth_mm'], worst=worst,
                  phases=phases, tasks=tasks, local_tasks=local_tasks, transfers=transfers,
                  local_tasks_within_reference=sum(r['frames_above_reference']==0 for r in local_tasks.values()),
                  transfer_connections_within_reference=sum(r['frames_above_reference']==0 for r in transfers.values()),
                  summed_capped_pair_observations=caps,
                  tasks_within_reference=sum(r['frames_above_reference']==0 for r in tasks.values()),
                  FCL_rerun=True, continuous_collision_verified=False, system_safe=False,
                  scope='All saved frames, capped at 8 triangle contacts per pair; interpolation and excluded mount/internal gun pairs are not certified')
    assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == sha
    (out/'all_saved_frame_collision_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('phases','tasks','local_tasks','transfers','worst')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
