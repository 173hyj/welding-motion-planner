"""Display the sealed and independently audited candidate selection result."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experimental.dual_arm_parallel_v1 import export_parallel,replay_parallel
from experimental.welding_angle_batches_v1.replay_window import verify_window
from experimental.welding_candidate_selection_v1.pool import digest


def verify_selection(folder):
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text())
    audit=json.loads((folder/'artifact-audit.json').read_text())
    if not report['accepted'] or not audit['passed'] or not audit['accepted']:
        raise ValueError('Accepted independently audited selection required')
    if digest(folder/'route.npz')!=audit['route_sha256'] or digest(folder/'report.json')!=audit['result_report_sha256']:
        raise ValueError('Changed audited result')
    for name,expected in report['manifest']['bindings'].items():
        if digest(name)!=expected:raise ValueError('Changed bound source: '+name)
    for edge in report['edges']:
        if edge['passed'] and digest(edge['file'])!=edge['sha256']:
            raise ValueError('Changed certified edge')
    original=verify_window(ROOT.parent/'positioner-batches/window-a')
    return dict(original,array_sha256=audit['route_sha256'],duration_s=report['after']['duration_s'])


if __name__=='__main__':
    original_verify=export_parallel.verify_replay
    def adapter(folder):
        export_parallel.verify_replay=original_verify
        try:return verify_selection(folder)
        finally:export_parallel.verify_replay=adapter
    export_parallel.verify_replay=adapter
    sys.modules['export_parallel']=export_parallel
    replay_parallel.main()
