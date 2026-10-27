"""Display only the independently audited final300mm route, without changing it."""
import hashlib
import json
from pathlib import Path
import sys

RUN_ROOT=Path(__file__).resolve().parent
ROOT=RUN_ROOT/'trajopt-worktree'
sys.path.insert(0,str(ROOT))
from experimental.dual_arm_parallel_v1 import export_parallel,replay_parallel


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_final(folder):
    folder=Path(folder).resolve()
    if folder!=RUN_ROOT/'retreat300-final-v6':raise ValueError('Final V6 route required')
    report=json.loads((folder/'report.json').read_text())
    audit=json.loads((RUN_ROOT/'retreat300-final-v6-independent-audit.json').read_text())
    if not all(report.get(k) is True for k in ('complete','accepted')):raise ValueError('Accepted complete final required')
    if not all(audit.get(k) is True for k in ('passed','complete','accepted','canonical_boundary_labels_valid','strict_global_timing_valid')):raise ValueError('Independent final audit required')
    if sha(folder/'route.npz')!=audit['route_sha256'] or sha(folder/'report.json')!=audit['report_sha256']:raise ValueError('Audited final artifact changed')
    for name,expected in report['manifest']['bindings'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=expected:raise ValueError('Bound source changed: '+str(path))
    pool_file=RUN_ROOT/'retreat300-library-v1/report.json'
    if sha(pool_file)!=audit['pool_sha256']:raise ValueError('Candidate library changed')
    pool=json.loads(pool_file.read_text())
    return dict(robot_ids=pool['robot_ids'],order=[l['weld_id'] for l in sorted(pool['layers'],key=lambda l:l['task'])],array_sha256=audit['route_sha256'],duration_s=report['after']['duration_s'])


if __name__=='__main__':
    export_parallel.verify_replay=verify_final
    sys.modules['export_parallel']=export_parallel
    replay_parallel.main()
