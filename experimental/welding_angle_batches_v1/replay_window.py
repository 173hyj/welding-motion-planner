"""Display the sealed local-window result in the existing full-factory viewer."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experimental.dual_arm_parallel_v1 import export_parallel,replay_parallel
from experimental.welding_angle_batches_v1.actions import digest


def verify_window(folder):
    folder=Path(folder)
    result=json.loads((folder/'audit.json').read_text())
    independent=json.loads((folder/'audit-independent.json').read_text())
    if not result['accepted'] or not result['full_source_accepted'] or not independent['passed']:
        raise ValueError('Accepted audited local window required')
    if digest(folder/'route.npz')!=result['array_sha256']:
        raise ValueError('Changed local-window trajectory')
    baseline=ROOT.parent/'dual-arm-results/local04/fastest/replay'
    if digest(baseline/'route.npz')!=result['baseline_sha256']:
        raise ValueError('Changed inherited baseline')
    original=export_parallel.verify_replay(baseline)
    for group in ('source_sha256','code_sha256'):
        for name,expected in result['manifest'][group].items():
            if digest(ROOT/name)!=expected:raise ValueError('Changed source/code: '+name)
    for name,expected in result['action_path_sha256'].items():
        if digest(name)!=expected:raise ValueError('Changed action')
    for edge in result['edges']:
        if digest(folder/edge['path_file'])!=edge['path_sha256']:raise ValueError('Changed transfer')
    return dict(original,array_sha256=result['array_sha256'],duration_s=result['duration_s'])


if __name__=='__main__':
    original_verify=export_parallel.verify_replay
    def adapter(folder):
        export_parallel.verify_replay=original_verify
        try:return verify_window(folder)
        finally:export_parallel.verify_replay=adapter
    export_parallel.verify_replay=adapter
    sys.modules['export_parallel']=export_parallel
    replay_parallel.main()
