"""Portable replay of the sealed 21-weld handoff, not a planning rerun."""
import argparse, hashlib, json, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    args,extra=p.parse_known_args()
    root=args.bundle.resolve();project=root/'project'
    manifest=json.loads((root/'SHA256SUMS.json').read_text())
    for name,expected in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Unsafe manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Hash mismatch: '+name)
    report=json.loads((root/'trajectory/report.json').read_text())
    audit=json.loads((root/'trajectory/independent-audit.json').read_text())
    assert report['accepted'] and report['complete'] and audit['passed']
    assert hashlib.sha256((root/'trajectory/route.npz').read_bytes()).hexdigest()==audit['route_sha256']
    fixture=json.loads((root/'fixture_positioner.json').read_text())
    selected=json.loads((root/'welds/selected_21.json').read_text())
    assert len(selected)==21 and {v['task'] for v in selected}==set(range(1,22))
    import numpy as np
    with np.load(root/'trajectory/route.npz',allow_pickle=False) as a:
        assert a['states'].shape==(62244,15)
        assert np.all(np.diff(a['time'])>0)
        assert abs(float(a['time'][-1])-349.830037133138)<1e-8
        first=a['states'][0].copy()
    sys.path.insert(0,str(project))
    from experimental.trajopt_transfer_v1.replay_optimized import LiveFK
    source=json.loads((project/'build/scene_placement_corrected_v1.json').read_text())
    fk=LiveFK(source,fixture['robot_ids']);fk.frame(first)
    print('PASS: all manifest hashes; 21 welds; 62244 states; 349.830037133138 s; source FK startup',flush=True)
    if args.verify_only:return
    from experimental.dual_arm_parallel_v1 import export_parallel,replay_parallel
    def verified_replay(folder):
        assert Path(folder).resolve()==root/'trajectory'
        return dict(robot_ids=fixture['robot_ids'],order=[v['weld_id'] for v in sorted(selected,key=lambda v:v['task'])],array_sha256=audit['route_sha256'],duration_s=report['after']['duration_s'])
    export_parallel.verify_replay=verified_replay
    sys.modules['export_parallel']=export_parallel
    sys.argv=[sys.argv[0],'--route',str(root/'trajectory')]+extra
    replay_parallel.main()

if __name__=='__main__':main()
