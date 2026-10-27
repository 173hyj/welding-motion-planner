"""Finite full-source probes at q6=0; no wrap and no connectivity claim."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import branch_point_job_v64 as search
from conservative_search_v30 import SearchBridge
from run_geometry import atomic, digest, initialize_worker

ROOT = search.ROOT
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh diagnostic output required')
    p = ROOT / 'build/welding_retreat_200cm_expanded_snapshot_v66/report.json'
    parent = json.loads(p.read_text())
    sources = {**parent['manifest']['source_sha256'], **parent['manifest']['code_sha256'],
        str(p.relative_to(ROOT)): digest(p)}
    codes = {str((HERE / n).relative_to(ROOT)): digest(HERE / n) for n in
        [Path(__file__).name, 'conservative_search_v30.py', 'world_bridge_v2.py']}
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed source: ' + name)
    out.mkdir(parents=True)
    atomic(out / 'manifest.json', dict(source_sha256=sources, code_sha256=codes,
        settings=dict(q6_rad=0., original_unsplit_limits=True, positioner_deg=30.,
            gun_openings_mm=[0., 0.], q1_rad=[-3., -1.5, 0.],
            q4_rad=[-3., -1.5, 0., 1.5, 3.], q5_rad=[-1.5, -.75, 0., .75, 1.5],
            other_first_axis_rad=[-3., 0., 1.],
            finite_diagnostic_only=True, no_full21_or_connectivity_claim=True)))
    started = time.perf_counter()
    initialize_worker()
    search.retreat.assert_frozen_geometry_cache()
    engine = search.retreat.VariableToolWorld(json.loads((ROOT / 'build/scene_placement_corrected_v1.json').read_text()))
    bridge = SearchBridge(engine)  # Original source limits: no imposed wrist sign.
    records = []
    positive = None
    base = np.zeros(15)
    base[:6] = [-3., -.8, .5, 0., 0., 0.]
    base[6:12] = [-3., -.8, .5, 0., 0., np.pi / 2.]
    base[12] = np.deg2rad(30.)
    for other_q1 in (-3., 0., 1.):
        for q1 in (-3., -1.5, 0.):
            for q4 in (-3., -1.5, 0., 1.5, 3.):
                for q5 in (-1.5, -.75, 0., .75, 1.5):
                    state = base.copy()
                    state[[0, 3, 4, 6]] = [q1, q4, q5, other_q1]
                    probe = bridge.probe(state)
                    row = dict(state=state.tolist(), passed=probe['passed'],
                        reason=probe['reason'], blocking_pairs=probe.get('blocking_pairs', []))
                    records.append(row)
                    if probe['passed']:
                        a, b = state.copy(), state.copy()
                        a[5], b[5] = -.2, .2
                        candidate = [a / bridge.scale, b / bridge.scale]
                        fine = bridge.certify_path(candidate)
                        row['short_crossing_fine_passed'] = fine
                        if fine:
                            frames = bridge.dense_frames(candidate, phase='wrist_sign_crossing')
                            atomic(out / 'short_crossing.json', dict(robot_ids=bridge.robot_ids,
                                start_state=a.tolist(), end_state=b.tolist(), frames=frames,
                                interpolation_resolution=bridge.resolutions.tolist(),
                                scope='Short q6 crossing only; not connected to any retreat endpoint, fresh cold required.'))
                            positive = row
                            break
                if positive:
                    break
            if positive:
                break
        if positive:
            break
    for name, sha in {**sources, **codes}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed source during diagnostic: ' + name)
    atomic(out / 'report.json', dict(manifest=json.loads((out / 'manifest.json').read_text()),
        summary=dict(complete=True, sampled=len(records), q6_zero_passed=sum(r['passed'] for r in records),
            short_crossing_found=positive is not None, wall_s=time.perf_counter()-started),
        rows=records, output_sha256={'short_crossing.json': digest(out / 'short_crossing.json')} if positive else {},
        scope='Full source collision and coupled limits at a finite park bank with original q6 bounds. '
              'No angular wrap; optional short interpolation uses original fine resolution. '
              'Neither negatives nor a short positive establish endpoint connectivity or a whole route.'))
    print('WRIST_ZERO_SUMMARY', len(records), sum(r['passed'] for r in records), positive is not None, flush=True)


if __name__ == '__main__':
    main()
