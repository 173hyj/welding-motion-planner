"""One-off source geometry experiment; no motion or route claims."""
import json
import itertools
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'welding_variable_open_v1')]
from run_transfer_v5 import VariableToolWorld, assert_frozen_geometry_cache
from world_bridge_v2 import SourceBridge
from run_geometry import atomic, digest


def main():
    start = time.perf_counter()
    assert_frozen_geometry_cache()
    inputs = [ROOT/'build/scene_placement_corrected_v1.json',
              ROOT/'build/welding_transfer_200mm_diagnostic_v5/nodes.json',
              ROOT/'build/welding_retreat_200cm_v2/paths/retreat_16.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    nodes = json.loads(inputs[1].read_text())['nodes']
    remote = json.loads(inputs[2].read_text())
    bridge = SourceBridge(VariableToolWorld(json.loads(inputs[0].read_text())))
    banks = [[], []]
    def add(i, q, name):
        q = np.asarray(q)
        if not any(np.array_equal(q, t[0]) for t in banks[i]):
            banks[i].append((q, name))
    for k, rid in enumerate(bridge.robot_ids):
        for node in nodes:
            add(k, node['state'][6*k:6*k+6], node['weld_id'])
        if rid == remote['robot_id']:
            for travel in (500., 1000., 1500., 2000.):
                frame = min(remote['frames'], key=lambda f: abs(f['travel_mm']-travel))
                add(k, frame['q_rad'], 'remote_'+str(travel))
        else:
            add(k, remote['other_q_rad'], 'remote_other_park')
        for j1, j2, j3 in itertools.product((-3., -1.5, 0., 1.5, 3.), (-.8, .2, .8), (-.2, .5)):
            add(k, [j1, j2, j3, 0., 0., np.pi/2], 'bounded_park')
    records, passing = [], []
    # Check each arm with the known valid other-arm state first. This is only
    # rejection filtering: joint candidates still require combined checks.
    reference = np.asarray(nodes[0]['state'])
    qualified = [[], []]
    for i, bank in enumerate(banks):
        for q, label in bank:
            for driver in (0., -30., -80.):
                s = reference.copy(); s[6*i:6*i+6] = q; s[13+i] = driver
                if bridge.valid_scaled(s/bridge.scale):
                    qualified[i].append((q, label, driver))
        print('QUALIFIED_ARM', i, len(qualified[i]), flush=True)
    for left, right in itertools.product(*qualified):
        s = np.zeros(15)
        s[:6], s[6:12] = left[0], right[0]
        s[13], s[14] = left[2], right[2]
        ok = True
        for angle in (0., 5., 25.):
            s[12] = np.deg2rad(angle)
            if not bridge.valid_scaled(s/bridge.scale):
                ok = False; break
        if ok:
            s[12] = 0.
            passing.append(dict(state=s.tolist(), labels=[left[1], right[1]],
                                checked_positioner_deg=[0., 5., 25.]))
    report = dict(source_sha256=hashes, code_sha256=digest(Path(__file__)),
                  checked_combinations=len(qualified[0])*len(qualified[1]),
                  passing=passing, elapsed_s=time.perf_counter()-start,
                  source_checks=bridge.checks, motion_verified=False)
    for name, value in hashes.items():
        if digest(ROOT/name) != value:
            raise RuntimeError('Probe input changed')
    atomic(ROOT/'build/welding_shared_park_probe_v1.json', report)
    print('SHARED_PARK_PROBE', len(passing), report['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
