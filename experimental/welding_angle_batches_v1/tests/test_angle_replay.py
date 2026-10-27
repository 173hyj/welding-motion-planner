import importlib.util
from pathlib import Path
import numpy as np

SPEC = importlib.util.spec_from_file_location('angle_replay', Path(__file__).parents[1]/'replay.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_rotation_episodes_are_contiguous_movements_not_distinct_angles():
    states = np.zeros((8,15))
    states[:,12] = np.deg2rad([0,1,2,2,2,3,4,4])
    assert replay.rotation_metrics(states) == dict(rotation_episodes=2, total_positioner_travel_deg=4.)


def test_timing_respects_every_axis_and_tcp_speed_and_welding_dwell():
    states = np.zeros((4,15))
    states[1:,0] = np.deg2rad(30.)
    states[2:,12] = np.deg2rad(30.)
    states[3:,13] = 80.
    tcp = np.zeros((4,2,3))
    tcp[1:,0,0] = 400.
    times = replay.route_times(states,tcp,['entry','entry','entry','weld'])
    np.testing.assert_allclose(np.diff(times), [2.,2.,1.])
    stationary = replay.route_times(np.zeros((2,15)),np.zeros((2,2,3)),['close','weld'])
    np.testing.assert_allclose(stationary,[0,.6])


def test_append_requires_exact_seam_tolerance():
    data = [np.zeros(15)]
    replay.append_section(data,[np.zeros(15),np.ones(15)])
    assert len(data)==2
    try:
        replay.append_section(data,[np.ones(15)+1e-8])
    except ValueError:
        pass
    else:
        raise AssertionError('Unmatched seam accepted')


def test_transfer_receipt_must_bind_current_actions_baseline_and_passed_payload():
    receipt = dict(actions_report_sha256='action',baseline_sha256='baseline')
    replay.check_transfer_binding(receipt,'action','baseline')
    for changed in [('other','baseline'),('action','other')]:
        try:
            replay.check_transfer_binding(receipt,*changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Stale transfer binding accepted')
    for payload in ({},dict(passed=False,full_source_accepted=True),dict(passed=True,full_source_accepted=False)):
        try:
            replay.check_transfer_payload(payload)
        except ValueError:
            pass
        else:
            raise AssertionError('Unaccepted source path accepted')
    replay.check_transfer_payload(dict(passed=True,full_source_accepted=True))
