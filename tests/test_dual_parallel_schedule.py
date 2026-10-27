import copy
import numpy as np
import pytest
from experimental.dual_arm_parallel_v1 import schedule


def test_alignment_preserves_orders_angles_and_pairs_blocks():
    arms = [[dict(id=f'{a}-{i}', angle=x) for i, x in enumerate(xs)]
            for a, xs in enumerate([[30,60,60,0,0]+[120]*9+[90], [30,30,90,0,120,120]])]
    original = copy.deepcopy(arms)
    stages = schedule.align_batches(arms)
    assert [s['angle'] for s in stages] == [30,60,90,0,120,90]
    assert arms == original
    for a in range(2):
        assert [t for s in stages for t in s['tasks'][a]] == arms[a]
    assert all(all(t['angle'] == s['angle'] for ts in s['tasks'] for t in ts) for s in stages)
    assert all(all(s['tasks']) for s in stages if s['angle'] in [30,0,120])


def test_state_at_clips_and_interpolates_and_accepts_consistent_seams():
    assert np.allclose(schedule.state_at([0,1,1,2], [[0],[2],[2],[4]], 1.5), [3])
    assert np.allclose(schedule.state_at([0,1], [[2],[4]], -1), [2])
    with pytest.raises(ValueError, match='duplicate'):
        schedule.state_at([0,1,1], [[0],[2],[3]], 1)


def test_merge_preserves_breakpoints_and_holds_before_shifted_start():
    s0 = np.zeros((3,7)); s0[1,0] = .002; s0[:,6] = [4,5,4]
    s1 = np.zeros((3,7)); s1[:,0] = [1,1.002,1]; s1[:,6] = [8,9,8]
    times, states = schedule.merge_samples(([0,1,2],s0), ([0,.7,2],s1), offsets=(0,.5), angle_deg=30)
    assert all(np.any(times == t) for t in [0,.5,1,1.2,2,2.5])
    assert np.allclose(states[times < .5,6:12], s1[0,:6])
    assert np.allclose(states[-1, :6], s0[-1,:6])
    assert states[-1,13] == 4 and states[-1,14] == 8
    assert np.allclose(states[:,12], np.deg2rad(30))
    limits = np.r_[np.full(12,np.deg2rad(.05)),np.deg2rad(.25),.5,.5]
    assert np.all(np.abs(np.diff(states,axis=0)) <= limits+1e-12)


def test_offset_search_finds_symmetric_best_safe_and_rejects_all_unsafe():
    visited = []
    def validate(offsets):
        visited.append(offsets)
        return offsets[0] >= 1
    result = schedule.choose_offset(3,2,validate,candidates=[(0,0),(0,.5),(1,0),(2,0)])
    assert result['offsets'] == (1.,0.)
    assert result['makespan'] == 4
    assert (3.,0.) not in visited  # serial reverse is (duration1,0)
    assert (2.,0.) in visited and (0.,3.) in visited
    assert schedule.choose_offset(3,2,lambda _:False,candidates=[(0,0)]) is None
