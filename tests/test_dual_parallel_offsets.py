import pytest
from experimental.dual_arm_parallel_v1.offset_candidates import offset_candidates


def chain(duration, intervals):
    return dict(times=[0.,duration],pieces=[dict(phase='weld',start=s,end=e) for s,e in intervals])


def test_same_makespan_prefers_full_weld_overlap_before_zero_shift():
    chains=[chain(20,[(8,8.6)]),chain(10,[(2,2.6)])]
    rows=offset_candidates(chains)
    assert rows[0]['offsets']==pytest.approx((0,6))
    assert rows[0]['makespan']==20
    assert rows[0]['max_weld_overlap_seconds']==pytest.approx(.6)
    zero=next(r for r in rows if r['offsets']==(0.,0.))
    assert rows.index(zero)>0 and zero['makespan']==20


def test_alignment_can_shift_either_arm_but_never_increases_primary_cost():
    chains=[chain(10,[(2,2.6)]),chain(20,[(8,8.6)])]
    rows=offset_candidates(chains,base_offsets=[(0,1)])
    assert rows[0]['offsets']==pytest.approx((6,0))
    assert rows[0]['makespan']==20
    assert [r['makespan'] for r in rows]==sorted(r['makespan'] for r in rows)
    assert (0.,10.) in [r['offsets'] for r in rows]
    assert (20.,0.) in [r['offsets'] for r in rows]


def test_minimum_makespan_wins_even_if_more_expensive_shift_has_overlap():
    rows=offset_candidates([chain(10,[(2,2.6)]),chain(10,[(7,7.6)])])
    assert rows[0]['offsets']==(0.,0.)
    assert rows[0]['total_weld_overlap_seconds']==0
    assert any(r['full_weld_overlap'] for r in rows)


def test_no_welds_deduplicates_baseline_and_rejects_outside_chain_piece():
    rows=offset_candidates([chain(3,[]),chain(2,[])],base_offsets=[(0,0),(0,0)])
    assert len({r['offsets'] for r in rows})==len(rows)
    assert rows[0]['offsets']==(0.,0.)
    with pytest.raises(ValueError,match='interval'):
        offset_candidates([chain(3,[(2,4)]),chain(2,[])])
