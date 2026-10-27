import time
import numpy as np
import pytest
import run_transfer_limited_v130 as limited


def test_budget_interrupts_every_state_query_including_cached_ones():
    bridge = limited.BudgetBridge.__new__(limited.BudgetBridge)
    bridge.search_deadline = time.perf_counter()-1.
    with pytest.raises(limited.SearchBudgetExpired):
        bridge.valid_scaled(np.zeros(15))


def test_budget_proposal_returns_no_candidate_and_clears_timer():
    class Bridge:
        search_deadline = 1.
    bridge = Bridge()
    def expired():
        raise limited.SearchBudgetExpired()
    path, record = limited.bounded_proposal(bridge, expired)
    assert path is None and record[0]['reason'] == 'search_wall_time_limit'
    assert bridge.search_deadline is None


def test_success_resets_timer_before_full_certification():
    class Bridge:
        search_deadline = 1.
    bridge = Bridge()
    path, records = limited.bounded_proposal(bridge, lambda: ([1, 2], []))
    assert path == [1, 2] and bridge.search_deadline is None
