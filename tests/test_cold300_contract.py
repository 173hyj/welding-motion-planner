"""Regression checks for the nested cold300 planning path."""
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]/'experimental/welding_retreat_transfer_v1'
sys.path.insert(0,str(HERE))

def test_feedback_uses_same_300mm_search_module():
    import feedback_reselection_cold300 as feedback
    import run_joint_weld_retreat_cold300 as search
    assert feedback.search is search

def test_coarse_continuation_reaches_300mm_not_1000mm():
    import joint_search_cold300 as search
    observed=[]
    def ik(model,target,previous,**kwargs):
        observed.append(target[:3,3].copy())
        return dict(passed=True,q_rad=np.zeros(6))
    with patch('analytic_ik_v2.analytical_continuous_ik',side_effect=ik):
        assert search.coarse_kinematic(None,np.eye(4),np.zeros(6),np.array([0.,0.,1.]))
    assert len(observed)==20
    np.testing.assert_allclose(observed[-1],[0,0,300])
    assert max(x[2] for x in observed)==300
