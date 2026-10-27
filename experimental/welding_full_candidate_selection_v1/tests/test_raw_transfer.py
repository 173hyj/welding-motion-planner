from types import SimpleNamespace
import sys
from unittest.mock import patch
import numpy as np


class FakeBridge:
    def __init__(self):
        self.robot_ids=['R0','R1'];self.scale=np.r_[np.ones(13),100.,100.]
        self.low=np.r_[np.full(13,-10.),-185.,-185.];self.high=np.r_[np.full(13,10.),0.,0.]
        self.resolutions=np.r_[np.full(12,.001),.004,.5,.5];self.checks=0
    def edge_scaled(self,a,b):self.checks+=1;return True
    def valid_scaled(self,a):self.checks+=1;return True


def test_raw_transfer_allows_opposite_wrist_signs_without_wrapping():
    from experimental.welding_full_candidate_selection_v1.raw_transfer import make_transfer
    for arm,sign in [(0,1.),(1,-1.)]:
        start=np.zeros(7);start[5]=sign
        goal=start.copy();goal[5]=sign*2.
        with patch.dict(sys.modules,{'world_bridge_v2':SimpleNamespace(SourceBridge=FakeBridge)}):
            path,stats=make_transfer(None,FakeBridge(),arm,30.,start,goal,park7=np.zeros(7))
        assert stats['fine_certified']
        np.testing.assert_array_equal(path[[0,-1]],[start,goal])


def test_raw_transfer_still_rejects_out_of_bounds():
    from experimental.welding_full_candidate_selection_v1.raw_transfer import make_transfer
    start=np.zeros(7);goal=start.copy();goal[0]=11.
    with patch.dict(sys.modules,{'world_bridge_v2':SimpleNamespace(SourceBridge=FakeBridge)}):
        path,_=make_transfer(None,FakeBridge(),0,30.,start,goal,park7=np.zeros(7),max_iterations=0)
    assert path is None


def test_positive_coarse_proposal_cannot_bypass_fine_source_gate():
    from experimental.welding_full_candidate_selection_v1.raw_transfer import make_transfer
    class FineReject(FakeBridge):
        def edge_scaled(self,a,b):return False
    start=np.zeros(7);start[5]=1.;goal=start.copy();goal[0]=.1
    with patch.dict(sys.modules,{'world_bridge_v2':SimpleNamespace(SourceBridge=FineReject)}):
        path,stats=make_transfer(None,FakeBridge(),0,30.,start,goal,park7=np.zeros(7),max_iterations=0)
    assert path is None and not stats['fine_certified']
