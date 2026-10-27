import numpy as np
import pytest
from experimental.welding_full_candidate_selection_v1.tail_inputs import remap_cached_edges
from experimental.welding_full_candidate_selection_v1.tail import strict_limits


def test_remapping_uses_hash_all_new_nodes_retained():
    old=[dict(actions=[dict(path_sha256='base'),dict(path_sha256='old')]) for _ in range(21)]
    new=[dict(actions=[dict(path_sha256='base'),dict(path_sha256='new'),dict(path_sha256='old')]) for _ in range(21)]
    rows=[dict(key=[-1,0,1],passed=True,file='sealed',sha256='same'),dict(key=[0,1,0],passed=False),dict(key=[13,1,0],passed=True)]
    result=remap_cached_edges(old,new,rows)
    assert result[0]['key']==[-1,0,2] and result[0]['file']=='sealed'
    assert result[1]['key']==[0,2,0] and result[2]['key']==[13,2,0]
    assert all(len(layer['actions'])==3 for layer in new)


def test_missing_hash_never_silently_uses_wrong_candidate():
    old=[dict(actions=[dict(path_sha256='base'),dict(path_sha256='missing')]) for _ in range(21)]
    new=[dict(actions=[dict(path_sha256='base')]) for _ in range(21)]
    with pytest.raises(ValueError,match='Cached action absent'):remap_cached_edges(old,new,[dict(key=[-1,0,1])])


def test_budget_compares_incumbent_not_original_baseline():
    prefix=np.array([100.,1000.,400.]);incumbent=np.array([459.655898,4000.,1600.]);original=np.array([461.555904,4100.,1620.])
    limits=strict_limits(incumbent,prefix)
    np.testing.assert_allclose(limits,[359.655898,2999.998999,1199.998999])
    assert np.all(limits<strict_limits(original,prefix))


def test_scalar_proposal_bans_budget_violating_complete_path_only():
    from experimental.welding_full_candidate_selection_v1.tail import scalar_proposal
    layers=[[0,1],[0,1]]
    costs=[np.array([[0.,2.],[1.,3.]])];rs=[np.zeros((2,2,3))]
    start=np.zeros(2);end=np.zeros(2);sr=np.array([[0.,9.,0.],[0.,0.,0.]]);er=np.zeros((2,3))
    result,diagnostic=scalar_proposal(layers,costs,rs,[1.,1.,1.],start,end,sr,er,max_paths=8)
    assert result['indices']==[1,0]
    assert diagnostic['budget_rejected_paths']==1
    assert diagnostic['examined_paths']==2


def test_scalar_proposal_includes_last_of_313_nodes_and_reports_cap():
    from experimental.welding_full_candidate_selection_v1.tail import scalar_proposal
    sr=np.zeros((313,3));sr[:312,0]=2.
    result,diagnostic=scalar_proposal([range(313)],[],[],[1.,1.,1.],np.arange(313.),np.zeros(313),sr,np.zeros((313,3)),max_paths=313)
    assert result['indices']==[312] and diagnostic['examined_paths']==313
    result,diagnostic=scalar_proposal([range(313)],[],[],[1.,1.,1.],np.arange(313.),np.zeros(313),sr,np.zeros((313,3)),max_paths=128)
    assert result is None and diagnostic['status']=='finite_proposal_enumeration_cap'
    assert diagnostic['max_paths']==128


def test_nonidentity_one_frame_exit_seed_is_not_forwarded(monkeypatch):
    from experimental.welding_full_candidate_selection_v1 import tail
    captured=[]
    monkeypatch.setattr(tail.original,'plan_worker',lambda job:(captured.append(job) or job[0],dict(passed=False,stats={})))
    monkeypatch.setattr(tail,'atomic',lambda *args:None)
    a=np.zeros(15);b=a.copy();b[0]=.1
    tail.refinement_worker(((13,5,0),a,b,np.array([b]),{},'unused.npz'))
    assert captured[0][3] is None


def test_identity_one_frame_seed_is_preserved(monkeypatch):
    from experimental.welding_full_candidate_selection_v1 import tail
    captured=[]
    monkeypatch.setattr(tail.original,'plan_worker',lambda job:(captured.append(job) or job[0],dict(passed=False,stats={})))
    monkeypatch.setattr(tail,'atomic',lambda *args:None)
    a=np.zeros(15);seed=np.array([a])
    tail.refinement_worker(((13,0,0),a,a,seed,{},'unused.npz'))
    assert captured[0][3] is seed
