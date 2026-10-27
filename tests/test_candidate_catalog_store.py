import importlib.util
from pathlib import Path
import pytest

FILE=Path(__file__).parents[1]/'experimental/welding_candidate_catalog_v1/store.py'
def module():
    assert FILE.exists(), 'durable catalogue store is not implemented'
    spec=importlib.util.spec_from_file_location('catalog_store',FILE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_resume_rejects_changed_inputs(tmp_path):
    m=module();m.CatalogStore(tmp_path/'a.sqlite',{'scene':'a'}).close()
    with pytest.raises(ValueError,match='identity'):
        m.CatalogStore(tmp_path/'a.sqlite',{'scene':'b'})

def test_angle_commit_keeps_every_trial_and_solution(tmp_path):
    m=module();s=m.CatalogStore(tmp_path/'a.sqlite',{'scene':'a'})
    rows=[{'robot_id':'R1','roll_deg':0,'axis_sign':1,'status':'candidates_found','accepted':[{'q_rad':[0]*6},{'q_rad':[6.28,0,0,0,0,0]}]},
          {'robot_id':'R2','roll_deg':0,'axis_sign':1,'status':'ik_not_found','accepted':[]}]
    s.save_relative('w',{'allowed':[{}]});s.save_angle('w',5,rows)
    assert s.has_angle('w',5)
    assert s.summary()['pose_trials']==2
    assert s.summary()['accepted_configurations']==2
    assert s.summary()['angle_tasks']==1
    s.close()

def test_failed_angle_transaction_leaves_no_completion(tmp_path):
    m=module();s=m.CatalogStore(tmp_path/'a.sqlite',{'scene':'a'})
    with pytest.raises(KeyError):s.save_angle('w',0,[{'invalid':'record'}])
    assert not s.has_angle('w',0)
    assert s.summary()['pose_trials']==0


def test_duplicate_angle_write_cannot_duplicate_candidates(tmp_path):
    m=module();s=m.CatalogStore(tmp_path/'a.sqlite',{'scene':'a'})
    r=[{'robot_id':'R1','roll_deg':0,'axis_sign':1,'status':'candidates_found','accepted':[{'q_rad':[0]*6}]}]
    s.save_angle('w',0,r);s.save_angle('w',0,r)
    assert s.summary()['accepted_configurations']==1


def test_geometry_only_completion_is_not_ik_completion(tmp_path):
    m=module();s=m.CatalogStore(tmp_path/'a.sqlite',{'scene':'a'})
    s.save_angle('w',0,[],stage='fixed')
    assert s.has_angle('w',0,'fixed')
    assert not s.has_angle('w',0,'ik')
