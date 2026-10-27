import importlib.util
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('angle_retry',Path(__file__).parents[1]/'retry_actions.py')
retry=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


def test_unattempted_first_last_interrupted_remains_retryable():
    rows=[dict(robot_id='a',roll_deg=0,axis_sign=1,q_rad=[q]*6) for q in (0,1,2)]
    attempts=[dict(candidate=dict(robot_id='a',roll_deg=0,axis_sign=1),q_rad=[q]*6) for q in (0,1)]
    result=retry.retry_order(rows,attempts)
    assert [row['q_rad'][0] for row in result]==[2,1,0]


def test_prior_acceptance_copies_path_without_changing_receipt(tmp_path):
    import hashlib,json
    source=tmp_path/'source'
    target=tmp_path/'target'
    source.mkdir()
    path=source/'path.json'
    path.write_text(json.dumps(dict(passed=True)))
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    row=dict(weld_id='w',passed=True,best=dict(path_file='path.json',path_sha256=checksum),
             online_acceptance=dict(passed=True,path_sha256=checksum,validation=dict(passed=True)))
    copied=retry.copy_accepted(row,source,target,3)
    assert copied['online_acceptance']==row['online_acceptance']
    assert copied['best']['path_file']=='paths/retreat_03.json'
    assert (target/copied['best']['path_file']).read_bytes()==path.read_bytes()
    assert row['best']['path_file']=='path.json'
