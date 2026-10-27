from pathlib import Path
import importlib.util,json,sqlite3
FILE=Path(__file__).parents[1]/'experimental/welding_candidate_catalog_v1/verify.py'
def verifier():
    assert FILE.exists(),'coverage verifier is not implemented'
    s=importlib.util.spec_from_file_location('catalog_verify',FILE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_partial_coverage_never_claims_complete(tmp_path):
    m=verifier();db=sqlite3.connect(tmp_path/'x.sqlite')
    db.executescript('CREATE TABLE relative(weld TEXT,payload TEXT); CREATE TABLE tasks(stage TEXT,weld TEXT,angle REAL,payload TEXT); CREATE TABLE trials(stage TEXT,weld TEXT,angle REAL,robot TEXT,roll REAL,sign INTEGER,status TEXT,payload TEXT); CREATE TABLE candidates(weld TEXT,angle REAL,robot TEXT,roll REAL,sign INTEGER,branch INTEGER,payload TEXT);')
    p={'robot_id':'r','roll_deg':0,'axis_sign':1}
    db.execute('INSERT INTO relative VALUES(?,?)',('w',json.dumps({'allowed':[p],'relative_rejections':[]})))
    db.execute('INSERT INTO tasks VALUES(?,?,?,?)',('fixed','w',0,'{}'))
    db.commit()
    result=m.audit_coverage(db,['w'],['r'],[0,5])
    assert not result['complete']
    assert result['missing_fixed_tasks']==1 and result['missing_ik_tasks']==2
    assert result['relative_grid_errors']
    assert result['trial_coverage_errors']
