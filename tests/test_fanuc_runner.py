from pathlib import Path
import importlib.util,sqlite3,json
ROOT=Path(__file__).parents[1]
FILE=ROOT/'experimental/welding_candidate_catalog_analytic_v1/run.py'
def runner():
 assert FILE.exists(),'analytic retry runner is missing'
 s=importlib.util.spec_from_file_location('analytic_run_test',FILE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_reuse_copies_only_geometry_and_preserves_pose_evidence(tmp_path):
 m=runner()
 from experimental.welding_candidate_catalog_v1.store import CatalogStore
 old=CatalogStore(tmp_path/'old.sqlite',{'input':1})
 old.save_relative('w',{'allowed':[dict(robot_id='r',roll_deg=0,axis_sign=1)]})
 row=dict(robot_id='r',roll_deg=0,axis_sign=1,status='fixed_clear',geometry={'passed':True},accepted=[])
 old.save_angle('w',0,[row],stage='fixed',metadata={'layout':{'blocking_pairs':[]}})
 old.save_angle('w',0,[dict(row,status='ik_not_found')],stage='ik')
 new=CatalogStore(tmp_path/'new.sqlite',{'input':2})
 m.copy_geometry(old.db,new)
 assert new.relative('w')==old.relative('w')
 assert new.angle_rows('w',0,'fixed')==[row]
 assert new.has_angle('w',0,'fixed') and not new.has_angle('w',0,'ik')
 assert new.summary()['accepted_configurations']==0
