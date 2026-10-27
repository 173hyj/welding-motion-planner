from pathlib import Path
import importlib.util
FILE=Path(__file__).parents[1]/'experimental/welding_candidate_catalog_v1/run.py'
def runner():
    assert FILE.exists(),'exhaustive runner is not implemented'
    s=importlib.util.spec_from_file_location('catalog_run',FILE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class World:
    def __init__(self,blocked=False):self.blocked=blocked;self.calls=[]
    def set_angle(self,a):return {'blocking_pairs':[1] if self.blocked else []}
    def fixed(self,w,p):self.calls.append(('fixed',p['roll_deg']));return {'passed':p['roll_deg']==0}
    def solve(self,w,p):self.calls.append(('ik',p['roll_deg']));return [{'q_rad':[0]*6},{'q_rad':[1]*6}],{'starts':2}
    def accept(self,w,p,s):return {'passed':True,'q_rad':s['q_rad']}

def poses():return [dict(robot_id='r',roll_deg=a,axis_sign=1) for a in (0,5)]

def test_fixed_stage_never_calls_ik():
    m=runner();w=World();rows,metadata=m.execute_angle(w,{'id':'w'},0,poses(),'fixed')
    assert len(rows)==2
    assert all(c[0]=='fixed' for c in w.calls)
    assert [r['status'] for r in rows]==['fixed_clear','geometry_rejected']

def test_ik_stage_keeps_all_found_states_after_geometry():
    m=runner();w=World();fixed,_=m.execute_angle(w,{'id':'w'},0,poses(),'fixed')
    w.calls=[];rows,_=m.execute_angle(w,{'id':'w'},0,poses(),'ik',fixed_rows=fixed)
    assert w.calls==[('ik',0)]
    assert len(rows[0]['accepted'])==2
    assert rows[1]['status']=='geometry_rejected'

def test_blocked_layout_is_recorded_for_every_pose():
    m=runner();w=World(True);rows,_=m.execute_angle(w,{'id':'w'},0,poses(),'fixed')
    assert len(rows)==2 and not w.calls
    assert all(r['status']=='geometry_rejected' for r in rows)

def test_bounded_smoke_skips_uncomputed_fixed_tasks(tmp_path):
    m=runner()
    from store import CatalogStore
    s=CatalogStore(tmp_path/'s.sqlite',{'a':1})
    s.save_relative('w',{'allowed':poses()})
    s.save_angle('w',0,[dict(**p,geometry={'passed':True},status='fixed_clear') for p in poses()],stage='fixed')
    jobs=m.make_jobs(s,[{'id':'w'}],'ik',[0,5],1)
    assert len(jobs)==1 and jobs[0][1]==0
    import pytest
    with pytest.raises(ValueError,match='fixed-environment'):
        m.make_jobs(s,[{'id':'w'}],'ik',[0,5],0)
