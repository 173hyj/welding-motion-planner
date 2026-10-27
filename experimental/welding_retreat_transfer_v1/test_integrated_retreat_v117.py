import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import integrated_retreat_v117 as online


def test_edge_union_contains_cold_midpoints_without_duplicate_endpoints():
    points=[np.array([x]) for x in (0., .5, 1.)]
    samples=list(online.edge_union(points))
    assert [float(q[0]) for q,save in samples] == [.25,.5,.75,1.]
    assert [float(q[0]) for q,save in samples if save] == [.5,1.]


def test_nonfinite_and_independent_fk_disagreement_rejected():
    start=np.eye(4); actual=start.copy();actual[0,3]=10.
    reference=actual.copy();reference[0,3]+=.00001
    with pytest.raises(ValueError,match='FK'):
        online.pose_metrics(actual,reference,start,[1.,0.,0.],0.,10.)
    actual[0,3]=np.nan
    with pytest.raises(ValueError):online.pose_metrics(actual,actual,start,[1.,0.,0.],0.,10.)


def test_actual_intermediate_curvature_rejected():
    start=np.eye(4);actual=start.copy();actual[0,3]=.5;actual[1,3]=.06
    with pytest.raises(ValueError,match='straight'):
        online.pose_metrics(actual,actual,start,[1.,0.,0.],0.,1.)


def minimal_path():
    return dict(weld_id='w',passed=True,robot_id='a',other_robot_id='b',q_start_rad=[0.]*6,
        q_end_rad=[1.]*6,other_q_rad=[2.]*6,positioner_deg=0.,driver_mm=-10.,
        length_mm=1000.,fixed_distance_mm=1000.,direction_world=[1.,0.,0.],
        frames=[dict(q_rad=[0.]*6,driver_mm=float(d),travel_mm=0.) for d in np.linspace(0.,-10.,21)]+
               [dict(q_rad=[1.]*6,driver_mm=-10.,travel_mm=1000.)],
        online_validation=dict(passed=True,raw_source_geometry=True,independent_xml_fk=True,
            saved_edge_midpoints=True,opening_step_mm=.5,opening_samples=21,retreat_samples=2,
            maximum_source_fk_matrix_difference=0.,maximum_line_error_mm=0.,
            maximum_rotation_error_deg=0.,final_travel_mm=1000.,endpoint_clearance_mm=5.))


def write_report(tmp_path):
    path=minimal_path(); p=tmp_path/'path.json';p.write_text(json.dumps(path));h=hashlib.sha256(p.read_bytes()).hexdigest()
    row=dict(weld_id='w',initial={k:path[k] for k in ['robot_id','other_robot_id','other_q_rad','positioner_deg']},best=dict(path_file='path.json',path_sha256=h,driver_mm=-10.,length_mm=1000.,amplitude_mm=10.),passed=True)
    row['initial']['q_rad']=path['q_start_rad']
    row['online_acceptance']=dict(path_sha256=h,passed=True,validation=copy.deepcopy(path['online_validation']))
    record=tmp_path/'point_00.json';record.write_text(json.dumps(row))
    return dict(summary=dict(complete=True,points=1,required_straight_distance_mm=1000.),rows=[row],output_sha256={'point_00.json':hashlib.sha256(record.read_bytes()).hexdigest(),'path.json':h})


def test_valid_receipt_and_changed_path_or_receipt_rejected(tmp_path):
    report=write_report(tmp_path)
    assert online.online_retreat_ids(report,tmp_path)=={'w'}
    bad=copy.deepcopy(report);bad['rows'][0]['online_acceptance']['validation']['saved_edge_midpoints']=False
    with pytest.raises(ValueError):online.online_retreat_ids(bad,tmp_path)
    bad=copy.deepcopy(report);bad['rows'][0].pop('online_acceptance')
    with pytest.raises(ValueError):online.online_retreat_ids(bad,tmp_path)
    p=tmp_path/'path.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):online.online_retreat_ids(report,tmp_path)


def test_changed_endpoint_and_incomplete_opening_rejected(tmp_path):
    report=write_report(tmp_path)
    path=json.loads((tmp_path/'path.json').read_text())
    path['q_end_rad']=[9.]*6
    with pytest.raises(ValueError,match='endpoint'):online.check_path_contract(path,report['rows'][0]['initial'],1000.)
    path=minimal_path();path['online_validation']['opening_samples']=2
    with pytest.raises(ValueError,match='opening'):online.check_path_contract(path,report['rows'][0]['initial'],1000.)


@pytest.mark.parametrize('blocked,reason',[(.25,'midpoint'),(0.,'opening')])
def test_online_generation_rejects_saved_edge_midpoint_collision(monkeypatch,blocked,reason):
    import types
    import analytic_trace_motion_v2 as old
    class Model:
        def tcp(self,q):
            t=np.eye(4);t[0,3]=q[0];return t
        def limits_valid(self,q):return True
    model=Model();engine=types.SimpleNamespace(world=types.SimpleNamespace(models={'a':model,'b':model},static=[]),
        parts={'a':{'cap':(None,'electrode',None)}},delta=np.eye(4))
    engine.online_tcp=lambda rid,q:model.tcp(q)
    engine.target=lambda weld,selected:(weld,None)
    engine.state=lambda *a,**k:{}
    engine.release_context=lambda *a,**k:dict(supports=[],initial_touching=[])
    engine.minimum_distance=lambda *a:10.
    checked=[]
    def probe(engine,rid,q,other,park,driver,weld,phase,context=None):
        checked.append(float(q[0]));return dict(passed=not np.isclose(q[0],blocked),touching=[],certificates={'cap':{'passed':True}})
    monkeypatch.setattr(old,'full_probe',probe)
    monkeypatch.setattr(old,'continuous_ik',lambda model,target,q:dict(passed=True,q_rad=[target[0,3],0.,0.,0.,0.,0.]))
    monkeypatch.setattr(old,'interpolation_probes',lambda a,b:[a,(a+b)/2,b])
    selected=dict(robot_id='a',other_robot_id='b',q_rad=[0.]*6,other_q_rad=[0.]*6,positioner_deg=0.)
    result=online.lift_path(engine,{'id':'w'},selected,selected,dict(direction_world=[1.,0.,0.],direction_frame='current_world',length_mm=1.),driver_mm=-1.)
    assert result['passed'] is False
    assert reason in result['reason']
    assert checked[-1]==blocked


def test_transfer_cli_requires_explicit_acceptance_mode():
    import subprocess,sys
    script=Path(__file__).with_name('run_transfer_integrated_v119.py')
    result=subprocess.run([sys.executable,str(script),'--retreat-out','x','--out','y'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'required' in result.stderr
    result=subprocess.run([sys.executable,str(script),'--retreat-out','x','--out','y','--retreat-online','--retreat-cold','z'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'not allowed' in result.stderr


def test_report_weld_relabel_and_intermediate_driver_change_rejected(tmp_path):
    report=write_report(tmp_path)
    bad=copy.deepcopy(report);bad['rows'][0]['weld_id']='wrong_weld'
    with pytest.raises(ValueError):online.online_retreat_ids(bad,tmp_path)
    path=minimal_path();path['frames'].insert(-1,dict(q_rad=[.5]*6,driver_mm=0.,travel_mm=500.))
    with pytest.raises(ValueError,match='driver'):online.check_path_contract(path,report['rows'][0]['initial'],1000.)


def test_changed_report_row_not_matching_sealed_point_is_rejected(tmp_path):
    report=write_report(tmp_path);report['rows'][0]['passed']=False
    with pytest.raises(ValueError):online.online_retreat_ids(report,tmp_path)


def test_self_consistent_point_record_cannot_relabel_other_weld_path(tmp_path):
    report=write_report(tmp_path);report['rows'][0]['weld_id']='other_weld'
    point=tmp_path/'point_00.json';point.write_text(json.dumps(report['rows'][0]))
    report['output_sha256']['point_00.json']=hashlib.sha256(point.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='weld identity'):online.online_retreat_ids(report,tmp_path)
