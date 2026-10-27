"""Production-loop regression for failed-edge feedback adoption and reordering."""
import json
import pytest
from types import SimpleNamespace
import numpy as np
import run_transfer_limited_v147 as planner


def test_visit_reordering_precedes_configuration_regeneration():
    assert not planner.should_reselect(0,2)
    assert not planner.should_reselect(1,2)
    assert planner.should_reselect(2,2)


@pytest.mark.parametrize("no_route_before_reselection",[False,True])
def test_production_adoption_replans_changed_nodes_incident_edge(monkeypatch,tmp_path,no_route_before_reselection):
    source,out=tmp_path/'source',tmp_path/'out';source.mkdir()
    monkeypatch.setattr(planner,'ROOT',tmp_path)
    monkeypatch.setattr(planner,'HERE',tmp_path)
    monkeypatch.setattr(planner,'digest',lambda p:'hash')
    monkeypatch.setattr(planner,'online_retreat_ids',lambda r,p:{'w0','w1','w2'})
    monkeypatch.setattr(planner,'assert_frozen_geometry_cache',lambda:None)
    monkeypatch.setattr(planner,'VariableToolWorld',lambda scene:object())
    (tmp_path/'build').mkdir()
    (tmp_path/'build/scene_placement_corrected_v1.json').write_text('{}')
    states=[];rows=[]
    for i in range(3):
        state=np.zeros(15);state[0]=i;state[5]=-1;state[11]=1
        states.append(state.tolist())
        (source/f'path{i}.json').write_text(json.dumps({'state':state.tolist()}))
        rows.append(dict(weld_id=f'w{i}',initial={'robot_id':'a','other_robot_id':'b'},
            best={'path_file':f'path{i}.json','path_sha256':'hash'}))
    manifest={'source_sha256':{},'code_sha256':{}}
    report=dict(rows=rows,manifest=manifest,output_sha256={},summary={'complete':True,'points':3})
    (source/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(planner,'node_state',lambda p,r:np.array(p['state']))
    visit_calls=[]
    def visit(costs):
        visit_calls.append(1)
        return None if no_route_before_reselection and len(visit_calls) in (2,3) else [0,1,2]
    monkeypatch.setattr(planner,'visit_order',visit)
    args=SimpleNamespace(workers=1,iterations=200,ordering_rounds=5,edge_seconds=12.,
        edge_attempts=2,reselect_seconds=45.,reselect_per_edge=2,reorder_first_rounds=2 if no_route_before_reselection else 0,
        retreat_out=source,out=out,resume=False,retreat_online=True,retreat_cold=None,
        seed_snapshot=None,same_angle_probes=False)
    monkeypatch.setattr(planner.argparse.ArgumentParser,'parse_args',lambda self:args)
    class Future:
        def __init__(self,value):self.value=value
        def result(self):return self.value
    class Pool:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,*a):return Future(fn(*a))
    monkeypatch.setattr(planner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(planner.concurrent.futures,'as_completed',lambda values:values)
    events=[]
    def transfer(i,j,nodes,folder,*args):
        events.append(('transfer',i,j,nodes[1]['state'][1]))
        passed=(i,j)!=(0,1) or nodes[1]['state'][1]==.1
        record=dict(from_index=i,to_index=j,passed=passed,metric_length=1.,elapsed_s=0.)
        if passed:
            data=dict(start_state=nodes[i]['state'],end_state=nodes[j]['state'])
            name=f'paths/transfer_{i:02d}_{j:02d}.json'
            planner.atomic(folder/name,data)
            record.update(path_file=name,path_sha256='hash')
        planner.atomic(folder/f'edge_{i:02d}_{j:02d}.json',record)
        return record
    monkeypatch.setattr(planner,'transition_job',transfer)
    class Reselection:
        def __init__(self,*args):pass
        def generate(self,folder,weld_id,candidate_out):
            events.append(('reselect',weld_id))
            assert weld_id=='w1'
            candidate_out.mkdir(parents=True)
            new_rows=json.loads(json.dumps(rows));changed=list(states[1]);changed[1]=.1
            new_rows[1]['best']['path_file']='path1.json'
            planner.atomic(candidate_out/'path1.json',dict(state=changed))
            for i in (0,2):planner.atomic(candidate_out/f'path{i}.json',dict(state=states[i]))
            candidate=dict(report,rows=new_rows,output_sha256={})
            planner.atomic(candidate_out/'report.json',candidate)
            return candidate,new_rows[1]
    monkeypatch.setattr(planner,'Reselection',Reselection)
    planner.main()
    result=json.loads((out/'report.json').read_text())
    assert result['summary']['all_required_points_route_planned']
    assert events==[('transfer',0,1,0.),('transfer',1,2,0.),('reselect','w1'),
                   ('transfer',0,1,.1),('transfer',1,2,.1)]
    assert (out/'invalidated/revision_01/edge_01_02.json').is_file()
    edge=json.loads((out/'paths/transfer_01_02.json').read_text())
    assert edge['start_state']==result['nodes'][1]['state']
