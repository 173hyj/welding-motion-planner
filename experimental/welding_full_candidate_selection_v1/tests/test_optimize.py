import numpy as np
from experimental.welding_full_candidate_selection_v1.optimize import build_graph,selected_transfers,endpoint_cost,local_metrics
from experimental.welding_full_candidate_selection_v1.graph import shortest_path_avoiding


def node(name,q,cost):
    return dict(id=name,endpoint=np.r_[q,np.zeros(6)],cost=cost)


def test_all_nodes_participate_including_last_and_complete_path_ban_only():
    layers={7:dict(arm=0),1:dict(arm=1),2:dict(arm=1)}
    nodes={7:[node('a'+str(i),float(i),4.-i) for i in range(4)],1:[node('b',0.,1.)],2:[node('c',0.,1.)]}
    parks=[np.zeros(7),np.zeros(7)];order,costs,sc,ec,keys=build_graph(layers,nodes,{},[[7],[1,2]],parks,parks,0)
    assert order==[7,1,2]
    assert len(sc)==4 and costs[0].shape==(4,1)
    assert (0,0,'start','a3') in keys
    first=shortest_path_avoiding([nodes[t] for t in order],costs,sc,ec)
    second=shortest_path_avoiding([nodes[t] for t in order],costs,sc,ec,forbidden_paths=[first['indices']])
    assert first['indices']!=second['indices']
    assert second['indices'][1:]==first['indices'][1:]


def test_transfers_belong_to_correct_active_arm_and_idle_arm_kept():
    nodes={3:[node('x',1.,1.)],4:[node('y',2.,1.)]};parks=[np.zeros(7)]*2
    rows=selected_transfers([3,4],[0,0],nodes,[[],[3,4]],parks,parks,1)
    assert [r[0] for r in rows]==[(1,0,'start','end'),(1,1,'start','x'),(1,1,'x','y'),(1,1,'y','end')]
    assert np.array_equal(rows[2][1],nodes[3][0]['endpoint'])


def test_actual_raw_wrist_cost_does_not_wrap():
    states=np.zeros((2,7));states[1,5]=2*np.pi
    result=local_metrics(states,np.array([0.,12.]))
    assert result['joint_travel_deg']==360.
    assert result['wrist_travel_deg']==360.
    assert endpoint_cost(states[0],states[1])>9.


def test_failed_finite_edge_blocked_but_other_node_retained():
    layers={8:dict(arm=0)};nodes={8:[node('bad',0.,0.),node('other',0.,1.)]};parks=[np.zeros(7)]*2
    records={(2,0,'start','bad'):dict(passed=False,stats=dict(status='unresolved'))}
    order,costs,sc,ec,keys=build_graph(layers,nodes,records,[[8],[]],parks,parks,2)
    assert np.isinf(sc[0]) and np.isfinite(sc[1])
    assert shortest_path_avoiding([nodes[8]],costs,sc,ec)['indices']==[1]
