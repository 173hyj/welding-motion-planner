import numpy as np
import pytest
from experimental.welding_candidate_selection_v1 import core
from experimental.welding_candidate_selection_v1.run import fold_graph, route_keys, slice_piece, measured_cost, verify_bindings


def test_node_costs_charged_once_and_time_budget_changes_choice():
    layers=[{'actions':[0,1]},{'actions':[0,1]}]
    nodes=[[{'cost':1.,'duration_s':1.},{'cost':2.,'duration_s':1.}],
           [{'cost':3.,'duration_s':1.},{'cost':4.,'duration_s':1.}]]
    endpoints={k:(np.zeros(15),np.zeros(15)) for k in [(-1,0,0),(-1,0,1),(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,1,0)]}
    records={k:dict(passed=True,cost=0.,duration_s=0.) for k in endpoints}
    records[0,0,0]['duration_s']=10.
    cs,ts,sc,ec,st,et=fold_graph(layers,nodes,records,endpoints,True)
    assert core.shortest_path([range(2)]*2,cs,sc,ec)=={'indices':[0,0],'cost':4.}
    result=core.shortest_path_with_time([range(2)]*2,cs,ts,sc,ec,st,et,time_budget=2.)
    assert result=={'indices':[0,1],'cost':5.,'time':2.}


def test_unknown_edge_excluded_and_failed_edge_never_optimistic():
    layers=[{'actions':[0]}];nodes=[[dict(cost=2.,duration_s=1.)]]
    endpoints={(-1,0,0):(np.zeros(15),np.zeros(15)),(0,0,0):(np.zeros(15),np.zeros(15))}
    optimistic=fold_graph(layers,nodes,{},endpoints)
    assert optimistic[2][0]==2.
    known=fold_graph(layers,nodes,{},endpoints,True)
    assert np.isinf(known[2][0])
    failed=fold_graph(layers,nodes,{(-1,0,0):dict(passed=False)},endpoints)
    assert np.isinf(failed[2][0])


def test_literal_slice_and_route_keys():
    data=dict(states=np.zeros((4,15)),times=np.array([0.,2.,5.,9.]),phases=np.full((4,2),'idle'))
    piece=slice_piece(data,1,3)
    np.testing.assert_array_equal(piece['time'],[0.,3.,7.])
    assert route_keys([2,1])==[(-1,0,2),(0,2,1),(1,1,0)]
    assert measured_cost(piece['states'],piece['time'])==pytest.approx(.35)


def test_frozen_hash_mismatch_fails_before_source_import(monkeypatch):
    from experimental.welding_candidate_selection_v1 import run
    monkeypatch.setattr(run,'digest',lambda p:'actual')
    p='source'
    with pytest.raises(ValueError,match='Changed frozen input'):
        verify_bindings({str(p):'wrong'})
