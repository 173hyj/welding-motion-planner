import hashlib
import json
import pytest
from experimental.trajopt_transfer_v1.export_replay import verify_route


def test_bound_replacement_hash_and_full_order(tmp_path):
    source = tmp_path/'report.json'
    source.write_text(json.dumps({'order_weld_ids':[1,2,3], 'order':[0,1,2], 'robot_ids':['a','b'], 'nodes':[], 'final_retreat_out':'baseline-actions'}))
    edge = tmp_path/'path.json'
    edge.write_text('{}')
    optimization = tmp_path/'optimization_report.json'
    optimization.write_text('{}')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    route = dict(source_report=str(source), source_report_sha256=digest(source),
                 optimization_report_sha256=digest(optimization), order=[0,1,2],
                 order_weld_ids=[1,2,3], robot_ids=['a','b'], nodes=[], final_retreat_out='baseline-actions', complete=True,
                 selected_edges=[dict(from_weld_id=a,to_weld_id=b,path_file=str(edge),path_sha256=digest(edge)) for a,b in [(1,2),(2,3)]])
    route_file = tmp_path/'optimized_route.json'
    route_file.write_text(json.dumps(route))
    assert verify_route(route_file)[0]['order_weld_ids'] == [1,2,3]
    route['order_weld_ids'] = [1,2]
    route_file.write_text(json.dumps(route))
    with pytest.raises(ValueError, match='complete source order_weld_ids'):
        verify_route(route_file)
    route['order_weld_ids'] = [1,2,3]
    route_file.write_text(json.dumps(route))
    edge.write_text('changed')
    with pytest.raises(ValueError, match='Changed bound input'):
        verify_route(route_file)
