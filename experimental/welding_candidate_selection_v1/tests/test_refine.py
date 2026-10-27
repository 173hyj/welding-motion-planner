import unittest
import numpy as np
from experimental.welding_candidate_selection_v1.refine import fold_resources
from experimental.welding_candidate_selection_v1.resource import shortest_path_with_resources


class RefineGraphTests(unittest.TestCase):
    def test_nodes_charged_once_and_all_resources_enforced(self):
        layers=[{'actions':[0,1]},{'actions':[0]}]
        nodes=[[{'cost':2.,'resources':np.array([2.,4.,2.])},
                {'cost':1.,'resources':np.array([2.,3.,3.])}],
               [{'cost':3.,'resources':np.array([3.,2.,1.])}]]
        records={}
        for key in [(-1,0,0),(-1,0,1),(0,0,0),(0,1,0),(1,0,0)]:
            records[key]={'passed':True,'cost':1.,'resources':[1.,1.,1.]}
        endpoints={key:(np.zeros(15),np.zeros(15)) for key in records}
        cs,rs,sc,ec,sr,er=fold_resources(layers,nodes,records,endpoints,True)
        result=shortest_path_with_resources([[0,1],[0]],cs,rs,[8,9,6],sc,ec,sr,er)
        self.assertEqual(result['indices'],[0,0])
        self.assertEqual(result['cost'],8.)
        self.assertEqual(result['resources'],[8.,9.,6.])

    def test_failed_and_unknown_edges_excluded_from_final_graph(self):
        layers=[{'actions':[0]}];nodes=[[{'cost':0.,'resources':np.zeros(3)}]]
        endpoints={(-1,0,0):(np.zeros(15),np.ones(15)),(0,0,0):(np.ones(15),np.zeros(15))}
        records={(0,0,0):{'passed':False}}
        _,_,sc,ec,_,_=fold_resources(layers,nodes,records,endpoints,True)
        self.assertTrue(np.isinf(sc).all());self.assertTrue(np.isinf(ec).all())


if __name__=='__main__':unittest.main()
