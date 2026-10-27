import itertools
import unittest
import numpy as np
from experimental.welding_retreat300_planning_v1 import graph

class GroupedPathTests(unittest.TestCase):
    def test_asymmetric_matches_bruteforce_with_all_candidates(self):
        rng=np.random.default_rng(731)
        sizes=[2,3,2,1]
        node=[rng.integers(0,9,(n,3)).astype(float) for n in sizes]
        start=[rng.integers(0,9,(n,3)).astype(float) for n in sizes]
        end=[rng.integers(0,9,(n,3)).astype(float) for n in sizes]
        edges={(a,b):rng.integers(0,20,(sizes[a],sizes[b],3)).astype(float) for a in range(4) for b in range(4) if a!=b}
        best=None
        for order in itertools.permutations(range(4)):
            for indices in itertools.product(*(range(sizes[t]) for t in order)):
                cost=start[order[0]][indices[0]]+end[order[-1]][indices[-1]]
                for t,i in zip(order,indices):cost=cost+node[t][i]
                for a,b,i,j in zip(order,order[1:],indices,indices[1:]):cost=cost+edges[a,b][i,j]
                best=min(best,tuple(cost)) if best is not None else tuple(cost)
        result=graph.grouped_open_path(node,edges,start,end)
        self.assertEqual(tuple(result['cost']),best)
        self.assertEqual(sorted(result['order']),list(range(4)))

    def test_last_candidate_is_not_pruned_and_duration_has_priority(self):
        node=[np.array([[0.,0.],[0.,0.],[0.,100000.]]),np.zeros((1,2))]
        edges={(0,1):np.array([[[10.,0.]],[[10.,0.]],[[0.,0.]]]),(1,0):np.full((1,3,2),np.inf)}
        result=graph.grouped_open_path(node,edges)
        self.assertEqual(result['indices'],[2,0])
        self.assertEqual(result['cost'],[0.,100000.])

    def test_infeasible_and_empty(self):
        self.assertIsNone(graph.grouped_open_path([np.zeros((0,1))],{}))
        self.assertEqual(graph.grouped_open_path([],{} )['order'],[])

    def test_angle_routes_use_open_interval_without_wrap(self):
        routes=graph.angle_routes([0,30,60,90,120],30)
        self.assertEqual(routes[0]['rotation_episodes'],4)
        self.assertEqual(routes[0]['angles'][0],30)
        self.assertEqual(graph.angle_routes([0,120],0)[0]['travel_deg'],120)
        self.assertTrue(all(r['rotation_episodes']==len([b for a,b in zip([30]+r['angles'],r['angles']) if a!=b]) for r in routes))

if __name__=='__main__':unittest.main()
