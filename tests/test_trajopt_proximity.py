import unittest
import numpy as np
from experimental.trajopt_transfer_v1.proximity import enumerate_pairs, polygon_signed_distance, sample_weights

class ProximityTests(unittest.TestCase):
    def test_source_exemptions_and_additional_checks(self):
        class World: pass
        w=World(); w.static=[('floor','floor',object())]
        w.robots={r:{k:object() for k in ['239','240','241','242','243','244','245','246','247','gun:p']} for r in ['a','b']}
        p=enumerate_pairs(w)
        labels={(a,b) for a,b,_,_ in p}
        self.assertNotIn((('a','239'),('static','floor')),labels)
        self.assertNotIn((('a','240'),('a','246')),labels)
        self.assertIn((('a','239'),('a','242')),labels)
        self.assertIn((('a','246'),('a','gun:p')),labels)
        self.assertNotIn((('a','245'),('a','gun:p')),labels)
        self.assertIn((('a','240'),('b','239')),labels)
        self.assertEqual(len(labels),len(p))
    def test_polygon_and_midpoint_weights(self):
        poly=np.array([[0,0],[2,0],[2,2],[0,2]])
        self.assertAlmostEqual(polygon_signed_distance([1,1],poly),1)
        self.assertAlmostEqual(polygon_signed_distance([3,1],poly),-1)
        samples=sample_weights(3)
        self.assertEqual(samples, [((0,1.),),((1,1.),),((2,1.),),((0,.5),(1,.5)),((1,.5),(2,.5))])

if __name__=='__main__': unittest.main()

class OracleTests(unittest.TestCase):
    def test_gradient_midpoint_margin_and_restoration(self):
        from unittest.mock import patch
        from experimental.trajopt_transfer_v1.proximity import SourceProximity
        class Model: polygons={}
        class World:
            robots={}; static=[]; models={'r':Model()}
            def update(self,*args): pass
        class Engine:
            def set_angle(self,a): pass
        class Bridge:
            world=World(); engine=Engine(); robot_ids=['r']; scale=np.r_[np.ones(13),100,100]
        oracle=SourceProximity(Bridge(),activation_mm=10)
        oracle.pairs=[(('r','x'),('static','s'),None,None)]
        seen=[]
        oracle.set_state=lambda x: seen.append(np.array(x))
        oracle._values=lambda pairs: np.full(len(pairs),seen[-1][0])
        path=np.zeros((3,15)); path[:,0]=[.02,.04,.06]
        g,j=oracle.oracle(path)
        np.testing.assert_allclose(g,[.02,.03,.06,.02,.04])
        np.testing.assert_allclose(j[3,[0,15]],[.5,.5],atol=1e-10)
        np.testing.assert_allclose(seen[-1],path[-1])
        self.assertEqual(j.shape,(5,45))
        self.assertIsNone(oracle.oracle(path,False)[1])

class LayoutPairTests(unittest.TestCase):
    def test_declared_rotary_interfaces_only_are_exempt(self):
        from experimental.trajopt_transfer_v1.proximity import layout_pairs
        class Env:
            moving_id='dong';fixed_id='jing';interface_id='interface'
            relative_ids={'dong','product'}
            objects={'dong':1,'product':2}
            fixed_env=[('jing','fixed',3),('interface','mount',4),('floor','floor',5)]
        pairs=layout_pairs(Env())
        self.assertEqual(len(pairs),4)
        self.assertTrue(any(p[0][1]=='product' and p[1][1]=='interface' for p in pairs))

class BoundsTests(unittest.TestCase):
    def test_transformed_box_and_conservative_gap(self):
        from experimental.trajopt_transfer_v1.proximity import transformed_aabb,aabb_gap
        class Obj:
            def getRotation(self): return np.array([[0,-1,0],[1,0,0],[0,0,1]])
            def getTranslation(self): return np.array([10,0,0])
        lo,hi=transformed_aabb(Obj(),(np.array([1,0,0]),np.array([2,1,3])))
        np.testing.assert_allclose(lo,[9,-1,-3]);np.testing.assert_allclose(hi,[11,3,3])
        self.assertEqual(aabb_gap((lo,hi),(np.array([12,0,0]),np.array([13,1,1]))),1)
        self.assertEqual(aabb_gap((lo,hi),(np.zeros(3),np.array([10,1,1]))),0)
    def test_capture_restores_factory_even_on_error(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from experimental.trajopt_transfer_v1.proximity import capture_mesh_bounds
        original=lambda mesh,t: object()
        module=SimpleNamespace(fcl_object=original)
        mesh=SimpleNamespace(vertices=np.array([[0,0,0],[2,4,6]]))
        with patch.dict('sys.modules',{'collision_audit':module}):
            with self.assertRaises(RuntimeError):
                with capture_mesh_bounds() as bounds:
                    obj=module.fcl_object(mesh,None)
                    np.testing.assert_allclose(bounds[id(obj)][0],[1,2,3])
                    raise RuntimeError('construction failed')
            self.assertIs(module.fcl_object,original)

class ParameterTests(unittest.TestCase):
    def test_invalid_parameters_rejected_before_engine_use(self):
        from experimental.trajopt_transfer_v1.proximity import SourceProximity
        for kwargs in ({'activation_mm':.5,'margin_mm':1},{'margin_mm':-1},{'activation_mm':float('nan')},{'h':0},{'h':float('inf')}):
            with self.assertRaises(ValueError): SourceProximity(None,**kwargs)

class AxisSubsetTests(unittest.TestCase):
    def test_pair_axis_dependencies(self):
        from experimental.trajopt_transfer_v1.proximity import pair_axes
        moving=object();fixed=object()
        self.assertEqual(pair_axes((('a','240'),('static','floor'),fixed,fixed),['a','b'],{id(moving)}),set(range(6)))
        self.assertEqual(pair_axes((('a','gun:p'),('b','241'),fixed,fixed),['a','b'],{id(moving)}),set(range(12))|{13})
        self.assertEqual(pair_axes((('static','product'),('static','floor'),moving,fixed),['a','b'],{id(moving)}),{12})
