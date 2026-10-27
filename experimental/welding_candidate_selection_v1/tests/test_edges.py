import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from experimental.welding_candidate_selection_v1 import edges

class Model:
    def tcp(self,q):
        m=np.eye(4); m[0,3]=1000*np.sin(q[0]); return m

class EdgeTests(unittest.TestCase):
    def setUp(self):
        self.engine=SimpleNamespace(world=SimpleNamespace(models={'F461':Model(),'F769':Model()}))
        self.a=np.zeros(15); self.b=self.a.copy(); self.b[0]=.2
    def test_inherit_exact_baseline(self):
        base=np.array([self.a,self.b])
        with patch.object(edges,'_dependencies',side_effect=AssertionError()):
            result=edges.connect(self.engine,self.a,self.b,base)
        np.testing.assert_array_equal(result['states'],base)
        self.assertTrue(result['stats']['inherited_source_validation'])
    def test_preserve_baseline_angle_profile(self):
        base=np.array([self.a,self.b,self.b.copy()]); base[:,12]=[0,.13,.2]
        a=self.a.copy(); a[0]=-.1
        b=base[-1].copy(); b[0]=.3
        calls=[]
        def adapter(engine,x,y,**kwargs):
            calls.append((x,y,kwargs))
            return dict(passed=True,states=np.array([x,y]),stats={})
        with patch.object(edges,'connect',side_effect=adapter):
            result=edges._baseline_adapters(self.engine,a,b,base,10,1,None,False)
        np.testing.assert_array_equal(result['states'][1:-1],base)
        self.assertEqual(len(calls),2)
        for x,y,kwargs in calls:
            self.assertEqual(x[12],y[12]); self.assertNotIn('baseline_states',kwargs)
    def test_cross_angle_requires_baseline(self):
        self.b[12]=.2
        self.assertFalse(edges.connect(self.engine,self.a,self.b)['passed'])
    def test_endpoint_and_fixed_axes(self):
        original=np.array([self.a,self.b]); candidate=original.copy(); candidate[-1,0]+=1e-10
        self.assertEqual(edges.accept_candidate(original,candidate,self.engine,True)[1],'endpoint_changed')
        candidate=np.array([self.a,(self.a+self.b)/2,self.b]); candidate[1,12]=.01
        self.assertEqual(edges.accept_candidate(original,candidate,self.engine,True)[1],'fixed_axes_changed')
    def test_fine_gate(self):
        states=np.array([self.a,self.b])
        self.assertEqual(edges.accept_candidate(states,states,self.engine,False)[1],'fine_source_failed')
    def test_duration_regression(self):
        original=np.array([self.a,self.b]); candidate=np.array([self.a,(self.a+self.b)/2,self.b])
        with patch.object(edges,'measured_duration',side_effect=[2.,1.]):
            self.assertEqual(edges.accept_candidate(original,candidate,self.engine,True)[1],'duration_regression')
    def test_duration_tcp(self):
        self.assertAlmostEqual(edges.measured_duration(np.array([self.a,self.b]),self.engine),max(.2/np.deg2rad(30),1000*np.sin(.2)/200))
    def test_compress_retains_corners(self):
        corner=self.b.copy(); end=corner.copy(); end[1]=.3
        states=np.vstack([np.linspace(self.a,corner,1001),np.linspace(corner,end,1001)[1:]])
        np.testing.assert_allclose(edges.compress_collinear(states),[self.a,corner,end],atol=1e-12)
    def test_inherited_rotation_skips_scp(self):
        base=np.array([self.a,self.b]); base[-1,12]=.2
        result=edges.connect(self.engine,base[0],base[-1],base,do_trajopt=True)
        self.assertFalse(result['stats']['trajopt']['attempted'])
        np.testing.assert_array_equal(result['states'],base)
    def test_trajopt_passes_callable_oracle_and_feasibility(self):
        import experimental.trajopt_transfer_v1.solver as solver
        import experimental.trajopt_transfer_v1.proximity as proximity
        middle=self.a.copy(); middle[1]=.1
        states=np.array([self.a,middle,self.b])
        bridge=SimpleNamespace(scale=edges.SCALE,low=np.full(15,-1000.),high=np.full(15,1000.))
        bridge.edge_scaled=lambda a,b: np.array_equal(a,b)
        oracle=SimpleNamespace(oracle=lambda x,linearize=True:([],None),sparse_feasibility=lambda x:True)
        def optimize(seed,low,high,fn,options):
            self.assertTrue(callable(fn)); self.assertTrue(callable(options.feasibility_callback))
            self.assertTrue(np.all(low[:,edges.FIXED]==high[:,edges.FIXED]))
            raise RuntimeError('test sentinel')
        with patch.dict('sys.modules', {'experimental.trajopt_transfer_v1.run_optimize':SimpleNamespace(refine_seed=lambda path,count:path)}),patch.object(proximity,'SourceProximity',return_value=oracle),patch.object(solver,'optimize',side_effect=optimize) as run:
            result,stats=edges._trajopt(states,self.engine,bridge,bridge,None,None,seed_path=states/edges.SCALE)
        self.assertTrue(run.called)
        self.assertIn('test sentinel',stats['reason'])
        np.testing.assert_array_equal(result,states)

