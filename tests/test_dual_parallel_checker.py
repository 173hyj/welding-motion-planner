import unittest
from types import SimpleNamespace
import numpy as np

try:
    from experimental.dual_arm_parallel_v1.checker import DualSourceChecker, ROBOT_IDS
except ImportError:
    DualSourceChecker = None
    ROBOT_IDS = ['F46171CC-24BD-45DD-82E8-21711C110EE9', 'F7690943-8A45-40E4-9AD3-F92856F982FA']


class DualCheckerTests(unittest.TestCase):
    def make(self, extra=()):
        self.assertIsNotNone(DualSourceChecker, 'simultaneous checker missing')
        model = SimpleNamespace(bounds=(np.full(6,-4.),np.full(6,4.)),
                                limits_valid=lambda q: True, tcp=lambda q: np.eye(4))
        actual = {}; checks = []
        def update(r,q,d): actual[r] = float(d)
        def check(r,q,gun_driver_mm,max_pairs):
            checks.append(dict(actual))
            return [dict(asset_id='p'+str(ROBOT_IDS.index(r)),link='gun:cap')] + list(extra)
        world = SimpleNamespace(models=dict.fromkeys(ROBOT_IDS,model),static=[],update=update,check=check)
        def tool(r,t,d,w,env,phase,release):
            actual[r] = 999. # tool contact queries may temporarily move source objects
            return dict(blocking_pairs=[],certificates={'cap':{'passed':True}},touching=['cap'])
        env = SimpleNamespace(positioner=SimpleNamespace(bounds=(-4.,4.)),layout_collisions=lambda a: {'blocking_pairs':[]})
        engine = SimpleNamespace(world=world,environment=env,state=tool,set_angle=lambda a:None)
        def qualify(pairs,r,p,t,phase):
            if phase not in ('closed','opening','release') or t['blocking_pairs']: return pairs
            return [x for x in pairs if not(x['checked_robot_id']==r and x['asset_id']==p and x['link']=='gun:cap' and 'distance_mm' not in x)]
        c = DualSourceChecker(engine,self_checks=lambda w,r:[],qualify_pairs=qualify)
        s=np.zeros(15); s[5]=-1.;s[11]=1.;s[13:]=[-25.,-70.]
        contexts=[dict(robot_id=r,weld={'product_id':'p'+str(i)},phase='opening') for i,r in enumerate(ROBOT_IDS)]
        return c,s,contexts,checks

    def test_two_own_contacts_are_independently_qualified(self):
        c,s,ctx,checks=self.make()
        self.assertTrue(c.probe(s,ctx)['passed'])
        self.assertTrue(all(x == dict(zip(ROBOT_IDS,[-25.,-70.])) for x in checks))

    def test_other_product_contact_remains_blocking(self):
        c,s,ctx,_=self.make([dict(asset_id='wrong',link='gun:cap')])
        self.assertFalse(c.probe(s,ctx)['passed'])

    def test_separated_contact_and_distance_pair_are_never_exempt(self):
        c,s,ctx,_=self.make([dict(asset_id='p0',link='gun:cap',distance_mm=0.)])
        ctx[0]['phase']='separated'
        self.assertFalse(c.probe(s,ctx)['passed'])

    def test_signed_domain_and_dense_context_samples(self):
        c,s,ctx,_=self.make(); bad=s.copy();bad[5]=1.
        self.assertFalse(c.probe(bad,ctx)['passed'])
        b=s.copy();b[13]+=1.1;seen=[]
        result=c.validate_edge(s,b,lambda u: seen.append(u) or ctx)
        self.assertTrue(result['passed']);self.assertEqual(len(seen),4)
        self.assertEqual(seen[0],0.);self.assertEqual(seen[-1],1.)

    def test_exact_delta_retains_product_mesh_cache_but_still_sets_angle(self):
        c,s,ctx,_=self.make();engine=c.engine
        engine.delta=np.eye(4);cache={'p0':object()};engine.product_cache=cache;calls=[]
        def set_angle(angle):
            calls.append(angle);engine.delta=np.eye(4);engine.product_cache={}
        engine.set_angle=set_angle
        c.probe(s,ctx);c.probe(s,ctx)
        self.assertIs(engine.product_cache,cache)
        self.assertEqual(len(calls),2)

    def test_changed_delta_discards_previous_product_mesh_cache(self):
        c,s,ctx,_=self.make();engine=c.engine
        engine.delta=np.eye(4);cache={'p0':object()};engine.product_cache=cache
        def set_angle(angle):
            engine.delta=np.eye(4);engine.delta[0,3]=1.;engine.product_cache={}
        engine.set_angle=set_angle
        c.probe(s,ctx)
        self.assertIsNot(engine.product_cache,cache)
        self.assertEqual(engine.product_cache,{})

if __name__=='__main__': unittest.main()
