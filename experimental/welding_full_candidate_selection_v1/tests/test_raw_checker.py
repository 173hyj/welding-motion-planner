import unittest
from types import SimpleNamespace
import numpy as np
from experimental.welding_full_candidate_selection_v1.raw_checker import RawDualSourceChecker,ROBOT_IDS


class RawCheckerTests(unittest.TestCase):
    def checker(self,collision=False,coupled=True):
        models={r:SimpleNamespace(bounds=(np.full(6,-10.),np.full(6,10.)),limits_valid=lambda q:coupled) for r in ROBOT_IDS}
        world=SimpleNamespace(models=models,static=[],update=lambda *args:None,
            check=lambda rid,*args,**kwargs:([{'collision':'physical'}] if collision and rid==ROBOT_IDS[0] else []))
        environment=SimpleNamespace(positioner=SimpleNamespace(bounds=(-4.,4.)),layout_collisions=lambda angle:{'blocking_pairs':[]})
        engine=SimpleNamespace(world=world,environment=environment,set_angle=lambda angle:None)
        return RawDualSourceChecker(engine,self_checks=lambda *args:[],qualify_pairs=lambda pairs,*args:pairs)

    def test_both_wrist_signs_allowed_within_physical_bounds(self):
        q=np.zeros(15);q[5]=1.;q[11]=-1.
        result=self.checker().probe(q)
        self.assertTrue(result['passed']);self.assertTrue(result['source_joint_limits_valid'])

    def test_source_bounds_and_coupled_limits_still_reject(self):
        q=np.zeros(15);q[0]=11.
        self.assertEqual(self.checker().probe(q)['reason'],'source_axis_bounds')
        self.assertEqual(self.checker(coupled=False).probe(np.zeros(15))['reason'],'source_robot_coupled_limits')

    def test_physical_collision_and_nonfinite_still_reject(self):
        self.assertFalse(self.checker(collision=True).probe(np.zeros(15))['passed'])
        q=np.zeros(15);q[3]=np.nan
        self.assertFalse(self.checker().probe(q)['passed'])


if __name__=='__main__':unittest.main()
