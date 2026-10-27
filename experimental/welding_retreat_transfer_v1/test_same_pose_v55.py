import importlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from welding_kinematics import RobotKinematics


class SamePoseTests(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('probe_same_pose_v55')
        except ModuleNotFoundError:self.fail('Same-pose source validation missing')
        scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        asset=next(a for a in scene['assets'] if a['kind']=='robot')
        self.model=RobotKinematics(scene,asset)
        self.q=np.array([-3.,-.8,.5,0.,.4,-np.pi/2])

    def test_actual_source_pose_must_be_preserved(self):
        self.assertTrue(self.module.same_source_pose(self.model,self.q,self.q))
        changed=self.q.copy();changed[0]+=.01
        self.assertFalse(self.module.same_source_pose(self.model,self.q,changed))

    def test_equivalent_fk_outside_source_bounds_is_rejected(self):
        changed=self.q.copy();changed[5]-=2*np.pi
        np.testing.assert_allclose(self.model.tcp(changed),self.model.tcp(self.q),atol=1e-8,rtol=0)
        self.assertFalse(self.module.same_source_pose(self.model,self.q,changed))
        self.assertFalse(self.module.same_source_pose(self.model,self.q,[np.nan]*6))


if __name__=='__main__':unittest.main()
