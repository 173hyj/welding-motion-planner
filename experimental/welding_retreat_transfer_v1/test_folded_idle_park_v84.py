import unittest
import json
import sys
from pathlib import Path
import numpy as np
try:
    import folded_idle_park_v84 as fold
except ImportError:
    fold = None
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from welding_kinematics import RobotKinematics
from signed_wrist_model_v40 import SignedWristModel


class FoldedIdleTests(unittest.TestCase):
    def test_finite_source_legal_bank_changes_shoulder_elbow_but_keeps_sign(self):
        self.assertIsNotNone(fold, 'folded idle park feature missing')
        root=Path(__file__).resolve().parents[2]
        scene=json.loads((root/'build/scene_placement_corrected_v1.json').read_text())
        asset=next(a for a in scene['assets'] if a['kind']=='robot')
        for sign in (-1,1):
            model=SignedWristModel(RobotKinematics(scene,asset),sign)
            bank=fold.park_bank(model,sign)
            self.assertTrue(bank)
            self.assertLessEqual(len(bank),175)
            self.assertEqual(len(bank),len({tuple(q) for q in bank}))
            for q in bank:
                self.assertTrue(model.limits_valid(q))
                self.assertEqual(q[3:],[0.,0.,sign*np.pi/2])
            self.assertGreater(len({q[1] for q in bank}),1)
            self.assertGreater(len({q[2] for q in bank}),1)

    def test_only_real_other_arm_failures_selected_and_exact_duplicates_removed(self):
        self.assertIsNotNone(fold, 'folded idle park feature missing')
        other=dict(reason='full_retreat_blocked',amplitude_mm=80.,direction_world=[1.,0.,0.],
                   last_accepted_travel_mm=500.,pairs=[{'asset_name':'other_robot'}])
        self_case=dict(other,pairs=[{'asset_name':'self'}])
        report={'weld_id':'w','seeds':[{'seed':{'candidate':{'robot_id':'r'}},
                  'trials':[other,dict(other),self_case,dict(other,reason='coarse_tool_geometry_rejected')]}]}
        selected=fold.blocked_rays(report,3)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['trial'],other)


if __name__=='__main__': unittest.main()
