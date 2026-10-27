import unittest
import pathlib
import json
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class KinematicsTests(unittest.TestCase):
    def test_current_pose_matches_independently_decoded_source_link_matrices(self):
        self.assertTrue((ROOT / "welding_kinematics.py").exists(), "kinematics not implemented")
        from welding_kinematics import RobotKinematics
        scene = json.loads((ROOT / "build/scene.json").read_text())
        asset = next(a for a in scene["assets"] if a["kind"] == "robot")
        model = RobotKinematics(scene, asset)
        links = model.links(asset["robot"]["q_initial"])
        for key, source in asset["robot"]["link_models"].items():
            expected = np.array(source["current_transform"]).reshape(4, 4)
            self.assertLess(np.linalg.norm(links[key][:3, 3] - expected[:3, 3]), 0.03)
            self.assertLess(np.linalg.norm(links[key][:3, :3] - expected[:3, :3]), 0.0001)

    def test_dependent_joint_limits_use_full_source_polygon(self):
        self.assertTrue((ROOT / "welding_kinematics.py").exists(), "kinematics not implemented")
        from welding_kinematics import RobotKinematics
        scene = json.loads((ROOT / "build/scene.json").read_text())
        asset = next(a for a in scene["assets"] if a["kind"] == "robot")
        model = RobotKinematics(scene, asset)
        self.assertTrue(model.limits_valid(asset["robot"]["q_initial"]))
        q = np.array(asset["robot"]["q_initial"])
        q[4] = np.deg2rad(119)
        q[5] = np.deg2rad(170)
        self.assertFalse(model.limits_valid(q))


if __name__ == "__main__":
    unittest.main()
