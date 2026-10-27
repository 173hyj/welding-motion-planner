"""Independent source-gun TCP registration; display LOD is deliberately unused."""
import base64
import json
import struct
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
from build_scene import flatten
from welding_kinematics import RobotKinematics


class TCPRegistrationTests(unittest.TestCase):
    def _source_electrode(self, gun):
        raw = json.loads((ROOT / "build" / (gun["name"] + ".json")).read_text())
        roots = [n["id"] for n in raw["nodes"] if n.get("type") == "InstanceNodeData"
                 and "K-07-0001.asm" in n["properties"].get("JT_PROP_NAME", "")]
        mesh = trimesh.util.concatenate([flatten(raw, [root]) for root in roots])
        mesh.apply_transform(np.asarray(gun["transform"], dtype=float).reshape(4, 4))
        return mesh

    def test_inverse_tcp_relative_frame_hits_source_electrode(self):
        scene = json.loads((ROOT / "build" / "scene.json").read_text())
        robot_raw = json.loads((ROOT / "build" / "robot.json").read_text())
        blob = base64.b64decode(robot_raw["rawPmi"][1]["bytes"])
        values = struct.unpack_from("<9f", blob, 112)
        pmi_tool = np.eye(4)
        pmi_tool[:3, 3] = values[:3]
        pmi_tool[:3, 0] = values[3:6]
        pmi_tool[:3, 1] = values[6:9]
        pmi_tool[:3, 2] = np.cross(pmi_tool[:3, 0], pmi_tool[:3, 1])
        robots = [asset for asset in scene["assets"] if asset["kind"] == "robot"]
        self.assertEqual(len(robots), 2)
        for robot in robots:
            kinematics = RobotKinematics(scene, robot)
            gun = next(asset for asset in scene["assets"] if asset["kind"] == "gun"
                       and ("R1" if "R1" in robot["name"] else "R2") in asset["name"])
            electrode = self._source_electrode(gun)
            tcp_relative = np.asarray(robot["robot"]["tcp_relative_frame"], dtype=float).reshape(4, 4)
            # PMI TOOL446 is the flange-local source frame. EngData TCPRelFrame
            # points from that frame to the engineering TCP reference, hence inverse.
            tcp_home = pmi_tool @ np.linalg.inv(tcp_relative)
            tcp = kinematics.base @ kinematics.links(kinematics.q_initial)["245"] @ tcp_home
            _, distance, _ = trimesh.proximity.closest_point_naive(electrode, tcp[:3, 3][None])
            self.assertLess(float(distance[0]), 0.01, robot["name"])
            wrong = kinematics.base @ kinematics.links(kinematics.q_initial)["245"] @ pmi_tool @ tcp_relative
            _, wrong_distance, _ = trimesh.proximity.closest_point_naive(electrode, wrong[:3, 3][None])
            self.assertGreater(float(wrong_distance[0]), 100., robot["name"])

    def test_source_weld_registration_pairs_all_points(self):
        scene = json.loads((ROOT / "build" / "scene.json").read_text())
        self.assertEqual(len(scene["welds"]), 29)
        for weld in scene["welds"]:
            self.assertIn("product_id", weld)
            self.assertIn("world_xyz", weld)
            self.assertLess(float(weld["registration_error_mm"]), 0.1)


if __name__ == "__main__":
    unittest.main()
