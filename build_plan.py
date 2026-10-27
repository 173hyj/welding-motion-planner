"""Source-grounded diagnostic FK sweep, without claiming verified welding."""
import json
from pathlib import Path
import numpy as np
import argparse
from welding_kinematics import RobotKinematics
from gun_kinematics import GunKinematics

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "build/fk_diagnostic_plan.json"); args = parser.parse_args()
    scene = json.loads((ROOT / "build/scene.json").read_text())
    robots = [a for a in scene["assets"] if a["kind"] == "robot"]
    models = {a["id"]: RobotKinematics(scene, a) for a in robots}
    mounts = {}
    gun_models = {}
    for asset in robots:
        k = models[asset["id"]]
        gun = next(g for g in scene["assets"] if g["kind"] == "gun" and ("R1" if "R1" in asset["name"] else "R2") in g["name"])
        mounts[asset["id"]] = (gun["id"], np.linalg.inv(k.links(k.q_initial, world=True)["245"]) @ np.array(gun["transform"]).reshape(4, 4))
        gun_models[gun["id"]] = GunKinematics(json.loads((ROOT / gun["gun"]["source_detail"]).read_text()))
    frames = []
    for t, delta in enumerate(np.linspace(0, 2 * np.pi, 25)):
        robots_frame, guns = {}, {}
        # The diagnostic sweep also drives each original gun mechanism through
        # its source range.  This is a visualization of the exact TxKin DOF,
        # not a collision-free welding trajectory.
        driver = -185.0 * (0.5 - 0.5 * np.cos(2 * delta))
        for asset in robots:
            k = models[asset["id"]]
            q = np.array(k.q_initial, dtype=float)
            q[0] += 0.0872664626 * np.sin(delta)
            links = k.links(q, world=True)
            robots_frame[asset["id"]] = {"q_rad": q.tolist(), "links": {key: value.flatten().tolist() for key, value in links.items()}}
            gun_id, mount = mounts[asset["id"]]
            gun_pose = links["245"] @ mount
            model = gun_models[gun_id]
            deltas = model.transforms(driver)
            parts = {}
            for part_id, part in next(g for g in scene["assets"] if g["id"] == gun_id)["gun"]["link_models"].items():
                parts[part_id] = (gun_pose @ deltas[part["link_key"]]).flatten().tolist()
            guns[gun_id] = {"transform": gun_pose.flatten().tolist(), "driver_mm": float(driver), "parts": parts}
        frames.append({"time_s": float(t * 0.1), "robots": robots_frame, "guns": guns})
    plan = {"status": {"diagnostic_only": True, "tcp_verified": True, "collision_verified": False, "weld_reachability_verified": True,
        "reason": "Frames are FK-only diagnostic replay from original robot axes and TxKin gun opening DOF. Source initial pose has potential self-collisions; endpoint candidates are documented separately in collision_plan_source_normal_roll_optimized.json."},
        "units": "mm/rad", "frames": frames, "welds": scene["welds"], "collision_audit": "collision_audit.json"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote diagnostic FK plan with {len(frames)} frames and {len(scene['welds'])} source weld markers")


if __name__ == "__main__":
    main()
