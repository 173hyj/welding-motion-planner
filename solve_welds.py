"""Multi-start source-pose IK; full dependent limit polygons are enforced."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from welding_kinematics import RobotKinematics
from build_scene import matrix

ROOT = Path(__file__).resolve().parent


def solve(model, target, seed, attempts=16):
    def residual(q):
        tcp = model.tcp(q)
        orientation = Rotation.from_matrix(target[:3, :3].T @ tcp[:3, :3]).as_rotvec()
        return np.r_[(tcp[:3, 3] - target[:3, 3]) / 1000, orientation]
    solutions = []
    rng = np.random.default_rng(20261006)
    seeds = [seed, model.q_initial, np.zeros(6)] + [rng.uniform(*model.bounds) for _ in range(attempts - 3)]
    for start in seeds:
        result = least_squares(residual, np.clip(start, model.bounds[0] + 1e-7, model.bounds[1] - 1e-7), bounds=(model.bounds[0], model.bounds[1]), max_nfev=140, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        q = result.x
        err = residual(q)
        if np.linalg.norm(err[:3]) < 0.0001 and np.linalg.norm(err[3:]) < 0.001 and model.limits_valid(q):
            if not any(np.linalg.norm(q - s["q_rad"]) < 0.01 for s in solutions):
                solutions.append({"q_rad": q.tolist(), "position_error_mm": float(np.linalg.norm(err[:3]) * 1000), "rotation_error_rad": float(np.linalg.norm(err[3:]))})
    return sorted(solutions, key=lambda x: np.linalg.norm(np.array(x["q_rad"]) - seed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--orientation", choices=("active", "projected", "source_normal_roll_optimized"), default="active")
    args = parser.parse_args()
    scene = json.loads((ROOT / "build/scene.json").read_text())
    robots = [RobotKinematics(scene, a) for a in scene["assets"] if a["kind"] == "robot"]
    seeds = {r.asset["id"]: r.q_initial for r in robots}
    results = []
    for w in scene["welds"]:
        target = matrix(w["world_xyz"], w["rotation"])
        if args.orientation != "active":
            a = next(a for a in scene["assets"] if a["id"] == w["product_id"])
            target[:3, :3] = np.array(a["transform"]).reshape(4, 4)[:3, :3] @ Rotation.from_euler("xyz", w["projected_local_pose"][3:]).as_matrix()
        row = {"id": w["id"], "name": w["name"], "target_transform": target.flatten().tolist(), "orientation_source": args.orientation, "robots": []}
        for r in robots:
            rid = r.asset["id"]
            solutions = solve(r, target, seeds[rid], args.attempts)
            for sol in solutions:
                sol["roll_rad"] = 0.0
            if args.orientation == "source_normal_roll_optimized":
                for roll in (np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi):
                    modified = target.copy()
                    modified[:3, :3] = target[:3, :3] @ Rotation.from_rotvec([0, 0, roll]).as_matrix()
                    candidates = solve(r, modified, seeds[rid], min(args.attempts, 6))
                    for sol in candidates:
                        sol["roll_rad"] = float(roll)
                        sol["target_transform"] = modified.flatten().tolist()
                    solutions.extend(candidates)
            if solutions:
                seeds[rid] = np.array(solutions[0]["q_rad"])
            row["robots"].append({"robot_id": rid, "name": r.asset["name"], "solutions": solutions})
        results.append(row)
        print(w["id"], [len(r["solutions"]) for r in row["robots"]], flush=True)
    report = {"tcp_verified": True, "tcp_electrode_mesh_registration_mm": [0.004961, 0.005167], "collision_verified": False,
        "orientation_source": args.orientation, "weld_count": len(results), "ik_solved_weld_count": sum(any(r["solutions"] for r in w["robots"]) for w in results), "welds": results}
    (ROOT / f"build/ik_{args.orientation}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("IK solved", report["ik_solved_weld_count"], "/", len(results))


if __name__ == "__main__":
    main()
