"""Validate bounded joint-space transit between strict endpoint candidates."""
import json
from pathlib import Path
import numpy as np
from collision_audit import CollisionWorld
from solve_welds import solve

ROOT = Path(__file__).resolve().parent
scene = json.loads((ROOT / "build/scene.json").read_text())
strict = json.loads((ROOT / "build/collision_plan_source_normal_roll_optimized.json").read_text())
safe = [x for x in strict["candidates"] if x["collision_free_environment"]]
by_name = {}
for x in safe:
    by_name.setdefault(x["weld_id"], []).append(x)
sequence = ["OP10L-SUB1-POINT-GW-7", "OP10L-SUB1-POINT-GW-6", "OP10L-SUB1-POINT-GW-8", "OP10L-SUB2-POINT-GW-3", "OP10L-SUB2-POINT-GW-4", "OP10L-SUB2-POINT-GW-5"]
world = CollisionWorld(scene)
items = []
for name in sequence:
    options = by_name.get(name, [])
    if not options:
        items.append({"weld_id": name, "status": "endpoint_collision_rejected"})
        continue
    chosen = options[0]
    items.append({"weld_id": name, "robot_id": chosen["robot_id"], "q_rad": chosen["q_rad"], "status": "endpoint_collision_valid", "collision_pairs": chosen["collision_pairs"]})
for prev, curr in zip(items, items[1:]):
    if "q_rad" not in prev or "q_rad" not in curr or prev.get("robot_id") != curr.get("robot_id"):
        continue
    q0, q1 = np.array(prev["q_rad"]), np.array(curr["q_rad"])
    failure = None
    for i in range(1, 21):
        q = q0 + (q1 - q0) * i / 20
        pairs = world.check(curr["robot_id"], q)
        if pairs:
            failure = {"sample": i, "pairs": pairs[:3]}
            break
    curr["transit_valid"] = failure is None
    if failure:
        curr["transit_failure"] = failure
        curr["status"] = "transit_rejected"
report = {"diagnostic_only": True, "tcp_verified": True, "collision_verified": False, "endpoint_collision_verified": True,
    "transit_collision_verified": False, "source_initial_unsafe": True,
    "alternative_simulation_start": True, "start_note": "No path is asserted from the supplied current pose; endpoint sequences start at a separately selected validated pose.",
    "endpoint_count": len(safe), "selected_sequence": items, "strict_audit": "collision_plan_source_normal_roll_optimized.json"}
approaches = []
for item in items:
    if "q_rad" not in item:
        continue
    model = world.models[item["robot_id"]]
    q_weld = np.array(item["q_rad"])
    pose = model.tcp(q_weld)
    weld = next(w for w in scene["welds"] if w["id"] == item["weld_id"])
    attempts = []
    for sign in (1, -1):
        target = pose.copy()
        target[:3, 3] += sign * 100 * pose[:3, 2]
        candidates = solve(model, target, q_weld, 6)
        for candidate in candidates:
            q_approach = np.array(candidate["q_rad"])
            failure = None
            for i in range(20, 0, -1):
                q = q_weld + (q_approach - q_weld) * i / 20
                pairs = world.check(item["robot_id"], q)
                if pairs:
                    failure = {"sample": i, "pairs": pairs[:2]}
                    break
            attempts.append({"sign": sign, "q_approach": q_approach.tolist(), "transit_valid": failure is None, "failure": failure})
            if failure is None:
                break
        if any(a["transit_valid"] for a in attempts):
            break
    approaches.append({"weld_id": item["weld_id"], "attempts": attempts, "approach_valid": any(a["transit_valid"] for a in attempts)})
report["approach_retreat_100mm"] = approaches
(ROOT / "build/welding_plan_strict.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("transit segments", sum(x.get("transit_valid", False) for x in items), "of", max(0, len(items)-1))
