"""Exact source mesh FCL checks; no proxy replacement geometry."""
import json
from pathlib import Path
import numpy as np
import trimesh
import fcl
from welding_kinematics import RobotKinematics
from build_scene import flatten
from gun_kinematics import GunKinematics, gun_mesh_parts

ROOT = Path(__file__).resolve().parent


def fcl_object(mesh, transform):
    model = fcl.BVHModel()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return fcl.CollisionObject(model, fcl.Transform(transform[:3, :3], transform[:3, 3]))


class CollisionWorld:
    def __init__(self, scene):
        self.scene = scene
        self.static = []
        self.robots = {}
        self.models = {}
        self.triangles = 0
        assets = scene["assets"]
        # Crop the source environment to reachable workcell bounds, preserving each retained face exactly.
        bases = np.array([np.array(a["transform"]).reshape(4, 4)[:3, 3] for a in assets if a["kind"] == "robot"])
        lo = bases.min(axis=0) - 5000
        hi = bases.max(axis=0) + 5000
        for a in assets:
            if a["kind"] in ("robot", "gun"):
                continue
            mesh = trimesh.load(ROOT / "build" / a["url"], force="mesh", process=False)
            t = np.array(a["transform"]).reshape(4, 4)
            mesh.apply_transform(t)
            if a["kind"] == "layout":
                triangles = mesh.vertices[mesh.faces]
                keep = np.all(triangles.max(axis=1) >= lo, axis=1) & np.all(triangles.min(axis=1) <= hi, axis=1)
                mesh = mesh.submesh([np.flatnonzero(keep)], append=True, repair=False)
            self.triangles += len(mesh.faces)
            self.static.append((a["id"], a["name"], fcl_object(mesh, np.eye(4))))
            print("FCL source asset", a["name"], len(mesh.faces), flush=True)
        for a in assets:
            if a["kind"] != "robot":
                continue
            k = RobotKinematics(scene, a)
            links = k.links(k.q_initial, world=True)
            objects = {}
            for key, data in a["robot"]["link_models"].items():
                mesh = trimesh.load(ROOT / "build" / data["url"], force="mesh", process=False)
                objects[key] = fcl_object(mesh, links[key])
            gun = next(g for g in assets if g["kind"] == "gun" and ("R1" if "R1" in a["name"] else "R2") in g["name"])
            gun_transform = np.array(gun["transform"]).reshape(4, 4)
            k.gun_id = gun["id"]
            k.gun_home = np.linalg.inv(links["245"]) @ gun_transform
            if gun.get("gun"):
                raw = json.loads((ROOT / gun['gun']['source_detail']).read_text())
                k.gun_model, parts = gun_mesh_parts(raw)
                k.gun_parts = {pid: {'link_key': key, 'kind': kind} for pid, (key, kind, _) in parts.items()}
                for pid, (_, _, mesh) in parts.items():
                    objects['gun:' + pid] = fcl_object(mesh, gun_transform)
                self.robots[a["id"]] = objects
                self.models[a["id"]] = k
                print('FCL original articulated gun', gun['name'], len(parts), sum(len(p[2].faces) for p in parts.values()), flush=True)
                continue
            gun_mesh = trimesh.load(ROOT / "build" / gun["url"], force="mesh", process=False)
            raw_path = ROOT / "build" / (gun["name"] + ".json")
            raw = json.loads(raw_path.read_text())
            caps = [n for n in raw["nodes"] if n.get("type") == "InstanceNodeData" and "K-07-0001.asm" in n["properties"].get("JT_PROP_NAME", "")]
            # Remove precisely the named source component subtrees; body geometry
            # inside an electrode bounding box cannot accidentally be exempted.
            body = flatten(raw, excluded_node_ids={c["id"] for c in caps})
            electrode = trimesh.util.concatenate([flatten(raw, [c["id"]]) for c in caps])
            objects["gun"] = fcl_object(body, gun_transform)
            if len(electrode.faces):
                objects["electrode"] = fcl_object(electrode, gun_transform)
            k.electrode_source_components = [{"node_id": c["id"], "name": c["properties"]["JT_PROP_NAME"]} for c in caps]
            print("FCL source electrodes", gun["name"], len(electrode.faces), "body", len(body.faces), flush=True)
            self.robots[a["id"]] = objects
            self.models[a["id"]] = k

    def update(self, robot_id, q, gun_driver_mm=0):
        k = self.models[robot_id]
        links = k.links(q, world=True)
        objects = self.robots[robot_id]
        for key, obj in objects.items():
            if key.startswith('gun:'):
                link_key = k.gun_parts[key[4:]]['link_key']
                t = links['245'] @ k.gun_home @ k.gun_model.transforms(gun_driver_mm)[link_key]
            else:
                t = links[key] if key not in ("gun", "electrode") else links["245"] @ k.gun_home
            obj.setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
        return links

    def check(self, robot_id, q, margin_mm=0, max_pairs=30, weld=None, gun_driver_mm=0):
        self.update(robot_id, q, gun_driver_mm)
        objects = self.robots[robot_id]
        other = [(rid + ":" + key, "other_robot", ob) for rid, obs in self.robots.items() if rid != robot_id for key, ob in obs.items()]
        pairs = []
        for key, obj in objects.items():
            if key == "239":
                # Robot mounting base is intentionally mounted in the source layout.
                continue
            for oid, name, environment in self.static + other:
                is_electrode = key == 'electrode' or (key.startswith('gun:') and self.models[robot_id].gun_parts[key[4:]]['kind'] == 'electrode')
                if is_electrode and weld is not None and oid == weld["product_id"]:
                    # Intentional contact is restricted to original K-07-0001 electrode
                    # components, the designated workpiece, and a verified TCP at the weld.
                    if np.linalg.norm(self.models[robot_id].tcp(q)[:3, 3] - np.array(weld["world_xyz"])) < 0.1:
                        contacts = fcl.CollisionResult()
                        fcl.collide(obj, environment, fcl.CollisionRequest(num_max_contacts=10000, enable_contact=True), contacts)
                        if contacts.is_collision and len(contacts.contacts) < 10000 and all(np.linalg.norm(c.pos - np.array(weld["world_xyz"])) <= 25 and c.penetration_depth <= 2 for c in contacts.contacts):
                            continue
                req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
                result = fcl.CollisionResult()
                fcl.collide(obj, environment, req, result)
                if result.is_collision:
                    contact = result.contacts[0] if result.contacts else None
                    pairs.append({"link": key, "asset_id": oid, "asset_name": name,
                        "contact_xyz": contact.pos.tolist() if contact is not None else None,
                        "penetration_mm": float(contact.penetration_depth) if contact is not None else None})
                elif margin_mm > 0:
                    distance = fcl.distance(obj, environment, fcl.DistanceRequest(), fcl.DistanceResult())
                    if distance < margin_mm:
                        pairs.append({"link": key, "asset_id": oid, "asset_name": name, "distance_mm": distance})
                if len(pairs) >= max_pairs:
                    return pairs
        # Source link assembly has intentional contacts; adjacent joints and cylinder mounts are omitted.
        keys = [str(i) for i in range(240, 246)]
        for i, ka in enumerate(keys):
            for kb in keys[i + 2:]:
                result = fcl.CollisionResult()
                fcl.collide(objects[ka], objects[kb], fcl.CollisionRequest(), result)
                if result.is_collision:
                    pairs.append({"link": ka, "asset_id": kb, "asset_name": "self"})
        # Gun-to-robot checks use every original part; only the flange mount is
        # an intentional assembled interface. Gun linkage internal interfaces
        # are not counted as robot self-collision certification.
        gun_keys = [key for key in objects if key.startswith('gun:') or key in ('gun', 'electrode')]
        for ka in keys[:-1]:
            for kb in gun_keys:
                result = fcl.CollisionResult()
                fcl.collide(objects[ka], objects[kb], fcl.CollisionRequest(), result)
                if result.is_collision:
                    pairs.append({'link': ka, 'asset_id': kb, 'asset_name': 'self'})
        return pairs


def main():
    scene = json.loads((ROOT / "build/scene.json").read_text())
    world = CollisionWorld(scene)
    report = {"method": "FCL triangle BVH, exact source meshes; reachable layout crop ±5000mm; no decimation", "static_triangles": world.triangles, "robots": []}
    for rid, model in world.models.items():
        pairs = world.check(rid, model.q_initial)
        report["robots"].append({"id": rid, "name": model.asset["name"], "current_pose_collision_pairs": pairs})
        print(model.asset["name"], "collision pairs", len(pairs), flush=True)
    (ROOT / "build/collision_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
