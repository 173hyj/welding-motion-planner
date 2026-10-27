"""Export original meshes and source scene coordinates, in millimetres."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from import_project import xml_root

ROOT = Path(__file__).resolve().parent


def matrix(position, angles):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = position
    return result


def values(element):
    return [float(x.text) for x in element] if element is not None else []


def flatten(data, root_ids=None, skip_root_transform=False, excluded_node_ids=None):
    nodes = {n["id"]: n for n in data["nodes"]}
    roots = root_ids if root_ids is not None else [n["id"] for n in nodes.values() if n["parent"] is None]
    result = []

    def visit(key, transform, color, first=False):
        if excluded_node_ids and key in excluded_node_ids:
            return
        node = nodes[key]
        if node["transform"] and not (first and skip_root_transform):
            transform = transform @ np.array(node["transform"]).reshape(4, 4).T
        color = node["color"] or color
        for mesh_id in node["meshes"]:
            mesh = data["meshes"][mesh_id]
            vertices = np.array(mesh["vertices"]).reshape(-1, 3)
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            faces = np.array(mesh["indices"]).reshape(-1, 3)
            keep = np.logical_and(faces[:, 0] != faces[:, 1], np.logical_and(faces[:, 1] != faces[:, 2], faces[:, 0] != faces[:, 2]))
            item = trimesh.Trimesh(vertices=vertices, faces=faces[keep], process=False)
            item.visual.face_colors = np.array(color) * 255
            result.append(item)
        children = node["children"]
        if node.get("type") == "RangeLODNodeData":
            children = children[:1]
        for child in children:
            visit(child, transform, color)

    for key in roots:
        visit(key, np.eye(4), [0.65, 0.68, 0.7, 1], True)
    return trimesh.util.concatenate(result) if result else None


def assemble(export=True):
    ps = xml_root((ROOT / "source/psz/StandaloneStudy_PsState.xml").read_bytes())
    eng = xml_root((ROOT / "source/psz/EngData/engData_ver2.xml").read_bytes())
    objects = list(ps.find("Objects"))
    index = {x.get("ExternalId"): x for x in objects}
    devices = {x.get("ID").split("#")[-1]: x for x in eng.find("devices")}
    raw_robot = json.loads((ROOT / "build/robot.json").read_text())
    kin_text = raw_robot["kinematicsSource"]
    kin = ET.fromstring(kin_text[kin_text.index("<devices>"):kin_text.index("</devices>") + 10]).find("device")
    kin_joints = []
    for j in kin.find("joints"):
        kin_joints.append({"key": j.get("key"), "parentLink": int(j.findtext("parentLink")), "childLink": int(j.findtext("childLink")),
            "type": int(j.findtext("jointType")), "axis1": values(j.find("axis1")), "axis2": values(j.find("axis2")),
            "maxSpeed": float(j.findtext("maxSpeed")), "range": ET.tostring(j.find("jointRange"), encoding="unicode"),
            "leadingJoints": [x.text for x in j.findall("leadingJoints/joint")], "leadingFactors": values(j.find("leadingFactors"))})
    raw_files = {"LZ-L-OP10-SUB1": "product1", "ST1-LZ-OP10L": "fixtureL", "robot": "robot"}
    output = {"units": "mm", "source_state": "StandaloneStudy_PsState.xml", "assets": [], "welds": [], "joints": kin_joints,
        "validation": {"collision_status": "not_yet_evaluated", "coordinate_convention": "extrinsic xyz Euler radians; row-major world matrices"}}
    mesh_dir = ROOT / "build/meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    link_models = {}
    for link in range(239, 248):
        node = next(n for n in raw_robot["nodes"] if n["properties"].get("TxGmProtoKey") == str(link))
        root = node if node["parent"] is None else next(n for n in raw_robot["nodes"] if n["id"] == node["parent"])
        transform = np.array(root["transform"]).reshape(4, 4).T if root["transform"] else np.eye(4)
        url = f"meshes/robot_link_{link}.glb"
        if export:
            mesh = flatten(raw_robot, [root["id"]], skip_root_transform=True)
            if mesh is not None:
                mesh.export(mesh_dir / f"robot_link_{link}.glb")
        link_models[str(link)] = {"url": url, "current_transform": transform.flatten().tolist()}
    for o in objects:
        if o.tag not in ("PmToolInstance", "PmPartInstance"):
            continue
        name = o.findtext("name", "")
        layout = index[o.findtext("layout")]
        prototype = index[o.findtext("prototype")]
        rep = index.get(prototype.findtext("threeDRep"))
        ref = index.get(rep.findtext("file")) if rep is not None else None
        path = ref.findtext("fileName", "") if ref is not None else ""
        basename = path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".cojt")
        robot = prototype.tag == "Robot"
        if robot:
            basename = "robot"
        raw_file = ROOT / "build" / (raw_files.get(basename, basename) + ".json")
        if not raw_file.exists():
            continue
        location = values(layout.find("location"))
        rotation = values(layout.find("rotation"))
        transform = matrix(location, rotation)
        # PsState stores PmLayout location relative to its assembly parent.
        # NodeInfo/absoluteLocation is the resolved world pose and must be used
        # for standalone exported meshes; otherwise nested fixtures can be
        # displaced by the parent assembly offset (4.83 m in this scene).
        absolute = layout.find("NodeInfo/absoluteLocation")
        if absolute is not None and all(absolute.findtext(axis) is not None
                                        for axis in ("x", "y", "z", "rx", "ry", "rz")):
            transform = matrix([float(absolute.findtext(axis)) for axis in ("x", "y", "z")],
                               [float(absolute.findtext(axis)) for axis in ("rx", "ry", "rz")])
        asset = {"id": o.get("ExternalId"), "name": name, "kind": "robot" if robot else "product" if o.tag == "PmPartInstance" else "gun" if prototype.tag == "Gun" else "fixture" if prototype.tag == "Fixture" else "layout",
            "source": path, "transform": transform.flatten().tolist()}
        if robot:
            device = devices[o.get("ExternalId")]
            asset["robot"] = {"q_initial": values(device.find("poses/currentPose")), "link_models": link_models,
                "tcp_current": [float(x) for x in device.findtext("tcpf/transformation").split()],
                "tcp_relative_frame": [float(x) for x in device.findtext("tcpRelFrame/transformation").split()]}
            # Native TOOLFRAME446 PMI is the flange-local frame. Its source binary stores
            # position then X/Y axes at offsets 112..148; the EngData TCPRelFrame is
            # expressed in this TCP frame and therefore composes by inverse.
            raw_pmi = json.loads((ROOT / "build/robot.json").read_text())["rawPmi"]
            if raw_pmi:
                import base64, struct
                blob = base64.b64decode(raw_pmi[1]["bytes"])
                vals = struct.unpack_from("<9f", blob, 112)
                tool = np.eye(4); tool[:3, 3] = vals[:3]; tool[:3, 0] = vals[3:6]; tool[:3, 1] = vals[6:9]; tool[:3, 2] = np.cross(tool[:3, 0], tool[:3, 1])
                asset["robot"]["tcp_home"] = (tool @ np.linalg.inv(np.array(asset["robot"]["tcp_relative_frame"]).reshape(4, 4))).flatten().tolist()
        else:
            asset["url"] = f"meshes/{basename}.glb"
            if export:
                raw = json.loads(raw_file.read_text())
                mesh = flatten(raw)
                if mesh is not None:
                    mesh.export(mesh_dir / f"{basename}.glb")
                    asset["triangles"] = len(mesh.faces)
        output["assets"].append(asset)
    csv_points = {}
    for path in (ROOT / "source/outer/上航发/AP31-LZ/LibraryRoot/MFG").glob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                csv_points[row["name"]] = [float(row[k]) for k in ("x", "y", "z")]
    operation_index = {x.findtext("name"): x for x in objects if x.tag == "PmWeldLocationOperation"}
    for w in objects:
        if w.tag != "PmWeldPoint" or w.findtext("name") not in csv_points:
            continue
        name = w.findtext("name")
        op = operation_index[name]
        product_id = w.findtext("operatesOn/item")
        asset = next(a for a in output["assets"] if a["id"] == product_id)
        transform = np.array(asset["transform"]).reshape(4, 4)
        local = np.array(csv_points[name])
        world = np.array(values(w.find("location")))
        group = index[op.findtext("parent")]
        sequence = [x.text for x in group.find("children")].index(op.get("ExternalId"))
        projected = [float(x) for x in op.findtext("projectedLocation").split("_")]
        output["welds"].append({"id": w.get("ExternalId"), "name": name, "group": group.findtext("name"), "product_id": product_id,
            "local_xyz": local.tolist(), "world_xyz": world.tolist(), "rotation": values(w.find("rotation")),
            "projected_local_pose": projected, "sequence": sequence,
            "registration_error_mm": float(np.linalg.norm(transform[:3, :3] @ local + transform[:3, 3] - world))})
    output["welds"].sort(key=lambda w: (w["group"], w["sequence"]))
    if (ROOT / 'build/gunR1_detail.json').exists() and (ROOT / 'build/gunR2_detail.json').exists():
        from gun_kinematics import attach_guns
        attach_guns(output, export=export)
    if export:
        (ROOT / "build/scene.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    result = assemble()
    print(f"Exported {len(result['assets'])} original assets and {len(result['welds'])} registered welds")
