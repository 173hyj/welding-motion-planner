"""Original TxKin gun driver and three source linkage equations, in mm/rad."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


class GunKinematics:
    def __init__(self, raw):
        text = raw['kinematicsSource']
        self.source_xml = text[text.index('<devices>'):text.index('</devices>') + 10]
        device = ET.fromstring(self.source_xml).find('device')
        self.joints = list(device.find('joints'))
        if len(self.joints) != 4:
            raise ValueError('Expected original four-joint gun mechanism')
        self.limits_mm = tuple(float(self.joints[0].findtext(f'jointRange/{k}')) for k in ('lowLimit', 'highLimit'))
        self.link_nodes = {n['properties']['TxGmProtoKey']: n for n in raw['nodes'] if n['properties'].get('TxObjType') == 'TxLink'}
        self.root_key = self.joints[0].findtext('parentLink')
        self.closed_values = self.joint_values(0.)
        self.home = self._absolute(self.closed_values)

    def joint_values(self, driver_mm):
        if not self.limits_mm[0] <= driver_mm <= self.limits_mm[1]:
            raise ValueError('Gun driver outside original TxKin limits')
        # Constants and functions below are copied from the source XML. The
        # empty driver link feeds the arm hinge; cylinder and rod follow it.
        theta = math.asin(driver_mm / 485.)
        length = math.sqrt(123190.999924 - 121440.952596 * math.cos(.686251 + theta))
        rod = length - 171.
        cylinder = .995203 - math.acos((143882.000258116 - 121440.952596 * math.cos(.686251 + theta)) / (536.43639 * length))
        return [driver_mm, theta, rod, cylinder]

    def _absolute(self, values):
        result = {self.root_key: np.eye(4)}
        pending = list(zip(self.joints, values))
        while pending:
            ready = [(j, v) for j, v in pending if j.findtext('parentLink') in result]
            if not ready:
                raise ValueError('Disconnected source gun mechanism')
            for joint, value in ready:
                point = np.array([float(joint.findtext('axis1/' + k)) for k in ('x', 'y', 'z')])
                tip = np.array([float(joint.findtext('axis2/' + k)) for k in ('x', 'y', 'z')])
                direction = (tip - point) / np.linalg.norm(tip - point)
                motion = np.eye(4)
                if joint.findtext('jointType') == '1':
                    motion[:3, 3] = direction * value
                else:
                    motion[:3, :3] = Rotation.from_rotvec(direction * value).as_matrix()
                    motion[:3, 3] = point - motion[:3, :3] @ point
                result[joint.findtext('childLink')] = result[joint.findtext('parentLink')] @ motion
                pending.remove((joint, value))
        return result

    def transforms(self, driver_mm):
        """Rigid deltas applied to each exact closed-source link mesh."""
        absolute = self._absolute(self.joint_values(float(driver_mm)))
        return {key: pose @ np.linalg.inv(self.home[key]) for key, pose in absolute.items()}


def gun_mesh_parts(raw):
    """Partition source geometry by TxLink and named electrode subtree."""
    from build_scene import flatten
    model = GunKinematics(raw)
    nodes = {n['id']: n for n in raw['nodes']}
    parts = {}
    for key, node in model.link_nodes.items():
        subtree = set()
        def visit(nid):
            subtree.add(nid)
            for child in nodes[nid]['children']:
                visit(child)
        visit(node['id'])
        caps = [n for nid, n in nodes.items() if nid in subtree and n['type'] == 'InstanceNodeData' and 'K-07-0001.asm' in n['properties'].get('JT_PROP_NAME', '')]
        # Flatten from actual link wrapper, retaining its source closed pose.
        root_id = node['id'] if node['parent'] in (None, 1) else node['parent']
        body = flatten(raw, [root_id], excluded_node_ids={n['id'] for n in caps})
        if body is not None:
            parts[f'{key}_body'] = (key, 'body', body)
        for cap in caps:
            # Include ancestor transforms, then crop exactly the cap subtree.
            cap_mesh = flatten(raw, [cap['id']])
            ancestor = nodes[cap['parent']]
            transform = np.eye(4)
            chain = []
            while True:
                chain.append(ancestor)
                if ancestor['parent'] is None:
                    break
                ancestor = nodes[ancestor['parent']]
            for ancestor in reversed(chain):
                if ancestor['transform']:
                    transform = transform @ np.array(ancestor['transform']).reshape(4, 4).T
            cap_mesh.apply_transform(transform)
            parts[f'{key}_electrode_{cap["id"]}'] = (key, 'electrode', cap_mesh)
    return model, parts


def attach_guns(scene, export=True):
    """Add real part URLs and kinematics to a scene, retaining source gun frame."""
    for asset in scene['assets']:
        if asset['kind'] != 'gun':
            continue
        label = 'R1' if 'R1' in asset['name'] else 'R2'
        raw_path = ROOT / f'build/gun{label}_detail.json'
        raw = json.loads(raw_path.read_text())
        model, parts = gun_mesh_parts(raw)
        links = {}
        for part_id, (key, kind, mesh) in parts.items():
            url = f'meshes/gun_{label}_{part_id}.glb'
            if export:
                mesh.export(ROOT / 'build' / url)
            links[part_id] = {'url': url, 'link_key': key, 'kind': kind, 'triangles': len(mesh.faces), 'current_transform': np.eye(4).flatten().tolist()}
        asset['gun'] = {'source_detail': str(raw_path.relative_to(ROOT)), 'source_xml': model.source_xml, 'driver_units': 'mm', 'driver_initial': 0., 'driver_limits_mm': list(model.limits_mm), 'link_models': links}
    return scene


if __name__ == '__main__':
    scene_path = ROOT / 'build/scene.json'
    scene = attach_guns(json.loads(scene_path.read_text()))
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2))
    print('Attached original gun mechanisms and source part meshes')
