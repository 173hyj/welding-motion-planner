#!/usr/bin/env python3
"""Conservative analytic reach envelope for the original source-mesh crop.

For POE rotations around anchors p_i, the moved vertex equals p_1 plus
rotated differences p_(i+1)-p_i plus a rotated final v-p_n. The triangle
inequality bounds its radius for every rotation, including coupled axes.
Prismatic translations add their maximum absolute displacement. This is
a crop-sufficiency check, not a collision or robot-dynamics certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from welding_kinematics import RobotKinematics
from gun_kinematics import GunKinematics


def chain_radius(anchors, vertices, translation_bound=0.):
    anchors = np.asarray(anchors, float).reshape(-1, 3)
    vertices = np.asarray(vertices, float).reshape(-1, 3)
    if not len(vertices):
        raise ValueError('No vertices in source mesh')
    if not len(anchors):
        return float(np.linalg.norm(vertices, axis=1).max() + translation_bound)
    fixed = np.linalg.norm(anchors[0]) + np.linalg.norm(np.diff(anchors, axis=0), axis=1).sum()
    return float(fixed + np.linalg.norm(vertices - anchors[-1], axis=1).max() + translation_bound)


def projection(point, origin, direction):
    direction = np.asarray(direction, float)
    return origin + direction * (np.dot(point - origin, direction) / np.dot(direction, direction))


def choose_robot_anchors(joints):
    # Any point on the same infinite source axis defines the same rotation.
    # Align the wrist axes around source J5 to avoid a needlessly loose bound.
    reference = np.asarray(joints[4]['axis1'], float)
    reference = projection(np.zeros(3), reference, np.asarray(joints[4]['axis2']) - reference)
    anchors = []
    for i, j in enumerate(joints):
        origin = np.asarray(j['axis1'], float)
        anchors.append(projection(np.zeros(3) if i < 3 else reference, origin, np.asarray(j['axis2']) - origin))
    return np.asarray(anchors)


def transform_vertices(vertices, matrix):
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


def audit(scene_path, crop_mm=5000.):
    scene = json.loads(scene_path.read_text())
    records, hashes = [], {'scene': hashlib.sha256(scene_path.read_bytes()).hexdigest()}
    anchors = choose_robot_anchors(scene['joints'][:6])

    def vertices(url):
        path = ROOT / 'build' / url
        hashes[url] = hashlib.sha256(path.read_bytes()).hexdigest()
        return np.asarray(trimesh.load(path, force='mesh', process=False).vertices)

    for asset in scene['assets']:
        if asset['kind'] != 'robot':
            continue
        k = RobotKinematics(scene, asset)
        for key, part in asset['robot']['link_models'].items():
            v = vertices(part['url'])
            if key == '239':
                bound = chain_radius([], v)
                formula = 'static base vertices'
            elif 240 <= int(key) <= 245:
                bound = chain_radius(anchors[:int(key) - 239], v)
                formula = 'POE rotational chain triangle inequality'
            else:
                # The source counterbalance orientation varies, but rotation
                # preserves each vertex radius about its attachment endpoint.
                pin = np.asarray(scene['joints'][6 if key == '246' else 8]['axis1'])
                prefix = anchors[:2 if key == '246' else 1]
                bound = chain_radius(prefix, [pin]) + np.linalg.norm(v - pin, axis=1).max()
                formula = 'moving attachment envelope plus arbitrary rigid orientation'
            records.append({'robot_id': asset['id'], 'part': key, 'radius_bound_mm': float(bound), 'proof': formula})

        gun = next(a for a in scene['assets'] if a['kind'] == 'gun' and ('R1' if 'R1' in asset['name'] else 'R2') in a['name'])
        raw_path = ROOT / gun['gun']['source_detail']
        hashes[gun['gun']['source_detail']] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        gm = GunKinematics(json.loads(raw_path.read_text()))
        gun_home = np.linalg.inv(k.links(k.q_initial, world=True)['245']) @ np.asarray(gun['transform']).reshape(4, 4)
        by_child = {j.findtext('childLink'): j for j in gm.joints}
        for pid, part in gun['gun']['link_models'].items():
            key = part['link_key']
            v = transform_vertices(vertices(part['url']), np.linalg.inv(gm.home[key]))
            v = transform_vertices(v, gun_home)
            path, at = [], key
            while at != gm.root_key:
                j = by_child[at]
                path.insert(0, j)
                at = j.findtext('parentLink')
            combined, translations = list(anchors), 0.
            for j in path:
                if j.findtext('jointType') == '0':
                    p = np.array([float(j.findtext('axis1/' + c)) for c in 'xyz'])
                    tip = np.array([float(j.findtext('axis2/' + c)) for c in 'xyz'])
                    p = projection(np.zeros(3), p, tip - p)
                    combined.append(transform_vertices(np.asarray([p]), gun_home)[0])
                else:
                    # The only rendered prismatic descendant is the original
                    # rod: |sqrt(A-B*cos(theta))-171| <= max(|sqrt(A±B)-171|).
                    if j.findtext('function'):
                        translations += max(abs(np.sqrt(123190.999924 - 121440.952596) - 171.), abs(np.sqrt(123190.999924 + 121440.952596) - 171.))
                    else:
                        translations += max(abs(x) for x in gm.limits_mm)
            bound = chain_radius(combined, v, translations)
            records.append({'robot_id': asset['id'], 'part': 'gun:' + pid, 'radius_bound_mm': bound,
                            'translation_bound_mm': translations, 'proof': 'robot and source gun rotary chain; cosine rod displacement bound'})
    max_bound = max(r['radius_bound_mm'] for r in records)
    return {'scene': str(scene_path), 'source_sha256': hashes, 'crop_mm': crop_mm, 'robot_anchors_mm': anchors.tolist(),
            'parts': records, 'maximum_radius_bound_mm': max_bound, 'minimum_crop_margin_mm': crop_mm - max_bound,
            'all_dynamic_meshes_inside_crop': bool(max_bound + 1e-6 < crop_mm),
            'proof_scope': 'Every source mesh vertex for any rotational angles and declared prismatic displacement envelopes; thus every source triangle is contained in the crop. A layout face whose AABB misses the crop cannot collide with these meshes.',
            'limitations': ['Assumes source rigid parts and the recovered source kinematic graph; unmodelled flexible hoses or external objects are outside this audit.', 'This result audits cropping only, not pair exclusions, swept collisions or process safety.']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, default=ROOT / 'build/scene.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = audit(a.scene.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ['maximum_radius_bound_mm', 'minimum_crop_margin_mm', 'all_dynamic_meshes_inside_crop']}))


if __name__ == '__main__':
    main()
