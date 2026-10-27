"""Fresh FCL recheck, independently parsed source arm FK and positioner axis."""
import argparse
import copy
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT/'experimental/welding_endpoints_v1'),
               str(ROOT), str(ROOT/'build/eight_hour_v17/experimental_release/multi_point_map')]
from world import prepare_scene, OUT, sha
from collision_audit import CollisionWorld, fcl_object
from build_scene import flatten
from gun_kinematics import gun_mesh_parts
from source_face_world_adapter_v1 import _check_simultaneous_source_faces
from acceptance import posture_passed
spec = importlib.util.spec_from_file_location('source_fk', ROOT/'scripts/audit_contact_release_v17.py')
fk = importlib.util.module_from_spec(spec); spec.loader.exec_module(fk)


def audit_layout_state(world, relative_ids, moving_id, interface_ids):
    """Recheck source fixture/products against fixed geometry after independent FK."""
    objects = {oid: obj for oid, name, obj in world.static}
    blocked = []; interfaces = []
    for oid in relative_ids:
        for fixed_id, name, other in world.static:
            if fixed_id in relative_ids:
                continue
            result = fcl.CollisionResult()
            fcl.collide(objects[oid], other, fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), result)
            if not result.is_collision:
                continue
            contact = result.contacts[0] if result.contacts else None
            pair = dict(part=oid, asset_id=fixed_id, asset_name=name,
                        contact_xyz_mm=np.asarray(contact.pos).tolist() if contact else None)
            if oid == moving_id and fixed_id in interface_ids:
                interfaces.append(pair)
            else:
                blocked.append(pair)
    return dict(blocking_pairs=blocked, uncertified_assembly_interface_pairs=interfaces)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--allow-partial', action='store_true'); args = parser.parse_args()
    scene_path = ROOT/'build/scene_placement_corrected_v1.json'
    scene = json.loads(scene_path.read_text())
    path = OUT/'report.json'; report_bytes = path.read_bytes(); report = json.loads(report_bytes)
    assert report['scene_sha256'] == sha(scene_path)
    assert args.allow_partial or report['phase'] == 'complete'
    adapted, audit = prepare_scene(scene)
    assert report['geometry_audit_sha256'] == sha(OUT/'geometry_audit.json')
    world = CollisionWorld(adapted)
    fixture = next(a for a in scene['assets'] if a['name'] == 'ST1-LZ-OP10L')
    base = np.asarray(fixture['transform']).reshape(4, 4)
    raw = json.loads((ROOT/'build/positionerL_detail.json').read_text())
    text = raw['kinematicsSource']; text = text[text.index('<devices>'):text.index('</devices>')+10]
    joint = ET.fromstring(text).find('device/joints/joint')
    points = np.array([[float(joint.findtext(field+'/'+axis)) for axis in ('x','y','z')] for field in ('axis1','axis2')])
    points = points@base[:3, :3].T+base[:3, 3]
    axis = points[1]-points[0]; axis /= np.linalg.norm(axis)
    # Rebuild both fixture BVHs from original raw source meshes in their local
    # frame, independently of the search world's world-baked GLB objects.
    fixture_raw = json.loads((ROOT/'build/fixtureL.json').read_text())
    moving = fcl_object(flatten(fixture_raw, [2]), base)
    fixed = fcl_object(flatten(fixture_raw, [6196]), base)
    moving_id, fixed_id = fixture['id']+':DONG', fixture['id']+':JING'
    world.static = [(oid, name, moving if oid == moving_id else fixed if oid == fixed_id else obj)
                    for oid, name, obj in world.static]
    objects = {oid: obj for oid, name, obj in world.static}
    products = {}; parts = {}; verified = []
    joints = fk.joints_from_raw(json.loads((ROOT/'build/robot.json').read_text()))[:6]
    for rid, model in world.models.items():
        gun = next(a for a in scene['assets'] if a['id'] == model.gun_id)
        _, parts[rid] = gun_mesh_parts(json.loads((ROOT/gun['gun']['source_detail']).read_text()))
    for row in report['rows']:
        if not row['passed']:
            continue
        selected = row['selected']; deg = selected['positioner_deg']; roll = selected['roll_deg']
        assert deg % 5 == 0 and roll % 5 == 0
        rotation = Rotation.from_rotvec(axis*np.deg2rad(deg)).as_matrix()
        delta = np.eye(4); delta[:3, :3] = rotation; delta[:3, 3] = points[0]-rotation@points[0]
        T = delta@base; moving.setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))
        for product in scene['assets']:
            if product['kind'] != 'product':
                continue
            objects[product['id']].setTransform(fcl.Transform(delta[:3, :3], delta[:3, 3]))
        layout_id = next(a['id'] for a in scene['assets'] if a['kind'] == 'layout')
        layout_state = audit_layout_state(world,
            {moving_id, *(a['id'] for a in scene['assets'] if a['kind'] == 'product')},
            moving_id, {fixed_id, layout_id+':hanging_interface'})
        weld = next(w for w in scene['welds'] if w['id'] == row['weld_id'])
        product = next(a for a in scene['assets'] if a['id'] == weld['product_id'])
        product_T = np.asarray(product['transform']).reshape(4, 4)
        original_pose = np.eye(4); original_pose[:3, 3] = weld['world_xyz']
        original_pose[:3, :3] = product_T[:3, :3]@Rotation.from_euler('xyz', weld['projected_local_pose'][3:]).as_matrix()
        target = delta@original_pose
        moved = copy.deepcopy(weld); moved['world_xyz'] = target[:3, 3].tolist()
        if selected['axis_sign'] < 0:
            target[:3, :3] = target[:3, :3]@Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
        target[:3, :3] = target[:3, :3]@Rotation.from_rotvec([0, 0, np.deg2rad(roll)]).as_matrix()
        rid = selected['robot_id']; model = world.models[rid]; q = np.asarray(selected['q_rad'])
        independent_pose = model.base@fk.flange_source(q, joints)@model.tcp_home
        difference = float(np.max(np.abs(independent_pose-model.tcp(q))))
        assert difference < 1e-7
        error = float(np.linalg.norm(independent_pose[:3, 3]-target[:3, 3]))
        angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(target[:3, :3].T@independent_pose[:3, :3]).as_rotvec())))
        pid = weld['product_id']
        if pid not in products:
            mesh = trimesh.load(ROOT/'build'/product['url'], force='mesh', process=False); mesh.apply_transform(product_T)
            products[pid] = mesh
        mesh = products[pid].copy(); mesh.apply_transform(delta)
        state = _check_simultaneous_source_faces(world, robot_id=rid, other_robot_id=selected['other_robot_id'],
            q=q, other_q=np.asarray(selected['other_q_rad']), driver_mm=0., weld=moved, phase='closed',
            source_parts=parts[rid], product_mesh=mesh)
        keys = ['gun:'+p for p, info in model.gun_parts.items() if info['kind'] == 'electrode']
        gaps = {key: float(fcl.distance(world.robots[rid][key], objects[pid], fcl.DistanceRequest(),
                fcl.DistanceResult())) for key in keys if key not in state['certificates']}
        passed = posture_passed(state, error, keys, gaps) and angle < .01 and not layout_state['blocking_pairs']
        verified.append(dict(weld_id=weld['id'], name=weld['name'], passed=passed, tcp_error_mm=error,
            rotation_error_deg=angle, raw_fk_difference_mm=difference, remaining_pairs=state['remaining_pairs'],
            certificates=state['certificates'], electrode_gap_mm=gaps, positioner_deg=deg, roll_deg=roll,
            independent_layout_check=layout_state))
        print('VERIFIED', weld['name'], 'PASS', passed, 'TCP', error, 'rotation', angle, flush=True)
    output = dict(report_sha256=__import__('hashlib').sha256(report_bytes).hexdigest(), scene_sha256=sha(scene_path),
        selected_count=len(verified), verified_count=sum(r['passed'] for r in verified),
        all_selected_passed=bool(verified) and all(r['passed'] for r in verified), rows=verified,
        scope='Independent raw-source FK, fresh original fixture BVHs and fixture/product-versus-fixed-layout checks; retained layout uses the duplicate-face audit. Assembly-interface pairs are logged but not certified; rotation motion qualification unchanged.')
    (OUT/'selected_independent_verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print('VERIFICATION SUMMARY', output['verified_count'], '/', len(verified), flush=True)
    assert output['all_selected_passed']


if __name__ == '__main__': main()
