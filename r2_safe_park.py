"""Search <=100 alternative R2 endpoint states, then replay frozen R1 locally."""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import fcl
import numpy as np

from collision_audit import CollisionWorld

ROOT = Path(__file__).resolve().parent


def additional_self_checks(world, rid):
    """Include base/counterbalance versus nonattached original robot parts."""
    objects = world.robots[rid]
    keys = [key for key in objects if key.isdigit()]
    interface_pairs = {tuple(sorted(p)) for p in [('239', '240'), ('240', '241'), ('241', '242'), ('242', '243'), ('243', '244'), ('244', '245'),
        ('240', '246'), ('241', '246'), ('240', '247'), ('241', '247'), ('246', '247')]}
    result = []
    for a, b in itertools.combinations(keys, 2):
        if not ({a, b} & {'239', '246', '247'}) or tuple(sorted((a, b))) in interface_pairs:
            continue
        contact = fcl.CollisionResult()
        fcl.collide(objects[a], objects[b], fcl.CollisionRequest(), contact)
        if contact.is_collision:
            result.append({'link': a, 'asset_id': b, 'asset_name': 'additional_robot_self'})
    gun_keys = [key for key in objects if key.startswith('gun:')]
    for a in ('239', '246', '247'):
        for b in gun_keys:
            contact = fcl.CollisionResult()
            fcl.collide(objects[a], objects[b], fcl.CollisionRequest(), contact)
            if contact.is_collision:
                result.append({'link': a, 'asset_id': b, 'asset_name': 'additional_robot_self'})
    return result


def main():
    scene_path, plan_path = ROOT / 'build/scene.json', ROOT / 'build/welding_plan.json'
    scene_bytes, plan_bytes = scene_path.read_bytes(), plan_path.read_bytes()
    scene, source = json.loads(scene_bytes), json.loads(plan_bytes)
    world = CollisionWorld(scene)
    rid1 = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R1' in a['name'])
    rid2 = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R2' in a['name'])
    k1, k2 = world.models[rid1], world.models[rid2]
    first_q = np.array(source['frames'][0]['robots'][rid1]['q_rad'])
    world.update(rid1, first_q)
    # Named source home/current poses and the independently validated R1
    # endpoint chart are preferred before deterministic alternative endpoints.
    candidates = [('source_home', np.zeros(6)), ('source_current', k2.q_initial.copy()), ('R1_endpoint_chart', first_q.copy())]
    for j1, j2, j3, j5 in itertools.product((0., -1.5, 1.5, 3.), (-.8, .2, .8), (-.2, .5), (0., -.6, .6)):
        candidates.append(('bounded_alternative', np.array([j1, j2, j3, 0., j5, 1.5707963268])))
    audit, selected = [], None
    for label, q in candidates[:100]:
        limits = k2.limits_valid(q)
        pairs = world.check(rid2, q) + additional_self_checks(world, rid2) if limits else []
        row = {'candidate': len(audit), 'source': label, 'q_rad': q.tolist(), 'limits_valid': bool(limits), 'collision_pairs': pairs, 'valid': bool(limits and not pairs)}
        audit.append(row)
        print('park',len(audit),label,'valid',row['valid'],'pairs',len(pairs),flush=True)
        if row['valid']:
            selected = row
            break
    report = {'alternative_park_found': selected is not None, 'source_initial_unsafe': True, 'source_to_park_path_verified': False,
        'system_safe': False, 'robot_id': rid2, 'selected': selected, 'candidate_limit': 100, 'candidate_count': len(audit), 'candidates': audit,
        'scene_sha256': hashlib.sha256(scene_bytes).hexdigest(), 'frozen_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
        'scope': 'All original robot and articulated gun parts versus environment/other robot; nonattached robot pairs including base/counterbalance; assembled gun internal interfaces not certified.'}
    out = ROOT / 'build/r2_safe_park_search.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if selected is None:
        return
    park = np.array(selected['q_rad'])
    world.update(rid2, park)
    result = copy.deepcopy(source)
    qrecords = result['frames']
    recheck = []
    # Check both robots against the exact simultaneous state at every original
    # source frame. R2 has no workpiece process-contact allowance while parked.
    for frame in qrecords:
        q1 = np.array(frame['robots'][rid1]['q_rad'])
        driver = frame['guns'][k1.gun_id]['driver_mm']
        weld = next(w for w in scene['welds'] if w['id'] == frame['weld_id'])
        pairs1 = world.check(rid1, q1, weld=weld, gun_driver_mm=driver) + additional_self_checks(world, rid1)
        pairs2 = world.check(rid2, park) + additional_self_checks(world, rid2)
        valid = k1.limits_valid(q1) and k2.limits_valid(park) and not pairs1 and not pairs2
        recheck.append({'time_s':frame['time_s'], 'phase':frame['phase'], 'both_robots_valid':bool(valid), 'R1_pairs':pairs1, 'R2_pairs':pairs2})
        links2 = k2.links(park, world=True)
        frame['robots'][rid2] = {'q_rad': park.tolist(), 'links': {key: matrix.flatten().tolist() for key, matrix in links2.items()}, 'tcp_xyz': k2.tcp(park)[:3,3].tolist()}
        gun = next(a for a in scene['assets'] if a['id'] == k2.gun_id)
        pose = links2['245'] @ k2.gun_home
        deltas = k2.gun_model.transforms(0.)
        frame['guns'][k2.gun_id] = {'transform':pose.flatten().tolist(), 'driver_mm':0., 'parts':{pid:(pose @ deltas[p['link_key']]).flatten().tolist() for pid,p in gun['gun']['link_models'].items()}}
        frame.update(collision_verified=bool(valid), collision_pairs=pairs1+pairs2)
    valid = all(row['both_robots_valid'] for row in recheck)
    result['status'].update(collision_verified=False, system_safe=False, both_robots_sampled_collision_verified=valid,
        local_sequence_collision_verified=valid, transit_collision_verified=valid, r2_alternative_park_verified=True,
        source_to_park_path_verified=False, source_initial_unsafe=True, diagnostic_only=not valid,
        note='两个机器人在仿真替代起点的采样检查通过。R2采用另行验证的静态停放姿态，R1执行两点焊接局部路线；未验证原起点到替代起点、焊枪内部装配及连续扫掠安全。' if valid else 'R2替代停放端点通过，但共同序列仍有拒绝采样。')
    result['status']['reason'] = 'R2 source初始姿态存在碰撞；本文件使用另行验证的静态替代停放姿态，双方机器人逐帧采样通过。source→park路径未验证，且不认证焊枪内部装配接触或连续扫掠安全。'
    result['alternative_park'] = report
    result['both_robots_recheck'] = recheck
    result['frozen_plan_sha256'] = report['frozen_plan_sha256']
    (ROOT / 'build/welding_plan_both_alternative.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('both alternative sampled sequence valid',valid,'frames',len(recheck),flush=True)
    assert scene_path.read_bytes() == scene_bytes and plan_path.read_bytes() == plan_bytes


if __name__ == '__main__':
    main()
