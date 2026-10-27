"""Sample original two-weld gun sequence; failed attempts remain diagnostic."""
import json
from pathlib import Path
import numpy as np

from collision_audit import CollisionWorld
from solve_welds import solve

ROOT = Path(__file__).resolve().parent


def main():
    scene = json.loads((ROOT / 'build/scene.json').read_text())
    candidates = json.loads((ROOT / 'build/collision_plan_source_normal_roll_optimized.json').read_text())['candidates']
    safe = [c for c in candidates if c['collision_free_environment']]
    world = CollisionWorld(scene)
    names = ['OP10L-SUB1-POINT-GW-7', 'OP10L-SUB1-POINT-GW-6']
    chosen = [next(c for c in safe if c['weld_id'] == name) for name in names]
    rid = chosen[0]['robot_id']
    model = world.models[rid]
    q0, q1 = [np.array(c['q_rad']) for c in chosen]
    welds = [next(w for w in scene['welds'] if w['id'] == c['weld_id']) for c in chosen]
    segments, frames, retreat_search = [], [], []
    current = {r: k.q_initial.copy() for r, k in world.models.items()}
    current[rid] = q0

    def frame(q, driver, phase, weld_id):
        current[rid] = q
        robot_frames, guns = {}, {}
        for robot_id, k in world.models.items():
            links = k.links(current[robot_id], world=True)
            robot_frames[robot_id] = {'q_rad': current[robot_id].tolist(), 'links': {key: m.flatten().tolist() for key, m in links.items()}, 'tcp_xyz': k.tcp(current[robot_id])[:3, 3].tolist()}
            gun = next(g for g in scene['assets'] if g['id'] == k.gun_id)
            value = driver if robot_id == rid else 0.
            pose = links['245'] @ k.gun_home
            deltas = k.gun_model.transforms(value)
            guns[k.gun_id] = {'transform': pose.flatten().tolist(), 'driver_mm': float(value), 'parts': {pid: (pose @ deltas[p['link_key']]).flatten().tolist() for pid, p in gun['gun']['link_models'].items()}}
        return {'time_s': len(frames) * .1, 'phase': phase, 'weld_id': weld_id, 'robots': robot_frames, 'guns': guns}

    def audit_segment(phase, qa, qb, da, db, weld):
        samples, failures = [], []
        for i, t in enumerate(np.linspace(0, 1, 21)):
            q = qa + t * (qb - qa)
            driver = da + t * (db - da)
            limit_valid = model.limits_valid(q)
            # Process contact only at a registered endpoint. At open/close the
            # stationary electrode remains at the designated point. Retreat
            # and approach receive the allowance only while TCP is within .1mm.
            pairs = world.check(rid, q, weld=weld, gun_driver_mm=driver, max_pairs=8)
            if pairs or not limit_valid:
                failures.append({'sample': i, 'fraction': float(t), 'joint_limits_valid': bool(limit_valid), 'pairs': pairs})
            samples.append({'q_rad': q.tolist(), 'driver_mm': float(driver), 'joint_limits_valid': bool(limit_valid), 'collision_verified': not pairs and bool(limit_valid), 'collision_pairs': pairs})
            frames.append(frame(q, driver, phase, weld['id']))
        result = {'phase': phase, 'weld_id': weld['id'], 'sample_count': 21, 'valid': not failures, 'failure_count': len(failures), 'failures': failures, 'samples': samples,
            'status': ('endpoint_open' if phase == 'open' else 'endpoint_close' if phase == 'close' else 'sampled_transit_valid') if not failures else 'transit_rejected',
            'gun_driver_start_mm': float(da), 'gun_driver_end_mm': float(db)}
        print(phase, weld['id'], 'valid', result['valid'], 'failed samples', len(failures), flush=True)
        return result

    # First validate opening, preserving the verified closed endpoint and all
    # original part geometry. No path from the unsafe source pose is asserted.
    segments.append(audit_segment('open', q0, q0, 0., -185., welds[0]))
    retreats = []
    for q, weld in zip((q0, q1), welds):
        pose = model.tcp(q)
        alternatives = []
        for sign in (1., -1.):
            target = pose.copy()
            target[:3, 3] += sign * 100. * pose[:3, 2]
            solutions = solve(model, target, q, 4)
            for solution in solutions[:2]:
                qr = np.array(solution['q_rad'])
                # Select one constant chart for the entire edge. Every sample
                # must remain in the original dependent joint-limit polygon.
                for lift in (0., -2 * np.pi, 2 * np.pi):
                    candidate = qr.copy(); candidate[5] += lift
                    path = [q + t * (candidate - q) for t in np.linspace(0, 1, 21)]
                    if all(model.limits_valid(s) for s in path):
                        alternatives.append((np.linalg.norm(candidate - q), candidate, sign))
        if not alternatives:
            retreats.append(None)
        else:
            attempts = []
            for distance, qr, sign in sorted(alternatives, key=lambda s: s[0]):
                if distance > 1.:
                    continue
                segment = audit_segment('retreat' if weld is welds[0] else 'approach_candidate', q, qr, -185., -185., weld)
                segment['normal_sign'] = sign
                attempts.append(segment)
                if segment['valid']:
                    break
            retreat_search.append({'weld_id': weld['id'], 'attempts': attempts})
            if not attempts:
                retreats.append(None)
            else:
                best = min(attempts, key=lambda s: s['failure_count'])
                retreats.append(np.array(best['samples'][-1]['q_rad']))
                segments.append(best)
    if all(q is not None for q in retreats):
        # Move frames for the second retreat candidate into the actual reverse
        # approach only below; both attempted edges retain collision evidence.
        segments.append(audit_segment('transfer', retreats[0], retreats[1], -185., -185., welds[1]))
        segments.append(audit_segment('approach', retreats[1], q1, -185., -185., welds[1]))
    segments.append(audit_segment('close', q1, q1, -185., 0., welds[1]))
    required = [s for s in segments if s['phase'] != 'approach_candidate']
    sequence_valid = len(required) == 5 and all(s['valid'] for s in required)
    # Rebuild chronological playback from the attempted sequence only.
    frames = []
    for segment in required:
        for sample in segment['samples']:
            entry = frame(np.array(sample['q_rad']), sample['driver_mm'], segment['phase'], segment['weld_id'])
            entry.update(collision_verified=sample['collision_verified'], collision_pairs=sample['collision_pairs'])
            frames.append(entry)
    result = {'status': {'diagnostic_only': not sequence_valid, 'tcp_verified': True, 'collision_verified': False,
        'local_sequence_collision_verified': sequence_valid,
        'endpoint_collision_verified': True, 'transit_collision_verified': sequence_valid, 'source_initial_unsafe': True,
        'system_safe': False, 'alternative_simulation_start': True, 'gun_dof_verified': True,
        'reachable_welds': 2 if sequence_valid else 0, 'planned_welds': 2 if sequence_valid else 0, 'partial_sequence': True,
        'collision_scope': 'R1 articulated gun and robot versus original environment, other robot geometry, and nonadjacent robot self-collision; 21 samples per phase',
        'note': ('2/29 原始焊点的局部路线：开枪、退刀、转移、进刀、合枪，共105个采样状态通过；从独立端点启动。原始起点不安全，R2未安全停放，整体系统未认证。' if sequence_valid else '原始焊枪开合与两个焊点之间的连接尝试未全部通过，失败采样已保留。'),
        'reason': 'Original TxKin gun sequence between two alternate projected-normal weld poses, sampled at 21 states per phase. Local R1 sequence result is separate from whole-system safety: no source-pose transit or whole-system safety is asserted; R2 remains at the unsafe supplied source pose. Internal assembled gun interfaces are not separately collision-certified.'},
        'units': 'mm/rad', 'frames': frames, 'welds': scene['welds'], 'selected_weld_ids': names, 'segments': segments, 'retreat_search': retreat_search,
        'sequence_valid': sequence_valid, 'transit_valid': sequence_valid, 'accepted_subsegments': [s['status'] for s in required if s['valid']],
        'endpoint_actions_only': not sequence_valid,
        'start_note': 'Start at strict separately validated endpoint; no path is asserted from unsafe source initial state.',
        'source_kinematics': [g['gun']['source_detail'] for g in scene['assets'] if g['kind'] == 'gun'],
        'collision_plan': 'collision_plan_source_normal_roll_optimized.json'}
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    (ROOT / 'build/gun_sequence_plan.json').write_text(output_text)
    if sequence_valid:
        (ROOT / 'build/welding_plan.json').write_text(output_text)
    print('two-weld articulated sequence valid:', sequence_valid, flush=True)


if __name__ == '__main__':
    main()
