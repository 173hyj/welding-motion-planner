"""Relative geometry -> stationary reachability -> remaining external-axis search."""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT/'experimental/welding_endpoints_v1'),
               str(ROOT), str(ROOT/'build/eight_hour_v17/experimental_release/multi_point_map')]
from world import PositionerWorld, OUT, sha
from positioner import transform_weld
from schedule import search_angles, remaining_rows
from refinement import roll_target, solve_seed_bank
from acceptance import posture_passed
from gun_kinematics import gun_mesh_parts
from r2_safe_park import additional_self_checks
from source_face_world_adapter_v1 import _check_simultaneous_source_faces


class Search:
    def __init__(self, scene):
        self.scene = scene
        self.environment = PositionerWorld(scene)
        self.world = self.environment.world
        self.rids = list(self.world.models)
        self.parts = {}; self.products = {}; self.parks = {}; self.park_cache = {}
        baseline = json.loads((ROOT/'build/welding_endpoints_v1_corrected_static.json').read_text())
        for rid, model in self.world.models.items():
            gun = next(a for a in scene['assets'] if a['id'] == model.gun_id)
            _, self.parts[rid] = gun_mesh_parts(json.loads((ROOT/gun['gun']['source_detail']).read_text()))
            parks = [np.asarray(baseline['static_parks'][rid]), model.q_initial, np.zeros(6)]
            parks += [np.array([j1, j2, j3, 0., j5, np.pi/2]) for j1, j2, j3, j5 in
                      itertools.product((3., -3., 1.5, -1.5, 0.), (-.8, .2, .8), (-.2, .5), (0., -.6, .6))]
            self.parks[rid] = [q for q in parks if model.limits_valid(q)]
        for pid in self.environment.positioner.attached_product_ids:
            asset = next(a for a in scene['assets'] if a['id'] == pid)
            mesh = trimesh.load(ROOT/'build'/asset['url'], force='mesh', process=False)
            mesh.apply_transform(np.asarray(asset['transform']).reshape(4, 4))
            self.products[pid] = mesh

    def gun_pairs(self, rid, target, weld, environment):
        model = self.world.models[rid]
        T = target@np.linalg.inv(model.tcp_home)@model.gun_home
        pairs = []
        for pid, (link, kind, mesh) in self.parts[rid].items():
            obj = self.world.robots[rid]['gun:'+pid]
            # Closed gun source parts have identity kinematic deltas.
            obj.setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))
            for oid, name, other in environment:
                if kind == 'electrode' and oid == weld['product_id']:
                    continue  # Intended contact is certified only in full_check.
                result = fcl.CollisionResult()
                if fcl.collide(obj, other, fcl.CollisionRequest(), result):
                    pairs.append(dict(part=pid, asset_id=oid, asset_name=name))
        return pairs

    def relative_screen(self, weld):
        self.environment.set_angle(0)
        _, base = transform_weld(self.scene, weld, np.eye(4))
        allowed = []; rejected = []
        for roll in range(0, 360, 5):
            for sign in (1, -1):
                target = roll_target(base, roll, sign)
                for rid in self.rids:
                    candidate = dict(roll_deg=roll, axis_sign=sign, robot_id=rid)
                    pairs = self.gun_pairs(rid, target, weld, self.environment.relative_env)
                    if pairs:
                        rejected.append(dict(**candidate, pairs=pairs))
                    else:
                        allowed.append(candidate)
        return dict(weld_id=weld['id'], name=weld['name'], passed=False, selected=None,
                    stationary_passed=False, stationary_done=False, positioner_done=False,
                    relative_gun_candidates=len(allowed), allowed=allowed, relative_rejections=rejected,
                    angle_trials=[], full_checks=[], status='relative_screen_complete')

    def clear_parks(self, rid, deg):
        if (rid, deg) in self.park_cache:
            return self.park_cache[rid, deg]
        clear = []
        for q in self.parks[rid]:
            pairs = self.world.check(rid, q, max_pairs=sys.maxsize)+additional_self_checks(self.world, rid)
            if not any(p['asset_name'] != 'other_robot' for p in pairs):
                clear.append(q)
                if len(clear) >= 6:
                    break
        self.park_cache[rid, deg] = clear
        return clear

    def full_check(self, row, weld, rid, q, deg, roll, sign, target, delta):
        model = self.world.models[rid]
        other = next(r for r in self.rids if r != rid)
        moved, _ = transform_weld(self.scene, weld, delta)
        mesh = self.products[weld['product_id']].copy(); mesh.apply_transform(delta)
        for park in self.clear_parks(other, deg):
            state = _check_simultaneous_source_faces(self.world, robot_id=rid, other_robot_id=other,
                q=q, other_q=park, driver_mm=0., weld=moved, phase='closed',
                source_parts=self.parts[rid], product_mesh=mesh)
            keys = ['gun:'+p for p, info in model.gun_parts.items() if info['kind'] == 'electrode']
            product = self.environment.objects[weld['product_id']]
            gaps = {key: float(fcl.distance(self.world.robots[rid][key], product, fcl.DistanceRequest(),
                    fcl.DistanceResult())) for key in keys if key not in state['certificates']}
            actual = model.tcp(q)
            error = float(np.linalg.norm(actual[:3, 3]-moved['world_xyz']))
            normal_error = float(np.degrees(np.arccos(np.clip(actual[:3, 2]@target[:3, 2], -1, 1))))
            rotation_error = float(np.degrees(np.linalg.norm(Rotation.from_matrix(target[:3, :3].T@actual[:3, :3]).as_rotvec())))
            passed = posture_passed(state, error, keys, gaps) and normal_error < .01 and rotation_error < .01
            record = dict(robot_id=rid, q_rad=q.tolist(), other_robot_id=other, other_q_rad=park.tolist(),
                positioner_deg=deg, roll_deg=roll, axis_sign=sign, transformed_weld=moved,
                target_tcp_matrix=target.ravel().tolist(), actual_tcp_matrix=actual.ravel().tolist(),
                positioner_delta=delta.ravel().tolist(), tcp_error_mm=error, normal_error_deg=normal_error,
                rotation_error_deg=rotation_error, remaining_pairs=state['remaining_pairs'],
                certificates=state['certificates'], electrode_gap_mm=gaps, raw_contacts=state['raw_contacts'],
                both_source_joint_limits_valid=state['both_source_joint_limits_valid'], passed=passed)
            row['full_checks'].append(record)
            if passed:
                row.update(selected=record, passed=True, stationary_passed=deg == 0)
                return True
            if any(p['asset_name'] != 'other_robot' for p in state['remaining_pairs']):
                break
        return False

    def search_angle(self, row, weld, maprow, deg, previous, save):
        delta = self.environment.set_angle(deg)
        moved, base = transform_weld(self.scene, weld, delta)
        layout = self.environment.layout_collisions(deg)
        trial = dict(positioner_deg=deg, **layout, gun_clear_candidates=0, ik_states=0, ik_seed_starts=0,
                     fixed_gun_rejections=[], completed=False)
        row['angle_trials'].append(trial)
        if layout['blocking_pairs'] or not row['allowed']:
            trial['completed'] = True
            return
        if deg == 0 and previous.get('selected'):
            old = previous['selected']; rid = old['robot_id']; q = np.asarray(old['q_rad'])
            actual = self.world.models[rid].tcp(q)
            sign = 1 if actual[:3, 2]@base[:3, 2] > 0 else -1
            start = roll_target(base, 0, sign)
            rotation = start[:3, :3].T@actual[:3, :3]
            roll = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 360)
            grid_roll = round(roll/5)*5 % 360
            candidate = dict(robot_id=rid, roll_deg=grid_roll, axis_sign=sign)
            if abs(roll/5-round(roll/5)) < .001 and candidate in row['allowed']:
                target = roll_target(base, grid_roll, sign)
                if not self.gun_pairs(rid, target, moved, self.environment.fixed_env):
                    if self.full_check(row, weld, rid, q, deg, grid_roll, sign, target, delta):
                        trial.update(completed=True, reused_seed_reverified=True)
                        return
        for candidate in row['allowed']:
            rid, roll, sign = (candidate[k] for k in ('robot_id', 'roll_deg', 'axis_sign'))
            target = roll_target(base, roll, sign)
            pairs = self.gun_pairs(rid, target, moved, self.environment.fixed_env)
            if pairs:
                trial['fixed_gun_rejections'].append(dict(**candidate, pairs=pairs))
                continue
            trial['gun_clear_candidates'] += 1
            model = self.world.models[rid]
            sources = [np.asarray(c['q_rad']) for c in maprow['candidates']
                       if c['robot_id'] == rid and c['fk_candidate_passed']]
            near = [np.asarray(r['q_rad']) for r in row['full_checks'][-12:] if r['robot_id'] == rid]
            solutions, stats = solve_seed_bank(model, target, sources+near, random_count=16)
            trial['ik_seed_starts'] += stats['starts']; trial['ik_states'] += len(solutions)
            for solution in solutions:
                if self.full_check(row, weld, rid, np.asarray(solution['q_rad']), deg, roll, sign, target, delta):
                    break
            print('CANDIDATE', weld['name'], deg, roll, sign, model.asset['name'],
                  'IK', len(solutions), 'PASS', row['passed'], flush=True)
            save()
            if row['passed']:
                break
        trial['completed'] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=['all', 'stationary'], default='all')
    args = parser.parse_args()
    path = OUT/'report.json'
    if path.exists() and not args.resume:
        raise FileExistsError('Preserve historical results; use --resume')
    source = ROOT/'build/scene_placement_corrected_v1.json'
    scene = json.loads(source.read_text()); search = Search(scene)
    inventory = json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
    previous = json.loads((ROOT/'build/welding_endpoints_fixed_normal_v2/report.json').read_text())
    maps = {r['weld_id']: r for r in inventory['rows']}; old = {r['weld_id']: r for r in previous['rows']}
    code_files = [Path(__file__), Path(__file__).with_name('schedule.py'), Path(__file__).with_name('world.py'),
                  Path(__file__).with_name('geometry.py'), ROOT/'experimental/welding_endpoints_v1/refinement.py']
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_files}
    report = json.loads(path.read_text()) if path.exists() else dict(
        schema='source_positioner_static_search_v2', scene_sha256=sha(source), code_sha256=code_hashes,
        roll_step_deg=5, angle_step_deg=5, positioner_angles_deg=[0]+search_angles('positioner'),
        positioner=search.environment.positioner.metadata, rows=[], phase='relative_geometry',
        geometry_audit_sha256=sha(OUT/'geometry_audit.json'),
        scope='Static closed-gun postures only. Hanging/rotary assembly interfaces explicitly grouped; locking release, positioner rotation path, approaches and transfers are not certified.',
        contact_rule='Original electrode/designated workpiece contact faces, 25 mm source face patch, <=2 mm intended discretized contact depth. All other robot/gun collisions rejected.',
        ik_search='Analytic source-axis Jacobian; source/inventory/nearby seeds plus 16 deterministic legal Sobol starts per geometrically clear pose; finite search does not prove infeasibility.',
        summary=dict(elapsed_s=0.))
    assert report['scene_sha256'] == sha(source)
    assert report['code_sha256'] == code_hashes
    started = time.time()-report['summary']['elapsed_s']
    def save():
        rows = report['rows']
        stationary_done = sum(r['stationary_done'] for r in rows)
        positioner_done = sum(r['positioner_done'] or r['passed'] for r in rows)
        report['summary'] = dict(source_welds=29, geometry_screened=len(rows), stationary_completed=stationary_done,
            positioner_resolved=positioner_done, stationary_passed=sum(r['stationary_passed'] for r in rows),
            additional_positioner_passed=sum(r['passed'] and not r['stationary_passed'] for r in rows),
            total_passed=sum(r['passed'] for r in rows), completed=report['phase'] == 'complete', elapsed_s=time.time()-started)
        temporary = path.with_suffix('.inprogress.json')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)); temporary.replace(path)
    for weld in scene['welds'][len(report['rows']):]:
        row = search.relative_screen(weld); report['rows'].append(row); save()
        print('RELATIVE', weld['name'], 'clear', row['relative_gun_candidates'], '/288', flush=True)
    report['phase'] = 'stationary'; save()
    for row, weld in zip(report['rows'], scene['welds']):
        assert row['weld_id'] == weld['id']
        if row['stationary_done']:
            continue
        search.search_angle(row, weld, maps[weld['id']], 0, old[weld['id']], save)
        row['stationary_done'] = True
        row['status'] = 'stationary_verified' if row['passed'] else 'stationary_not_found'
        if not row['allowed']:
            row.update(positioner_done=True, status='all_5deg_rolls_blocked_by_co_rotating_geometry')
        save(); print('STATIONARY', weld['name'], row['passed'], flush=True)
    if args.stage == 'stationary':
        print('STATIONARY SUMMARY', report['summary'], flush=True); return
    report['phase'] = 'positioner'; save()
    for index in remaining_rows(report['rows']):
        row = report['rows'][index]; weld = scene['welds'][index]
        done = {t['positioner_deg'] for t in row['angle_trials'] if t['completed']}
        for deg in search_angles('positioner'):
            if row['passed']:
                break
            if deg in done:
                continue
            search.search_angle(row, weld, maps[weld['id']], deg, old[weld['id']], save)
            save(); print('ANGLE', weld['name'], deg, 'PASS', row['passed'], flush=True)
        row['positioner_done'] = True
        row['status'] = 'positioner_verified' if row['passed'] else 'not_found_in_5deg_finite_search'
        save()
    report['phase'] = 'complete'; save()
    assert sha(source) == report['scene_sha256']
    print('SUMMARY', report['summary'], flush=True)


if __name__ == '__main__': main()
