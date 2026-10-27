"""Finite exhaustive source-pose adapter; opposite-arm compatibility is deferred."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def filter_single_arm_pairs(pairs, robot_id):
    """Keep all own self/environment blockers, never opposite-arm blockers."""
    return [p for p in pairs if p.get('checked_robot_id', robot_id) == robot_id
            and p.get('asset_name') != 'other_robot']


def qualified_single_arm_pairs(filtered, robot_id):
    """Consume the source certificate adapter's structured return value."""
    return filter_single_arm_pairs(filtered['remaining_pairs'], robot_id)


def unique_raw_solutions(solutions, tolerance=1e-4):
    """Deduplicate actuator coordinates without collapsing 2π windings."""
    unique = []
    for solution in solutions:
        if not any(math.sqrt(sum((a-b)**2 for a,b in zip(solution['q_rad'], old['q_rad']))) < tolerance
                   for old in unique):
            unique.append(solution)
    return unique


def inventory_from_static_report(report):
    """Use every historical posture as a seed, independent of collision outcome."""
    inventory = {}
    for row in report['rows']:
        records = list(row.get('full_checks', []))
        if row.get('selected'):
            records.append(row['selected'])
        candidates = [dict(robot_id=r['robot_id'],q_rad=list(r['q_rad']),fk_candidate_passed=True)
                      for r in records if 'q_rad' in r and 'robot_id' in r]
        inventory[row['weld_id']] = dict(candidates=candidates)
    return inventory


class SourceCatalog:
    def __init__(self, seed_count=48):
        sys.path[:0] = [str(ROOT/'experimental/welding_positioner_v2'),
                       str(ROOT/'experimental/welding_endpoints_v1'), str(ROOT),
                       str(ROOT/'build/eight_hour_v17/experimental_release/multi_point_map')]
        import numpy as np
        from search import Search
        from refinement import roll_target, PoseProblem
        from positioner import transform_weld
        from r2_safe_park import additional_self_checks
        from source_face_radius_v1 import certify_source_faces
        from source_face_world_adapter_v1 import filter_certified_pairs
        self.np = np
        self.roll_target, self.PoseProblem = roll_target, PoseProblem
        self.transform_weld = transform_weld
        self.self_checks, self.certify, self.filter_certified = additional_self_checks, certify_source_faces, filter_certified_pairs
        self.scene = json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
        self.search = Search(self.scene)
        self.environment, self.world = self.search.environment, self.search.world
        self.welds, self.robot_ids = self.scene['welds'], self.search.rids
        self.seed_count = seed_count
        self.seed_source_path = ROOT/'build/welding_positioner_static_v2/report.json'
        self.inventory = inventory_from_static_report(json.loads(self.seed_source_path.read_text()))
        self.nearby = {rid:[] for rid in self.robot_ids}
        self.set_angle(0)

    def relative(self, weld):
        return self.search.relative_screen(weld)

    def set_angle(self, angle):
        self.angle = angle
        self.delta = self.environment.set_angle(angle)
        layout = self.environment.layout_collisions(angle)
        return dict(layout, passed=not layout['blocking_pairs'])

    def _target(self, weld, pose):
        moved, base = self.transform_weld(self.scene, weld, self.delta)
        return moved, self.roll_target(base, pose['roll_deg'], pose['axis_sign'])

    def fixed(self, weld, pose):
        moved, target = self._target(weld, pose)
        pairs = self.search.gun_pairs(pose['robot_id'], target, moved, self.environment.fixed_env)
        return dict(passed=not pairs, pairs=pairs, reason='fixed_gun_collision' if pairs else 'clear')

    def solve(self, weld, pose):
        from scipy.optimize import least_squares
        from scipy.stats import qmc
        np = self.np
        rid = pose['robot_id']; model = self.world.models[rid]
        _, target = self._target(weld, pose)
        seeds = [np.asarray(c['q_rad'],float) for c in self.inventory[weld['id']]['candidates']
                 if c['robot_id'] == rid and c['fk_candidate_passed']]
        starts = seeds + self.nearby.get(rid, [])[-12:] + [model.q_initial, np.zeros(6)]
        sobol_count = 0
        if self.seed_count:
            unit = qmc.Sobol(6,scramble=True,seed=20261008).random_base2(int(np.ceil(np.log2(max(128,self.seed_count*4)))))
            random = model.bounds[0]+unit*(model.bounds[1]-model.bounds[0])
            sobol = [q for q in random if model.limits_valid(q)][:self.seed_count]
            sobol_count = len(sobol)
            starts += sobol
        starts = list({tuple(np.round(q,7)):q for q in starts}.values())
        problem = self.PoseProblem(model,target)
        solutions = []; stats = dict(starts=len(starts),converged_pose_count=0,illegal_solutions=0,best_residual=None,
                                    deduplication='raw_actuator_l2_1e-4_rad', finite_search=True,
                                    sobol_requested=self.seed_count,sobol_legal_starts=sobol_count)
        for seed in starts:
            fit = least_squares(lambda q:problem.evaluate(q)[0], np.clip(seed,model.bounds[0]+1e-8,model.bounds[1]-1e-8),
                jac=lambda q:problem.evaluate(q)[1],bounds=(model.bounds[0],model.bounds[1]),max_nfev=220,
                ftol=1e-11,xtol=1e-11,gtol=1e-11)
            error = problem.evaluate(fit.x)[0]; residual = float(np.linalg.norm(error))
            if not math.isfinite(residual):
                continue
            stats['best_residual'] = residual if stats['best_residual'] is None else min(stats['best_residual'],residual)
            if np.linalg.norm(error[:3])*1000. >= .05 or np.linalg.norm(error[3:]) >= 1e-5: continue
            stats['converged_pose_count'] += 1
            limits_valid = bool(model.limits_valid(fit.x))
            if not limits_valid:
                stats['illegal_solutions'] += 1
            solutions.append(dict(q_rad=fit.x.tolist(),source_joint_limits_valid=limits_valid,position_error_mm=float(np.linalg.norm(error[:3])*1000.),
                                  rotation_error_rad=float(np.linalg.norm(error[3:]))))
        solutions = unique_raw_solutions(solutions)
        self.nearby.setdefault(rid, []).extend(np.asarray(s['q_rad']) for s in solutions)
        self.nearby[rid] = self.nearby.get(rid, [])[-12:]
        return solutions, stats

    def accept(self, weld, pose, solution):
        np = self.np
        rid = pose['robot_id']; model = self.world.models[rid]
        q = np.asarray(solution['q_rad'],float)
        if not model.limits_valid(q):
            return dict(passed=False,reason='source_joint_limits_invalid',robot_id=rid,
                        q_rad=q.tolist(),source_joint_limits_valid=False)
        import fcl
        from scipy.spatial.transform import Rotation
        moved, target = self._target(weld, pose)
        # Restrict collision enumeration itself, so no other-arm state can gate
        # this catalog entry. Restore even when a strict source check raises.
        robots = self.world.robots
        try:
            self.world.robots = {rid:robots[rid]}
            original = self.world.check(rid,q,max_pairs=sys.maxsize) + self.self_checks(self.world,rid)
        finally:
            self.world.robots = robots
        original = [dict(p,checked_robot_id=rid) for p in original]
        links = model.links(q,world=True)
        product_mesh = self.search.products[weld['product_id']].copy()
        product_mesh.apply_transform(self.delta)
        product = self.environment.objects[weld['product_id']]
        certificates = {}; gaps = {}; keys = []
        transforms = model.gun_model.transforms(0.)
        actual = model.tcp(q)
        for pid,(link,kind,mesh) in self.search.parts[rid].items():
            if kind != 'electrode': continue
            key = 'gun:'+pid; keys.append(key)
            transform = links['245']@model.gun_home@transforms[link]
            obj = robots[rid][key]
            obj.setTransform(fcl.Transform(transform[:3,:3],transform[:3,3]))
            result = fcl.CollisionResult()
            fcl.collide(obj,product,fcl.CollisionRequest(num_max_contacts=10000,enable_contact=True),result)
            contacts = [dict(moving_face=int(c.b1),product_face=int(c.b2),penetration_depth_mm=float(c.penetration_depth),
                             contact_xyz_mm=np.asarray(c.pos).tolist(),contact_normal=np.asarray(c.normal).tolist()) for c in result.contacts]
            if result.is_collision:
                cert = self.certify(vertices=mesh.vertices,faces=mesh.faces,transform=transform,
                    origin=moved['world_xyz'],tcp_xyz=actual[:3,3],contacts=contacts,product_face_count=len(product_mesh.faces),
                    raw_collision=bool(result.is_collision),truncated=len(contacts)>=10000,electrode_is_source=True,designated_product=True)
                certificates[key] = {k:v for k,v in cert.items() if k != 'source_face_bounds'}
            else:
                gaps[key] = float(fcl.distance(obj,product,fcl.DistanceRequest(),fcl.DistanceResult()))
        remaining = self.filter_certified(original,robot_id=rid,product_id=weld['product_id'],certificates=certificates,phase='closed')
        remaining = qualified_single_arm_pairs(remaining,rid)
        error = float(np.linalg.norm(actual[:3,3]-np.asarray(moved['world_xyz'])))
        rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec())))
        normal = float(np.degrees(np.arccos(np.clip(actual[:3,2]@target[:3,2],-1,1))))
        limits = bool(model.limits_valid(q))
        contacts_ok = len(keys)==2 and all(certificates[k]['passed'] if k in certificates else math.isfinite(gaps.get(k,math.inf)) and gaps[k] <= .1 for k in keys)
        passed = limits and not remaining and contacts_ok and math.isfinite(error) and error < .1 and rotation < .01 and normal < .01
        return dict(passed=bool(passed),robot_id=rid,q_rad=q.tolist(),source_joint_limits_valid=limits,
                    remaining_pairs=remaining,certificates=certificates,electrode_gap_mm=gaps,
                    tcp_error_mm=error,rotation_error_deg=rotation,normal_error_deg=normal,
                    target_tcp_matrix=target.ravel().tolist(),actual_tcp_matrix=actual.ravel().tolist(),
                    positioner_deg=self.angle,roll_deg=pose['roll_deg'],axis_sign=pose['axis_sign'])

