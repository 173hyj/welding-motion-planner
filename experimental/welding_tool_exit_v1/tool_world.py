"""Complete articulated source gun; robot-independent task-space queries."""
import json
import sys
from pathlib import Path
import fcl
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT/'experimental/welding_positioner_v2'),
               str(ROOT/'experimental/welding_endpoints_v1'), str(ROOT)]
from search import Search
from refinement import roll_target
from positioner import transform_weld
from source_face_radius_v1 import certify_source_faces
sys.path.insert(0, str(HERE))
from contracts import ContactTracker, certify_translation, direction_bank


def product_planes(product_mesh, contacts, origin):
    planes = {}
    for contact in contacts:
        i = contact['product_face']
        if not 0 <= i < len(product_mesh.faces):
            continue
        tri = product_mesh.vertices[product_mesh.faces[i]]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]); length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        if normal[np.argmax(abs(normal))] < 0: normal *= -1
        offset = float((tri[0]-origin)@normal)
        key = tuple(np.round(np.r_[normal, offset], 5))
        planes[key] = dict(normal=normal.tolist(), offset_mm=offset)
    return list(planes.values())


def patch_certificate(*, vertices, faces, transform, origin, contacts,
                      product_mesh, truncated, allowed_planes):
    """Explicit *release* contact rule, distinct from closed TCP certification."""
    reasons = []; maximum_radius = 0.; maximum_depth = 0.
    origin = np.asarray(origin); transform = np.asarray(transform)
    if truncated or len(contacts) >= 10000: reasons.append('truncated_contacts')
    if not contacts: reasons.append('missing_contact_evidence')
    if not allowed_planes: reasons.append('missing_initial_product_planes')
    for contact in contacts:
        mf, pf = contact['moving_face'], contact['product_face']
        depth = float(contact['penetration_depth_mm'])
        if not 0 <= mf < len(faces) or not 0 <= pf < len(product_mesh.faces):
            reasons.append('invalid_source_face'); continue
        if not np.isfinite(depth) or not 0 <= depth <= 2.: reasons.append('contact_depth_gt_2mm')
        maximum_depth = max(maximum_depth, depth)
        triangle = vertices[faces[mf]]@transform[:3,:3].T+transform[:3,3]
        radius = float(np.max(np.linalg.norm(triangle-origin, axis=1)))
        maximum_radius = max(maximum_radius, radius)
        if radius+1e-6 > 25.: reasons.append('source_triangle_outside_25mm_patch')
        product_triangle = product_mesh.vertices[product_mesh.faces[pf]]
        n = np.cross(product_triangle[1]-product_triangle[0], product_triangle[2]-product_triangle[0])
        norm = np.linalg.norm(n)
        on_plane = False
        if norm > 1e-12:
            n /= norm
            for plane in allowed_planes:
                pn = np.asarray(plane['normal'])
                if (abs(n@pn) > 1.-1e-6 and
                    np.max(abs((product_triangle-origin)@pn-plane['offset_mm'])) < .002):
                    on_plane = True; break
        if not on_plane: reasons.append('new_off_initial_plane_contact')
    return dict(passed=not reasons, failure_reasons=sorted(set(reasons)),
                maximum_source_face_radius_mm=maximum_radius,
                maximum_contact_depth_mm=maximum_depth, contact_count=len(contacts),
                rule='initial electrode/local source patch/initial product planes, release only')


def closed_contact_passed(state, electrode_ids):
    if state['blocking_pairs']: return False
    return all(state['certificates'].get(pid, {}).get('passed', False) or
               0 <= state['electrode_gaps_mm'].get(pid, float('inf')) < .1
               for pid in electrode_ids)


class ToolWorld:
    def __init__(self, scene):
        self.search = Search(scene)
        self.environment = self.search.environment
        self.world = self.search.world
        self.scene = scene
        self.parts = self.search.parts
        self.products = self.search.products
        self.delta = np.eye(4)
        self.driver_cache = {}; self.product_cache = {}
        self.rays = direction_bank()
        self.transforms_inv = {rid: np.linalg.inv(model.tcp_home)@model.gun_home
                               for rid, model in self.world.models.items()}

    def set_angle(self, angle):
        self.delta = self.environment.set_angle(angle)
        self.product_cache = {}
        return self.delta

    def target(self, weld, candidate):
        moved, base = transform_weld(self.scene, weld, self.delta)
        return moved, roll_target(base, candidate['roll_deg'], candidate['axis_sign'])

    def product(self, weld):
        pid = weld['product_id']
        if pid not in self.product_cache:
            mesh = self.products[pid].copy(); mesh.apply_transform(self.delta)
            self.product_cache[pid] = mesh
        return self.product_cache[pid]

    def poses(self, rid, target, driver):
        key = rid, float(driver)
        if key not in self.driver_cache:
            self.driver_cache[key] = self.world.models[rid].gun_model.transforms(driver)
        base = target@self.transforms_inv[rid]
        return {pid: base@self.driver_cache[key][link]
                for pid, (link, kind, mesh) in self.parts[rid].items()}

    def state(self, rid, target, driver, weld, environment, phase='opening', release=None):
        poses = self.poses(rid, target, driver)
        pairs = []; certificates = {}; gaps = {}; touching = set(); planes = {}
        product = self.product(weld)
        for pid, (link, kind, mesh) in self.parts[rid].items():
            T = poses[pid]; obj = self.world.robots[rid]['gun:'+pid]
            obj.setTransform(fcl.Transform(T[:3,:3], T[:3,3]))
            for oid, name, other in environment:
                intended = kind == 'electrode' and oid == weld['product_id'] and phase != 'separated'
                result = fcl.CollisionResult()
                request = fcl.CollisionRequest(num_max_contacts=10000 if intended else 1,
                                               enable_contact=intended)
                fcl.collide(obj, other, request, result)
                if not result.is_collision:
                    if kind == 'electrode' and oid == weld['product_id']:
                        gaps[pid] = float(fcl.distance(obj, other, fcl.DistanceRequest(), fcl.DistanceResult()))
                    continue
                if not intended:
                    return dict(blocking_pairs=[dict(part=pid, asset_id=oid, asset_name=name)],
                                certificates=certificates, electrode_gaps_mm=gaps,
                                touching=list(touching), planes=planes)
                contacts = [dict(moving_face=int(c.b1), product_face=int(c.b2),
                                 penetration_depth_mm=float(c.penetration_depth),
                                 contact_xyz_mm=np.asarray(c.pos).tolist()) for c in result.contacts]
                if phase == 'release':
                    cert = patch_certificate(vertices=mesh.vertices, faces=mesh.faces, transform=T,
                        origin=weld['world_xyz'], contacts=contacts, product_mesh=product,
                        truncated=len(contacts)>=10000, allowed_planes=release['planes'].get(pid, []))
                else:
                    cert = certify_source_faces(vertices=mesh.vertices, faces=mesh.faces, transform=T,
                        origin=weld['world_xyz'], tcp_xyz=target[:3,3], contacts=contacts,
                        product_face_count=len(product.faces), raw_collision=True,
                        truncated=len(contacts)>=10000, electrode_is_source=True, designated_product=True)
                    cert = {k:v for k,v in cert.items() if k != 'source_face_bounds'}
                    cert.update(contact_count=len(contacts),
                                maximum_contact_depth_mm=max((c['penetration_depth_mm'] for c in contacts),default=0.))
                certificates[pid] = cert
                planes[pid] = product_planes(product, contacts, np.asarray(weld['world_xyz']))
                touching.add(pid)
                if not cert['passed']:
                    return dict(blocking_pairs=[dict(part=pid, asset_id=oid, asset_name=name,
                                failure_reasons=cert['failure_reasons'])], certificates=certificates,
                                electrode_gaps_mm=gaps, touching=list(touching), planes=planes)
        return dict(blocking_pairs=pairs, certificates=certificates,
                    electrode_gaps_mm=gaps, touching=list(touching), planes=planes)

    def minimum_distance(self, rid, target, driver, environment):
        poses = self.poses(rid, target, driver)
        minimum = float('inf')
        for pid, T in poses.items():
            obj = self.world.robots[rid]['gun:'+pid]
            obj.setTransform(fcl.Transform(T[:3,:3], T[:3,3]))
            for oid, name, other in environment:
                distance = float(fcl.distance(obj, other, fcl.DistanceRequest(), fcl.DistanceResult()))
                minimum = min(minimum, distance)
                if minimum <= 0: return minimum
        return minimum

    def opening(self, rid, target, weld, step_mm=1., environment=None):
        environment = self.environment.relative_env if environment is None else environment
        electrode_ids = [pid for pid, (_,kind,_) in self.parts[rid].items() if kind == 'electrode']
        start = self.state(rid, target, 0., weld, environment, 'closed')
        if not closed_contact_passed(start, electrode_ids):
            return dict(passed=False, reason='closed_contact_not_qualified', driver_mm=0., failure=start)
        count = int(np.ceil(185./step_mm)); last = start
        for driver in np.linspace(0., -185., count+1)[1:]:
            last = self.state(rid, target, float(driver), weld, environment, 'opening')
            if last['blocking_pairs']:
                return dict(passed=False, reason='opening_blocked', driver_mm=float(driver),
                            checked_samples=count+1, failure=last)
        return dict(passed=True, checked_samples=count+1, driver_step_mm=185./count,
                    initial_state=start, open_state=last,
                    scope='exact source meshes at sampled source-linkage opening states')

    def release_context(self, rid, target, weld, opened):
        state = opened['open_state']; normals = []; supports = []
        poses = self.poses(rid, target, -185.)
        origin = np.asarray(weld['world_xyz'])
        for pid in state['touching']:
            mesh = self.parts[rid][pid][2]; T = poses[pid]
            vertices = mesh.vertices@T[:3,:3].T+T[:3,3]
            for plane in state['planes'][pid]:
                n = np.asarray(plane['normal']); offset = plane['offset_mm']
                sign = 1. if float((vertices.mean(0)-origin)@n-offset) >= 0 else -1.
                outward = sign*n
                supports.append(dict(part=pid, outward_normal=outward.tolist(),
                    initial_minimum_signed_plane_distance_mm=float(np.min(sign*((vertices-origin)@n-offset)))))
                normals.append(outward)
        return dict(planes=state['planes'], initial_touching=state['touching'], supports=supports,
                    preferred_outward_normal=np.mean(normals,axis=0).tolist() if normals else (-target[:3,2]).tolist())

    def straight_exit(self, rid, target, weld, context, direction, environment=None,
                      max_length_mm=500., release_step_mm=.1, coarse_step_mm=5.):
        environment = self.environment.relative_env if environment is None else environment
        direction = np.asarray(direction, float); direction /= np.linalg.norm(direction)
        for support in context['supports']:
            if np.asarray(support['outward_normal'])@direction <= 1e-6:
                return dict(passed=False, reason='not_outward_from_initial_contact_planes')
        def pose(s):
            p = target.copy(); p[:3,3] += s*direction; return p
        # Reject obvious blocking rays before spending dense local contact
        # queries. This is only a rejection precheck: every positive ray is
        # still rechecked below with the separated interval certificate.
        length = None
        for s in np.arange(coarse_step_mm,max_length_mm+1e-6,coarse_step_mm):
            state = self.state(rid, pose(float(s)), -185., weld, environment, 'separated')
            if state['blocking_pairs']:
                return dict(passed=False, reason='retreat_blocked', failure_mm=float(s), failure=state)
            if s >= 100.:
                distance = self.minimum_distance(rid, pose(float(s)), -185., environment)
                if distance >= 5.:
                    length = float(s); break
        if length is None:
            return dict(passed=False, reason='no_5mm_clear_endpoint_within_500mm')
        tracker = ContactTracker(context['initial_touching'])
        separated = 0. if not context['initial_touching'] else None
        release_samples = []
        if separated is None:
            for s in np.linspace(0., 5., int(np.ceil(5./release_step_mm))+1):
                state = self.state(rid, pose(float(s)), -185., weld, environment, 'release', context)
                if state['blocking_pairs'] or not tracker.advance(state['touching'], float(s)):
                    return dict(passed=False, reason='release_contact_or_body_blocked', failure_mm=float(s),
                                failure=state)
                release_samples.append(dict(travel_mm=float(s), touching=state['touching'],
                    maximum_depth_mm=max((c['maximum_contact_depth_mm'] for c in state['certificates'].values()),default=0.)))
                if not state['touching'] and self.minimum_distance(rid, pose(float(s)), -185., environment) > .05:
                    separated = float(s); break
            if separated is None:
                return dict(passed=False, reason='not_separated_within_5mm')
        certificate = certify_translation(lambda s:self.minimum_distance(rid, pose(s), -185., environment),
                                           separated, length)
        if not certificate['passed']:
            return dict(passed=False, reason='separated_interval_not_certified', certificate=certificate)
        return dict(passed=True, direction_world=direction.tolist(), length_mm=length,
            separation_mm=separated, endpoint_tcp_matrix=pose(length).ravel().tolist(),
            endpoint_clearance_mm=self.minimum_distance(rid, pose(length), -185., environment),
            release_samples=release_samples, separated_translation_certificate=certificate,
            release_scope='sampled bounded local electrode contact, not continuous contact certification')

    def search_exits(self, rid, target, weld, opened, count=2):
        context = self.release_context(rid, target, weld, opened)
        normal = np.asarray(context['preferred_outward_normal'])
        rays = [target[:3,:3]@ray for ray in self.rays]
        # Normal first; progressively tilted rays reveal side extraction.
        rays.sort(key=lambda ray: -float(ray@normal))
        exits = []; failures = {}; tested = 0
        for ray in rays:
            if any(np.dot(ray, e['direction_world']) > np.cos(np.deg2rad(20)) for e in exits):
                continue
            tested += 1
            result = self.straight_exit(rid, target, weld, context, ray)
            if result['passed']:
                exits.append(result)
                if len(exits) >= count: break
            else:
                reason = result['reason']; failures[reason] = failures.get(reason, 0)+1
        return dict(exits=exits, rays_tested=tested, failure_counts=failures,
                    release_context=context, direction_bank_size=len(self.rays))
