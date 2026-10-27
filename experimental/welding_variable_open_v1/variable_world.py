"""Explicit driver-dependent queries, same source contact and exit rules as v2."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'welding_tool_exit_v1'))
from tool_world import ToolWorld
from contracts import ContactTracker, certify_translation
from variable_core import driver_samples


class VariableToolWorld(ToolWorld):
    def release_context(self, rid, target, weld, opened, *, driver_mm):
        driver_samples(driver_mm)
        state = opened['open_state']; normals = []; supports = []
        poses = self.poses(rid, target, driver_mm)
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

    def straight_exit(self, rid, target, weld, context, direction, *, driver_mm, environment=None,
                      max_length_mm=500., release_step_mm=.1, coarse_step_mm=5.):
        driver_samples(driver_mm)
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
            state = self.state(rid, pose(float(s)), driver_mm, weld, environment, 'separated')
            if state['blocking_pairs']:
                return dict(passed=False, reason='retreat_blocked', failure_mm=float(s), failure=state)
            if s >= 100.:
                distance = self.minimum_distance(rid, pose(float(s)), driver_mm, environment)
                if distance >= 5.:
                    length = float(s); break
        if length is None:
            return dict(passed=False, reason='no_5mm_clear_endpoint_within_500mm')
        tracker = ContactTracker(context['initial_touching'])
        separated = 0. if not context['initial_touching'] else None
        release_samples = []
        if separated is None:
            for s in np.linspace(0., 5., int(np.ceil(5./release_step_mm))+1):
                state = self.state(rid, pose(float(s)), driver_mm, weld, environment, 'release', context)
                if state['blocking_pairs'] or not tracker.advance(state['touching'], float(s)):
                    return dict(passed=False, reason='release_contact_or_body_blocked', failure_mm=float(s),
                                failure=state)
                release_samples.append(dict(travel_mm=float(s), touching=state['touching'],
                    maximum_depth_mm=max((c['maximum_contact_depth_mm'] for c in state['certificates'].values()),default=0.)))
                if not state['touching'] and self.minimum_distance(rid, pose(float(s)), driver_mm, environment) > .05:
                    separated = float(s); break
            if separated is None:
                return dict(passed=False, reason='not_separated_within_5mm')
        certificate = certify_translation(lambda s:self.minimum_distance(rid, pose(s), driver_mm, environment),
                                           separated, length)
        if not certificate['passed']:
            return dict(passed=False, reason='separated_interval_not_certified', certificate=certificate)
        return dict(passed=True, direction_world=direction.tolist(), length_mm=length,
            separation_mm=separated, endpoint_tcp_matrix=pose(length).ravel().tolist(),
            endpoint_clearance_mm=self.minimum_distance(rid, pose(length), driver_mm, environment),
            release_samples=release_samples, separated_translation_certificate=certificate,
            release_scope='sampled bounded local electrode contact, not continuous contact certification')

    def search_exits(self, rid, target, weld, opened, *, driver_mm, count=2):
        context = self.release_context(rid, target, weld, opened, driver_mm=driver_mm)
        normal = np.asarray(context['preferred_outward_normal'])
        rays = [target[:3,:3]@ray for ray in self.rays]
        # Normal first; progressively tilted rays reveal side extraction.
        rays.sort(key=lambda ray: -float(ray@normal))
        exits = []; failures = {}; tested = 0
        for ray in rays:
            if any(np.dot(ray, e['direction_world']) > np.cos(np.deg2rad(20)) for e in exits):
                continue
            tested += 1
            result = self.straight_exit(rid, target, weld, context, ray, driver_mm=driver_mm)
            if result['passed']:
                exits.append(result)
                if len(exits) >= count: break
            else:
                reason = result['reason']; failures[reason] = failures.get(reason, 0)+1
        return dict(exits=exits, rays_tested=tested, failure_counts=failures,
                    release_context=context, direction_bank_size=len(self.rays))
