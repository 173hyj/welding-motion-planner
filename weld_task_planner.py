"""Evidence-grounded source welding tasks, alternate-pose sampled planning.

Writes versioned results only; preserves source order and all failed tasks.
Neither source-start reachability nor industrial/continuous safety is asserted.
"""
import argparse
import hashlib
import itertools
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'build/eight_hour_v17'


def segment_duration(qa, qb, speeds, accelerations):
    """Duration of a rest-to-rest quintic respecting all linear axis limits."""
    delta = np.abs(np.asarray(qb) - np.asarray(qa))
    v, a = np.asarray(speeds), np.asarray(accelerations)
    if np.any(v <= 0) or np.any(a <= 0):
        raise ValueError('Source speed and acceleration must be positive')
    return max(float(np.max(1.875 * delta / v)), float(np.sqrt(np.max((10 / math.sqrt(3)) * delta / a))), .001)


def classify_route(samples):
    return 'planned' if samples and all(s.get('collision_verified', False) for s in samples) else 'transit_rejected'


def validate_segment_chain(segments, tolerance=1e-8):
    """Require phase seams to share the exact terminal state.

    A route may repeat a seam state for evidence, but a phase transition cannot
    teleport in joint or gun-driver coordinates between independently sampled
    edges.
    """
    for previous, current in zip(segments, segments[1:]):
        if not previous.get('samples') or not current.get('samples'):
            raise ValueError('Empty segment cannot form a phase chain')
        a, b = previous['samples'][-1], current['samples'][0]
        if np.max(np.abs(np.asarray(a['q_rad']) - np.asarray(b['q_rad']))) > tolerance or abs(float(a['driver_mm']) - float(b['driver_mm'])) > tolerance:
            raise ValueError('Discontinuous phase seam')
        if float(b.get('time_s', 0.)) < 0.:
            raise ValueError('Negative phase-local time')
    return True


def plan_source_tasks(scene, output_dir, inventory=None):
    """Create the total source-task contract before any collision search."""
    inventory = inventory or {}
    ids = [w['id'] for w in scene['welds']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source weld IDs')
    if set(inventory) - set(ids):
        raise ValueError('Unknown source weld IDs in endpoint inventory')
    tasks = [{'id': w['id'], 'group': w['group'], 'sequence': w['sequence'], 'product_id': w['product_id'],
              'status': 'endpoint_unavailable', 'endpoint_candidates': inventory.get(w['id'], []), 'failure_reasons': []} for w in scene['welds']]
    return {'schema': 'source_weld_tasks_v17', 'status': {'collision_verified': False, 'system_safe': False,
        'source_initial_unsafe': True, 'source_to_park_path_verified': False, 'alternative_simulation_start': True,
        'orientation_source': 'alternate original projected normal; source isProjected=0; active source pose not inferred'},
        'source_order': ids, 'welds': tasks, 'routes': [], 'output_dir': str(output_dir)}


def constant_charts(q, reference, model):
    """Only whole-edge constant 2pi chart lifts; never wrap each sample."""
    charts = []
    for lift in (-2 * np.pi, 0., 2 * np.pi):
        candidate = np.asarray(q, dtype=float).copy()
        candidate[5] += lift
        if model.limits_valid(candidate):
            charts.append(candidate)
    return sorted(charts, key=lambda x: np.linalg.norm(x - reference))


def fractions(qa, qb, da=0., db=0., joint_step=.01, gun_step=5.):
    n = max(1, math.ceil(np.max(np.abs(np.asarray(qb) - qa)) / joint_step), math.ceil(abs(db - da) / gun_step))
    return np.linspace(0., 1., n + 1)


def contact_boundary_fractions(model, qa, qb, weld):
    """Probe the strict contact cutoff near each welded endpoint.

    Ordinary angular samples can miss the narrow region immediately outside
    the 0.1 mm TCP contract while a fixed electrode still intersects the panel.
    These probes add evidence along the same joint-space edge; finite checking
    remains explicitly distinct from continuous certification.
    """
    if weld is None or np.max(np.abs(np.asarray(qb)-qa)) < 1e-12:
        return []
    point=np.asarray(weld['world_xyz']);qa=np.asarray(qa);qb=np.asarray(qb)
    result=[]
    for start,end,reverse in [(qa,qb,False),(qb,qa,True)]:
        if np.linalg.norm(model.tcp(start)[:3,3]-point) >= .1:
            continue
        for distance in [.11,.2,.5,1.,2.,5.]:
            high=1e-7
            while high < 1. and np.linalg.norm(model.tcp(start+high*(end-start))[:3,3]-point) < distance:
                high=min(1.,high*2.)
            if np.linalg.norm(model.tcp(start+high*(end-start))[:3,3]-point) < distance:
                continue
            low=0.
            for _ in range(32):
                mid=(low+high)/2.
                if np.linalg.norm(model.tcp(start+mid*(end-start))[:3,3]-point) < distance:low=mid
                else:high=mid
            result.append(1.-high if reverse else high)
    return sorted(set(result))


def source_robot_limits(scene):
    raw = json.loads((ROOT / 'build/robot.json').read_text())
    text = raw['kinematicsSource']
    device = ET.fromstring(text[text.index('<devices>'):text.index('</devices>') + 10]).find('device')
    joints = list(device.find('joints'))[:6]
    speeds = np.array([float(j.findtext('maxSpeed')) for j in joints])
    accelerations = np.array([float(j.findtext('maxAcc')) for j in joints])
    coupling = np.eye(6)
    key_index = {j['key']: i for i, j in enumerate(scene['joints'][:6])}
    for i, joint in enumerate(scene['joints'][:6]):
        for key, factor in zip(joint['leadingJoints'], joint['leadingFactors']):
            coupling[i, key_index[key]] += factor
    return speeds, accelerations, coupling


def passive_joint_values(scene, q):
    """Source 473/474/475 closure values, positive about/along source axes."""
    from welding_kinematics import axis_matrix
    q=np.asarray(q);a=np.array(scene['joints'][6]['axis1']);b=np.array(scene['joints'][8]['axis1'])
    j=scene['joints'][1];point=np.array(j['axis1']);axis=np.array(j['axis2'])-point
    moved=axis_matrix(point,axis,q[1])@np.r_[a,1.]
    span=moved[:3]-b;home=a-b
    theta=np.arctan2(home[2],home[0])-np.arctan2(span[2],span[0])
    return np.array([theta-q[1],np.linalg.norm(span)-np.linalg.norm(home),theta])


class SourceTiming:
    """Quintic actuator timing including source nonlinear gun linkage axes."""
    def __init__(self, scene, gun):
        self.speeds, self.accelerations, self.coupling = source_robot_limits(scene)
        self.scene=scene
        self.gun = gun
        self.gun_speeds = np.array([float(j.findtext('maxSpeed')) for j in gun.joints])
        self.gun_accelerations = np.array([float(j.findtext('maxAcc')) for j in gun.joints])
        raw=json.loads((ROOT/'build/robot.json').read_text())['kinematicsSource']
        passive=list(ET.fromstring(raw[raw.index('<devices>'):raw.index('</devices>')+10]).find('device/joints'))[6:9]
        self.passive_speeds=np.array([float(j.findtext('maxSpeed')) for j in passive])
        self.passive_accelerations=np.array([float(j.findtext('maxAcc')) for j in passive])

    def metadata(self):
        return {'name': 'rest_to_rest_quintic', 'polynomial': '10*tau^3-15*tau^4+6*tau^5',
            'normalized_time_domain': [0., 1.], 'boundary_velocity': [0., 0.], 'boundary_acceleration': [0., 0.],
            'duration_margin_factor': 1.05, 'robot_axes': {
                'maxSpeed_rad_s': self.speeds.tolist(), 'maxAcc_rad_s2': self.accelerations.tolist(),
                'coupling_matrix': self.coupling.tolist(), 'scope': 'Independent source joints and coupled physical axes'},
            'gun_axes': {'maxSpeed_source_units_s': self.gun_speeds.tolist(),
                'maxAcc_source_units_s2': self.gun_accelerations.tolist(),
                'units': ['mm', 'rad', 'mm', 'rad'], 'derivative_sample_count': 1001,
                'bound_method': 'Numerical normalized-time source-linkage derivatives; finite sampling'},
            'counterbalance_axes': {'keys':['473','474','475'],'units':['rad','mm','rad'],
                'maxSpeed_source_units_s':self.passive_speeds.tolist(),
                'maxAcc_source_units_s2':self.passive_accelerations.tolist(),'derivative_sample_count':1001,
                'source_sign_scope':'Relative rotation about source positive Y axes and signed extension along source 474 axis'},
            'source_sha256': {'robot': hashlib.sha256((ROOT/'build/robot.json').read_bytes()).hexdigest(),
                'gun_xml': hashlib.sha256(self.gun.source_xml.encode()).hexdigest(),
                'welding_kinematics':hashlib.sha256((ROOT/'welding_kinematics.py').read_bytes()).hexdigest()},
            'limitations': ['No motor torque, jerk limit, welding dwell or continuous collision certification']}

    def duration(self, qa, qb, da, db):
        # Independent source joint values and coupled physical axes both obey
        # their limits. Stops at every piecewise edge avoid velocity seams.
        duration = max(segment_duration(qa, qb, self.speeds, self.accelerations),
                       segment_duration(self.coupling @ qa, self.coupling @ qb, self.speeds, self.accelerations))
        t = np.linspace(0., 1., 1001)
        u = 10*t**3 - 15*t**4 + 6*t**5
        if qa[1] != qb[1]:
            passive=np.array([passive_joint_values(self.scene,qa+value*(qb-qa)) for value in u])
            velocity=np.gradient(passive,t,axis=0,edge_order=2)
            acceleration=np.gradient(velocity,t,axis=0,edge_order=2)
            duration=max(duration,float(np.max(abs(velocity)/self.passive_speeds)),
                float(np.sqrt(np.max(abs(acceleration)/self.passive_accelerations))))
        if da != db:
            # Dense normalized-time source linkage derivatives bound selected
            # timing numerically, with an explicit 5% margin and report scope.
            t = np.linspace(0., 1., 1001)
            u = 10*t**3 - 15*t**4 + 6*t**5
            positions = np.array([self.gun.joint_values(da + value*(db-da)) for value in u])
            velocity = np.gradient(positions, t, axis=0, edge_order=2)
            acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
            duration = max(duration, float(np.max(abs(velocity)/self.gun_speeds)),
                           float(np.sqrt(np.max(abs(acceleration)/self.gun_accelerations))))
        return duration * 1.05

    def sample_motion(self, qa, qb, da, db, joint_step=.01, gun_step=5.):
        T = self.duration(qa, qb, da, db)
        # Geometric intervals remain ≤ declared angle/driver resolution under
        # the quintic profile (peak normalized speed 1.875).
        n = max(1, len(fractions(qa, qb, da, db, joint_step/1.875, gun_step/1.875)) - 1)
        for tau in np.linspace(0., 1., n + 1):
            u = 10*tau**3 - 15*tau**4 + 6*tau**5
            yield float(T*tau), np.asarray(qa) + u*(np.asarray(qb)-qa), float(da + u*(db-da))


def load_inventory():
    """Use all saved inverse branches, plus independently found finer rolls."""
    rows = json.loads((ROOT / 'build/ik_source_normal_roll_optimized.json').read_text())['welds']
    inventory = {}
    for weld in rows:
        inventory[weld['id']] = [dict(solution, robot_id=robot['robot_id'], evidence='original_ik')
                                for robot in weld['robots'] for solution in robot['solutions']]
    fine = json.loads((ROOT / 'build/roll15_parked_search.json').read_text())
    scene = json.loads((ROOT / 'build/scene.json').read_text())
    rid = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R1' in a['name'])
    for row in fine['welds']:
        for attempt in row['attempts']:
            inventory[row['weld_id']].append(dict(attempt, robot_id=rid, evidence='roll15'))
    return inventory


class WeldTaskPlanner:
    def __init__(self, scene, output_dir, joint_step=.01, gun_step=5.):
        from collision_audit import CollisionWorld
        self.scene, self.output = scene, Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.world = CollisionWorld(scene)
        self.rid = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R1' in a['name'])
        self.other = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R2' in a['name'])
        self.model = self.world.models[self.rid]
        self.park = np.array(json.loads((ROOT / 'build/r2_safe_park_search.json').read_text())['selected']['q_rad'])
        self.world.update(self.other, self.park)
        self.timing = SourceTiming(scene, self.model.gun_model)
        self.joint_step, self.gun_step = joint_step, gun_step
        self.driver_open = float(self.model.gun_model.limits_mm[0])
        self.cache = {}
        self.audit = []

    def check(self, q, driver, weld=None):
        from r2_safe_park import additional_self_checks
        key = (tuple(np.round(q, 11)), round(driver, 9), None if weld is None else weld['id'])
        if key in self.cache:
            return self.cache[key]
        legal = self.model.limits_valid(q) and self.world.models[self.other].limits_valid(self.park)
        pairs = []
        if legal:
            pairs += self.world.check(self.rid, q, gun_driver_mm=driver, weld=weld)
            pairs += additional_self_checks(self.world, self.rid)
            pairs += self.world.check(self.other, self.park)
            pairs += additional_self_checks(self.world, self.other)
        result = {'joint_limits_valid': bool(legal), 'collision_verified': bool(legal and not pairs), 'pairs': pairs}
        self.cache[key] = result
        return result

    def edge(self, qa, qb, da, db, phase, weld=None, stop_on_failure=True):
        samples = []
        qa,qb=np.asarray(qa),np.asarray(qb)
        motion=list(self.timing.sample_motion(qa, qb, da, db, self.joint_step, self.gun_step))
        boundary=[] if weld is None else contact_boundary_fractions(self.model,qa,qb,weld)
        duration=self.timing.duration(qa,qb,da,db)
        for u in boundary:
            low,high=0.,1.
            for _ in range(40):
                tau=(low+high)/2.
                if 10*tau**3-15*tau**4+6*tau**5 < u:low=tau
                else:high=tau
            motion.append((duration*high,qa+u*(qb-qa),da+u*(db-da)))
        for t, q, d in sorted(motion,key=lambda x:x[0]):
            check = self.check(q, d, weld)
            samples.append(dict(check, time_s=t, q_rad=q.tolist(), driver_mm=d))
            if not check['collision_verified'] and stop_on_failure:
                break
        result = {'phase': phase, 'weld_id': None if weld is None else weld['id'], 'samples': samples,
                  'status': classify_route(samples), 'duration_s': self.timing.duration(np.asarray(qa), np.asarray(qb), da, db),
                  'time_law': self.timing.metadata(), 'q_start_rad': np.asarray(qa).tolist(),
                  'q_end_rad': np.asarray(qb).tolist(), 'driver_start_mm': float(da), 'driver_end_mm': float(db),
                  'contact_boundary_probe_fractions': boundary}
        self.audit.append({k: v for k, v in result.items() if k != 'samples'} | {'checked_samples': len(samples),
                           'failure': next((s for s in samples if not s['collision_verified']), None)})
        return result

    def options(self, weld, candidates, distances=(30., 60., 100., 160.)):
        from solve_welds import solve
        options, seen = [], []
        for candidate in candidates:
            if candidate['robot_id'] != self.rid:
                continue
            q = np.array(candidate['q_rad'])
            if any(np.linalg.norm(q - other) < .01 for other in seen):
                continue
            seen.append(q)
            endpoint = self.check(q, 0., weld)
            if not endpoint['collision_verified']:
                self.audit.append({'phase': 'endpoint', 'weld_id': weld['id'], 'status': 'endpoint_rejected',
                    'checked_samples': 1, 'failure': dict(endpoint, time_s=0., q_rad=q.tolist(), driver_mm=0.)})
                continue
            opening = self.edge(q, q, 0., self.driver_open, 'open', weld)
            if opening['status'] != 'planned':
                continue
            pose = self.model.tcp(q)
            for distance in distances:
                for sign in (-1., 1.):
                    target = pose.copy(); target[:3, 3] += sign * distance * pose[:3, 2]
                    for solution in solve(self.model, target, q, 4)[:2]:
                        for qr in constant_charts(solution['q_rad'], q, self.model)[:2]:
                            retreat = self.edge(q, qr, self.driver_open, self.driver_open, 'retreat', weld)
                            if retreat['status'] != 'planned':
                                continue
                            approach = self.edge(qr, q, self.driver_open, self.driver_open, 'approach', weld)
                            closing = self.edge(q, q, self.driver_open, 0., 'close', weld)
                            if approach['status'] == closing['status'] == 'planned':
                                options.append({'q_rad': q.tolist(), 'retreat_q_rad': qr.tolist(), 'distance_mm': sign*distance,
                                    'segments': [opening, retreat, approach, closing], 'candidate': candidate})
                            break
                    if len(options) >= 4:
                        return options
        return options

    def frame(self, sample, t, phase, weld_id):
        robots, guns = {}, {}
        for rid, model in self.world.models.items():
            q = np.array(sample['q_rad']) if rid == self.rid else self.park
            d = sample['driver_mm'] if rid == self.rid else 0.
            links = model.links(q, world=True)
            robots[rid] = {'q_rad': q.tolist(), 'links': {key: m.flatten().tolist() for key, m in links.items()}, 'tcp_xyz': model.tcp(q)[:3, 3].tolist()}
            pose = links['245'] @ model.gun_home
            deltas = model.gun_model.transforms(d)
            guns[model.gun_id] = {'transform': pose.flatten().tolist(), 'driver_mm': d,
                'parts': {pid: (pose @ deltas[p['link_key']]).flatten().tolist() for pid, p in model.gun_parts.items()}}
        return {'time_s': t, 'phase': phase, 'weld_id': weld_id, 'robots': robots, 'guns': guns,
                'collision_verified': sample['collision_verified'], 'collision_pairs': sample['pairs']}

    def connect(self, previous, current, weld):
        best = None
        for pa, pb in itertools.product(previous, current):
            transfer = self.edge(pa['retreat_q_rad'], pb['retreat_q_rad'], self.driver_open, self.driver_open, 'transfer')
            if transfer['status'] == 'planned':
                cost = transfer['duration_s']
                if best is None or cost < best[0]:
                    best = (cost, pa, pb, transfer)
        return best

    def run(self, group=None):
        result = plan_source_tasks(self.scene, self.output, load_inventory())
        all_options = {}
        for task in result['welds']:
            weld = next(w for w in self.scene['welds'] if w['id'] == task['id'])
            if group and weld['group'] != group:
                task['failure_reasons'] = ['outside_requested_group']
                continue
            print('TASK', weld['id'], 'endpoint candidates', len(task['endpoint_candidates']), flush=True)
            options = self.options(weld, task['endpoint_candidates'])
            task['verified_option_count'] = len(options)
            task['status'] = 'transit_rejected' if options else 'endpoint_unavailable'
            task['failure_reasons'] = [] if options else ['no_verified_endpoint_opening_and_retreat_option']
            all_options[task['id']] = options
            (self.output / 'progress.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        # Preserve source order. Unavailable tasks divide validated subroutes;
        # no transition over a missing source operation is silently implied.
        route, last, prev = [], None, None
        for task in result['welds']:
            options = all_options.get(task['id'], [])
            if not options or last is None or task['group'] != last['group']:
                if route:
                    result['routes'].append(route)
                route, prev, last = [], None, None
            if not options:
                continue
            if prev is None:
                route = [{'weld_id': task['id'], 'option': options[0], 'connector': None}]
                prev = [options[0]]
            else:
                weld = next(w for w in self.scene['welds'] if w['id'] == task['id'])
                connection = self.connect(prev, options, weld)
                if connection is None:
                    result['routes'].append(route)
                    route = [{'weld_id': task['id'], 'option': options[0], 'connector': None}]
                    task['failure_reasons'] = ['no_verified_connector_from_previous_source_task']
                    prev = [options[0]]
                else:
                    _, pa, pb, transfer = connection
                    route.append({'weld_id': task['id'], 'option': pb, 'connector': transfer})
                    prev = [pb]
            last = task
        if route:
            result['routes'].append(route)
        for index, route in enumerate(result['routes']):
            self.export_route(index, route)
            for item in route:
                task = next(t for t in result['welds'] if t['id'] == item['weld_id'])
                task['status'] = 'planned'
                task['route_index'] = index
        result['collision_search'] = self.audit
        result['coverage'] = {'source_welds': len(result['welds']), 'planned_source_tasks': sum(w['status'] == 'planned' for w in result['welds']),
            'connected_subroutes': [len(route) for route in result['routes']], 'unique_checked_states': len(self.cache)}
        result['status']['both_robots_sampled_collision_verified'] = bool(result['routes'])
        result['provenance'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('build/scene.json', 'build/robot.json', 'weld_task_planner.py', 'collision_audit.py', 'gun_kinematics.py', 'r2_safe_park.py')}
        result['timing_scope'] = 'Source independent/coupled robot axis maxSpeed/maxAcc; source nonlinear gun linkage speed/acc dense numerical timing bound with 5% margin; rest-to-rest quintic for every edge. Process dwell, motor torque and continuous collision are not certified.'
        result['time_law'] = self.timing.metadata()
        (self.output / 'task_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def export_route(self, index, route):
        frames, time_s, segments = [], 0., []
        # All states start at a separately collision-checked source endpoint,
        # rather than silently dropping an unsafe original current first frame.
        for i, item in enumerate(route):
            op = item['option']
            to_run = []
            if item['connector'] is not None:
                to_run.append(item['connector'])
                to_run += op['segments'][2:]
            else:
                # First source endpoint has close-state identity + actual open,
                # retreat, approach, close cycle as a verified standalone task.
                to_run += op['segments'][:2] + op['segments'][2:]
            for segment in to_run:
                for sample in segment['samples']:
                    frames.append(self.frame(sample, time_s + sample['time_s'], segment['phase'], item['weld_id']))
                time_s += segment['duration_s']
                segments.append(segment)
            if i < len(route)-1:
                # Next task connector begins with open gun at retreat. Explicit
                # open+retreat follows each weld closure, retaining phase seams.
                for segment in op['segments'][:2]:
                    for sample in segment['samples']:
                        frames.append(self.frame(sample, time_s + sample['time_s'], segment['phase'], item['weld_id']))
                    time_s += segment['duration_s']; segments.append(segment)
        status = {'collision_verified': False, 'system_safe': False, 'source_initial_unsafe': True, 'source_to_park_path_verified': False,
            'alternative_simulation_start': True, 'both_robots_sampled_collision_verified': True, 'planned_welds': len(route), 'partial_sequence': True,
            'orientation_source': 'alternate source projected normal; isProjected=0', 'diagnostic_only': False}
        validate_segment_chain(segments)
        output = {'status': status, 'frames': frames, 'segments': segments, 'selected_weld_ids': [x['weld_id'] for x in route],
            'source_order_indices': [next(w['sequence'] for w in self.scene['welds'] if w['id'] == x['weld_id']) for x in route],
            'welds': self.scene['welds'], 'start_contract': 'All original source-start collisions remain documented; this route starts at separately checked alternate endpoint with static verified R2 park, source→alternative transition unverified.',
            'duration_s': time_s, 'joint_resolution_rad': self.joint_step, 'gun_resolution_mm': self.gun_step,
            'park_q_rad': self.park.tolist(), 'time_law': self.timing.metadata()}
        (self.output / f'route_{index:02d}.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--group')
    parser.add_argument('--joint-step', type=float, default=.01)
    parser.add_argument('--gun-step', type=float, default=5.)
    args = parser.parse_args()
    scene = json.loads((ROOT / 'build/scene.json').read_text())
    planner = WeldTaskPlanner(scene, args.output, args.joint_step, args.gun_step)
    result = planner.run(args.group)
    print(result['coverage'], flush=True)


if __name__ == '__main__':
    main()
