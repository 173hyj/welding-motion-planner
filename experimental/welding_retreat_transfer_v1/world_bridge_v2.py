"""Source-mesh checker for both robots, both guns and the external axis."""
import sys
from pathlib import Path
import numpy as np
from state_space import interpolate

ROOT = Path(__file__).resolve().parents[2]


def node_state(node, robot_ids):
    rid, other = node['robot_id'], node['other_robot_id']
    if len(robot_ids) != 2 or {rid, other} != set(robot_ids):
        raise ValueError('Two different source robots must be present')
    state = np.zeros(15)
    for i, robot in enumerate(robot_ids):
        state[6*i:6*i+6] = node['q_end_rad'] if robot == rid else node['other_q_rad']
        state[13+i] = node['driver_mm'] if robot == rid else 0.
    state[12] = np.deg2rad(node['positioner_deg'])
    return state


class SourceBridge:
    def __init__(self, engine):
        sys.path.insert(0, str(ROOT))
        from r2_safe_park import additional_self_checks
        self.self_checks = additional_self_checks
        self.engine = engine
        self.world = engine.world
        self.robot_ids = sorted(self.world.models)
        self.scale = np.r_[np.ones(13), [100., 100.]]
        self.low = np.concatenate([*[self.world.models[r].bounds[0] for r in self.robot_ids],
                                  [engine.environment.positioner.bounds[0], -185., -185.]])
        self.high = np.concatenate([*[self.world.models[r].bounds[1] for r in self.robot_ids],
                                   [engine.environment.positioner.bounds[1], 0., 0.]])
        self.resolutions = np.r_[np.full(12, np.deg2rad(.05)), np.deg2rad(.25), .5, .5]
        self.checks = 0
        self.cache = {}
        self.last_failure = None

    def probe(self, state):
        state = np.asarray(state, float)
        if state.shape != (15,) or not np.isfinite(state).all():
            return dict(passed=False, reason='invalid_state_shape_or_nonfinite')
        if np.any(state < self.low-1e-8) or np.any(state > self.high+1e-8):
            return dict(passed=False, reason='source_axis_bounds')
        for i, rid in enumerate(self.robot_ids):
            if not self.world.models[rid].limits_valid(state[6*i:6*i+6]):
                return dict(passed=False, reason='source_robot_coupled_limits', robot_id=rid)
        angle = float(np.rad2deg(state[12]))
        layout = self.engine.environment.layout_collisions(angle)
        if layout['blocking_pairs']:
            return dict(passed=False, reason='positioner_layout_collision',
                        blocking_pairs=layout['blocking_pairs'])
        # layout_collisions may return a cached answer without setting transforms.
        self.engine.set_angle(angle)
        for i, rid in enumerate(self.robot_ids):
            self.world.update(rid, state[6*i:6*i+6], float(state[13+i]))
        pairs = []
        for i, rid in enumerate(self.robot_ids):
            raw = self.world.check(rid, state[6*i:6*i+6], gun_driver_mm=float(state[13+i]),
                                   max_pairs=sys.maxsize)
            raw += self.self_checks(self.world, rid)
            pairs.extend(dict(p, checked_robot_id=rid) for p in raw)
        return dict(passed=not pairs, reason='full_source_check' if not pairs else 'robot_or_gun_collision',
                    blocking_pairs=pairs)

    def valid_scaled(self, scaled):
        state = np.asarray(scaled)*self.scale
        # Exact float keys avoid rounding a narrowly valid state into another state.
        key = tuple(state)
        if key not in self.cache:
            self.checks += 1
            result = self.probe(state)
            self.cache[key] = result['passed']
            if not result['passed']:
                self.last_failure = dict(state=state.tolist(), **result)
        return self.cache[key]

    def edge_scaled(self, a, b):
        # Early rejection only; every positive edge still undergoes the full
        # fine sampling. This reduces long invalid-edge checks near obstacles.
        a, b = np.asarray(a), np.asarray(b)
        for u in (0., 1., .5, .25, .75):
            if not self.valid_scaled(a+(b-a)*u):
                return False
        return all(self.valid_scaled(s) for s in interpolate(a, b, self.resolutions/self.scale))

    def frame(self, state, phase='transfer'):
        state = np.asarray(state)
        return dict(state=state.tolist(), positioner_deg=float(np.rad2deg(state[12])),
                    robots={rid: dict(q_rad=state[6*i:6*i+6].tolist(),
                                     driver_mm=float(state[13+i]),
                                     tcp_xyz_mm=self.world.models[rid].tcp(state[6*i:6*i+6])[:3, 3].tolist())
                            for i, rid in enumerate(self.robot_ids)}, phase=phase)

    def dense_frames(self, path, phase='transfer'):
        frames = []
        for a, b in zip(path, path[1:]):
            samples = interpolate(a, b, self.resolutions/self.scale)
            for state in samples if not frames else samples[1:]:
                if not self.valid_scaled(state):
                    raise RuntimeError('Saved transfer interpolation failed source check')
                frames.append(self.frame(np.asarray(state)*self.scale, phase))
        if len(path) == 1:
            if not self.valid_scaled(path[0]):
                raise RuntimeError('Invalid single transfer state')
            frames.append(self.frame(np.asarray(path[0])*self.scale, phase))
        return frames
