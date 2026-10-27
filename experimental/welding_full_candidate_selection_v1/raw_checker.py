"""Original source collision checks with independent, narrowly scoped contacts.

Inputs use radians for the twelve robot joints and shared positioner, and mm
for the two drivers. Welds in contexts must already be transformed to that
positioner angle. This checker does not authorize or generate weld tasks.
"""
import sys
from pathlib import Path
import numpy as np

ROBOT_IDS = ['F46171CC-24BD-45DD-82E8-21711C110EE9',
             'F7690943-8A45-40E4-9AD3-F92856F982FA']


class RawDualSourceChecker:
    def __init__(self, engine, *, self_checks=None, qualify_pairs=None):
        root = Path(__file__).resolve().parents[2] if self_checks is None or qualify_pairs is None else None
        if self_checks is None:
            sys.path.insert(0, str(root))
            from r2_safe_park import additional_self_checks
            self_checks = additional_self_checks
        if qualify_pairs is None:
            sys.path.insert(0, str(root/'experimental/welding_tool_exit_v1'))
            from run_arm import qualify_full_pairs
            qualify_pairs = qualify_full_pairs
        self.engine = engine
        self.world = engine.world
        self.robot_ids = sorted(self.world.models)
        if self.robot_ids != ROBOT_IDS:
            raise ValueError('Exact source robot order required')
        self.self_checks = self_checks
        self.qualify_pairs = qualify_pairs
        self.low = np.concatenate([*[self.world.models[r].bounds[0] for r in self.robot_ids],
                                   [engine.environment.positioner.bounds[0], -185., -185.]])
        self.high = np.concatenate([*[self.world.models[r].bounds[1] for r in self.robot_ids],
                                    [engine.environment.positioner.bounds[1], 0., 0.]])
        self.resolutions = np.r_[np.full(12, np.deg2rad(.05)), np.deg2rad(.25), .5, .5]
        self.checks = 0

    def probe(self, state15, contexts=(None, None)):
        self.checks += 1
        state = np.asarray(state15, float)
        if state.shape != (15,) or not np.isfinite(state).all():
            return dict(passed=False, reason='invalid_state_shape_or_nonfinite')
        if np.any(state < self.low-1e-8) or np.any(state > self.high+1e-8):
            return dict(passed=False, reason='source_axis_bounds')
        if len(contexts) != 2:
            raise ValueError('Two per-arm contexts required')
        for i, rid in enumerate(self.robot_ids):
            if not self.world.models[rid].limits_valid(state[6*i:6*i+6]):
                return dict(passed=False, reason='source_robot_coupled_limits', robot_id=rid)
            context = contexts[i]
            if context is not None and (context['robot_id'] != rid or
                    context['phase'] not in ('closed', 'opening', 'release', 'separated')):
                raise ValueError('Context must match source arm and contact phase')
        angle = float(np.rad2deg(state[12]))
        layout = self.engine.environment.layout_collisions(angle)
        if layout['blocking_pairs']:
            return dict(passed=False, reason='positioner_layout_collision', blocking_pairs=layout['blocking_pairs'])
        # Always restore source scene transforms (layout queries may be cached),
        # but retain transformed static meshes when their exact transform is
        # unchanged. No contact certificates or collision results are cached.
        previous_delta = getattr(self.engine, 'delta', None)
        if previous_delta is not None:
            previous_delta = np.array(previous_delta, copy=True)
        previous_products = getattr(self.engine, 'product_cache', None)
        self.engine.set_angle(angle)
        if previous_delta is not None and previous_products is not None and np.array_equal(
                previous_delta, getattr(self.engine, 'delta', None)):
            self.engine.product_cache = previous_products
        def restore():
            for i, rid in enumerate(self.robot_ids):
                self.world.update(rid, state[6*i:6*i+6], float(state[13+i]))
        restore()
        tools = {}
        try:
            for i, rid in enumerate(self.robot_ids):
                context = contexts[i]
                if context is not None:
                    tools[rid] = self.engine.state(rid,
                        self.world.models[rid].tcp(state[6*i:6*i+6]), float(state[13+i]),
                        context['weld'], self.world.static, context['phase'],
                        context.get('release_context'))
        finally:
            # Contact queries move gun objects; source pair checks need both
            # actual drivers even when a query fails or returns early.
            restore()
        pairs = []
        for i, rid in enumerate(self.robot_ids):
            raw = self.world.check(rid, state[6*i:6*i+6],
                gun_driver_mm=float(state[13+i]), max_pairs=sys.maxsize)
            raw += self.self_checks(self.world, rid)
            pairs.extend(dict(p, checked_robot_id=rid) for p in raw)
        remaining = pairs
        blocking = []
        for i, rid in enumerate(self.robot_ids):
            if rid in tools:
                context = contexts[i]
                remaining = self.qualify_pairs(remaining, rid, context['weld']['product_id'],
                                               tools[rid], context['phase'])
                blocking.extend(dict(p, checked_robot_id=rid) for p in tools[rid]['blocking_pairs'])
        passed = not remaining and not blocking
        return dict(passed=passed,
                    reason='full_source_pair_check' if passed else 'full_arm_or_tool_collision',
                    remaining_pairs=remaining, blocking_pairs=remaining+blocking,
                    tool_blocking_pairs=blocking, tool_states=tools,
                    source_joint_limits_valid=True)

    def validate_edge(self, a, b, context_at=None):
        """Check endpoints and all original-resolution interpolation samples.

        The caller splits at timeline phase boundaries; context_at(u) supplies
        per-arm contexts for each sample, including both endpoints.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != (15,) or b.shape != (15,) or not np.isfinite([a,b]).all():
            return dict(passed=False, reason='invalid_edge_state')
        count = max(1, int(np.ceil(np.max(np.abs(b-a)/self.resolutions))))
        for index in range(count+1):
            u = index/count
            result = self.probe(a+(b-a)*u, context_at(u) if context_at else (None,None))
            if not result['passed']:
                return dict(passed=False, reason='dense_edge_source_check_failed',
                            samples=index+1, failure_u=u, failure=result)
        return dict(passed=True, reason='dense_edge_source_checked', samples=count+1)
