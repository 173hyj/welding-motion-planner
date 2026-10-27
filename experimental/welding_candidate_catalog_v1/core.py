"""Pure candidate catalogue enumeration primitives."""

import json
import math


def pose_grid(robot_ids):
    """Enumerate relative gun poses independently for each robot."""
    return [{'robot_id': robot_id, 'roll_deg': roll, 'axis_sign': sign}
            for robot_id in robot_ids for roll in range(0, 360, 5)
            for sign in (1, -1)]


def angle_grid():
    """Positioner angles in center-out order with only one seam endpoint."""
    return [0.0] + [float(v) for n in range(5, 180, 5) for v in (n, -n)] + [180.0]


def candidate_key(weld_id, angle, pose):
    """Stable, unambiguous JSON key including every discrete dimension."""
    return json.dumps([str(weld_id), float(angle), str(pose['robot_id']),
                       float(pose['roll_deg']), int(pose['axis_sign'])],
                      ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def deduplicate_joint_states(qs, tolerance=1e-4):
    """Keep first raw actuator vector within an absolute per-joint tolerance.

    No angular wrapping is performed: different actuator windings are different
    states. This utility applies within a solver's results, never across poses.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    retained = []
    for q in qs:
        values = list(q)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('joint states must contain finite values')
        if not any(len(values) == len(previous) and
                   all(abs(float(a) - float(b)) <= tolerance
                       for a, b in zip(values, previous)) for previous in retained):
            retained.append(values)
    return retained


def enumerate_angle(weld_id, angle, poses, geometry_check, solve_ik, accept):
    """Visit every pose, checking its fixed geometry before solving any IK.

    Callbacks close over weld/angle geometry. Acceptance must evaluate only the
    active arm and fixed obstacles; another arm's parked configuration must not
    filter this single-arm catalogue. Every reported solver solution is retained,
    including rejected solutions and distinct windings. Search effort/stats are
    preserved, but numerical IK completeness is never asserted.
    """
    records = []
    for pose in poses:
        geometry = dict(geometry_check(pose))
        record = {'candidate_key': candidate_key(weld_id, angle, pose),
                  'weld_id': weld_id, 'angle_deg': float(angle), 'pose': dict(pose),
                  'geometry': geometry, 'ik_stats': {}, 'solutions': [],
                  'accepted': [], 'numerically_complete': False}
        if not geometry['passed']:
            record['status'] = 'geometry_rejected'
        else:
            solutions, stats = solve_ik(pose)
            record['ik_stats'] = stats
            for solution in solutions:
                outcome = dict(solution)
                outcome['acceptance'] = dict(accept(pose, solution))
                record['solutions'].append(outcome)
                if outcome['acceptance']['passed']:
                    record['accepted'].append(outcome)
            record['status'] = ('candidates_found' if record['accepted'] else
                                'no_accepted_ik' if record['solutions'] else 'ik_not_found')
        records.append(record)
    return records
