"""Cheap rejection before expensive geometry; never a trajectory certificate."""


def filter_ray(cache, key, kinematic, geometry):
    """Cache only kinematics within one frozen model domain, not geometry.

    The caller must include source robot, exact start configuration and direction
    in key. A successful result still requires the original full trace and cold
    verification. In particular each opening amplitude needs its own geometry.
    """
    if key not in cache:
        cache[key] = kinematic()
    kin = cache[key]
    if not kin['passed']:
        return dict(passed=False, reason='finite_continuation_prefilter_rejected',
                    kinematic_prefilter=kin)
    geom = geometry()
    if not geom['passed']:
        return dict(passed=False, reason='geometry_certificate_failed',
                    kinematic_prefilter=kin, geometry=geom)
    return dict(passed=True, reason='both_prefilters_passed_not_a_trace_certificate',
                kinematic_prefilter=kin, geometry=geom)
