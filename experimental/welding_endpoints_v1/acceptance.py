import math


def posture_passed(state, tcp_error_mm, electrode_keys, gaps_mm):
    if (not math.isfinite(tcp_error_mm) or tcp_error_mm >= .1 or
            not state['both_source_joint_limits_valid'] or state['remaining_pairs']):
        return False
    for key in electrode_keys:
        certificate = state['certificates'].get(key)
        if certificate is not None:
            if not certificate['passed']:
                return False
        elif not math.isfinite(gaps_mm.get(key, math.inf)) or gaps_mm.get(key, math.inf) > .1:
            return False
    return len(electrode_keys) == 2
