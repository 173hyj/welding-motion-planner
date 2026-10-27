"""Merge exact seed metadata without duplicate keywords or identity ambiguity."""
import copy


def selected_pose(original, candidate, **fields):
    if candidate['robot_id'] != fields['robot_id']:
        raise ValueError('Selected robot differs from source candidate')
    result=copy.deepcopy(original)
    result.update(copy.deepcopy(candidate))
    result.update(copy.deepcopy(fields))
    return result
