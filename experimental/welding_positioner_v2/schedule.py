"""Strict stage order and finite five-degree search grids."""
def search_angles(stage):
    if stage == 'stationary':
        return [0]
    if stage == 'positioner':
        return [sign*a for a in range(5, 180, 5) for sign in (1, -1)] + [180]
    raise ValueError(stage)


def remaining_rows(rows):
    return [i for i, row in enumerate(rows) if not row['passed'] and row['relative_gun_candidates']]
