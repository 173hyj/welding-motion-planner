"""Rank archived starts for a fresh direction domain, never reuse negatives."""
import copy


def select_seeds(reports, weld_id, limit):
    if limit < 1:
        raise ValueError('Positive finite seed budget required')
    unique = {}
    for name, report in reports:
        if report['weld_id'] != weld_id:
            continue
        angle = report['initial']['positioner_deg']
        for pose in report.get('alternative_pose_trials', []):
            for trial in pose['trials']:
                rays = trial.get('directions', [])
                if not rays:
                    continue
                def rank(ray):
                    return (ray.get('last_accepted_travel_mm') or 0.,
                            ray.get('kinematic_prefilter', {}).get('travel_mm') or 0.,
                            ray.get('endpoint_ik_candidates') or 0.)
                ray = max(rays, key=rank)
                score = rank(ray)
                key = (pose['candidate']['robot_id'], angle, tuple(trial['q_rad']))
                item = dict(source_report=name, positioner_deg=angle,
                    candidate=copy.deepcopy(pose['candidate']), q_rad=list(trial['q_rad']),
                    reference=list(ray['direction_world']), rank=list(score))
                if key not in unique or score > tuple(unique[key]['rank']):
                    unique[key] = item
    return rank_seeds(list(unique.values()), limit)


def rank_seeds(items, limit):
    if limit < 1:
        raise ValueError('Positive finite seed budget required')
    unique = {}
    for item in items:
        key = (item['candidate']['robot_id'], item['positioner_deg'], tuple(item['q_rad']))
        if key not in unique or tuple(item['rank']) > tuple(unique[key]['rank']):
            unique[key] = item
    ordered = sorted(unique.values(), key=lambda x: tuple(x['rank']), reverse=True)
    if not ordered:
        return []
    chosen = [ordered.pop(0)]
    while ordered and len(chosen) < limit:
        used = {x['candidate']['robot_id'] for x in chosen}
        index = next((i for i, x in enumerate(ordered) if x['candidate']['robot_id'] not in used), 0)
        chosen.append(ordered.pop(index))
    return chosen
