"""External-axis expansion; reuse only sealed, identical finite-search negatives."""
import hashlib
import json
import math


def angle_sequence(current, angles):
    current = float(current)
    candidates = list(dict.fromkeys([current, *map(float, angles)]))
    if any(not math.isfinite(a) or not -180. <= a <= 180. for a in candidates):
        raise ValueError('Finite source external-axis angles in [-180, 180] required')
    return [current] + sorted((a for a in candidates if a != current),
                              key=lambda a: abs(a-current))


def search_until_positive(weld_id, angles, failed, solve):
    records = []
    for angle in angles:
        if (weld_id, angle) in failed:
            records.append(dict(weld_id=weld_id, positioner_deg=angle,
                                skipped_certified_negative=failed[weld_id, angle]))
            continue
        result = solve(angle)
        records.append(dict(weld_id=weld_id, positioner_deg=angle,
                            new_positive_found=bool(result['best']),
                            report_file=f'{weld_id}_angle_{angle:g}.json',
                            elapsed_s=result.get('elapsed_s')))
        if result['best']:
            return result, records
    return None, records


def normalize_result(result, weld_id, angle):
    result = dict(result)
    if (result.get('weld_id') != weld_id
            or result.get('initial', {}).get('positioner_deg') != angle):
        raise ValueError('Point-angle result identity mismatch')
    if result.get('reason') == 'positioner_layout_blocked' and not result['best']:
        result['required_straight_distance_mm'] = 2000.
    if result.get('required_straight_distance_mm') != 2000.:
        raise ValueError('Wrong or missing retreat distance')
    return result


def read_certified_failures(evidence, root, required_code_hashes):
    report = json.loads((evidence/'report.json').read_text())
    manifest = report['manifest']
    settings = manifest['settings']
    expected = dict(distance_mm=2000., posture_roll_step_deg=5.,
                    source_wrist_search_low_rad=1e-6, both_robot_assignment=True,
                    full_direction_bank=True, no_midpath_wrap=True,
                    cold_uses_full_source_limits=True)
    if report['summary'].get('complete') is not True or any(
            settings.get(k) != v for k, v in expected.items()):
        raise ValueError('Only completed identical positive-wrist finite searches reusable')
    if any(manifest['code_sha256'].get(k) != v for k, v in required_code_hashes.items()):
        raise ValueError('Different finite-search implementation')
    seals = {**manifest['source_sha256'], **manifest['code_sha256']}
    seals.update({str((evidence/name).relative_to(root)): sha
                  for name, sha in report['output_sha256'].items()})
    for name, sha in seals.items():
        p = root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError('Changed finite-search evidence '+name)
    failures = {}
    for record in report['search_records']:
        if record.get('new_positive_found') is not False:
            continue
        name = record['report_file']
        if name not in report['output_sha256']:
            raise ValueError('Negative result not sealed')
        result = json.loads((evidence/name).read_text())
        if (result.get('weld_id') != record['weld_id'] or result.get('best') is not None
                or result.get('passed') is not False
                or result.get('required_straight_distance_mm') != 2000.):
            raise ValueError('Negative identity/domain mismatch')
        angle = float(record['positioner_deg'])
        if not math.isfinite(angle) or not -180. <= angle <= 180.:
            raise ValueError('Invalid evidence angle')
        normalize_result(result, record['weld_id'], angle)
        failures[record['weld_id'], angle] = str((evidence/name).relative_to(root))
    return failures
