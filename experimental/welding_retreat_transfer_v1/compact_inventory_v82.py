"""Reference inherited search metadata and link sealed, locally readable paths."""
import copy
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_rows(rows, report_path, root):
    report_path = Path(report_path).resolve()
    archived = json.loads(report_path.read_text())['rows']
    if len(rows) != len(archived):
        raise ValueError('Archive row count mismatch')
    seal = sha256(report_path)
    result = copy.deepcopy(rows)
    for index, (row, old) in enumerate(zip(result, archived)):
        if row['weld_id'] != old['weld_id']:
            raise ValueError('Archive identity mismatch')
        if 'alternative_pose_trials' in row:
            if row['alternative_pose_trials'] != old.get('alternative_pose_trials'):
                raise ValueError('Archive search metadata mismatch')
            row.pop('alternative_pose_trials')
            row['search_archive_reference'] = dict(
                report_file=str(report_path.relative_to(Path(root).resolve())),
                report_sha256=seal, row_index=index, weld_id=row['weld_id'],
                field='alternative_pose_trials')
    return result


def resolve_trials(row, root):
    ref = row['search_archive_reference']
    path = Path(root) / ref['report_file']
    if sha256(path) != ref['report_sha256']:
        raise ValueError('Search archive changed')
    old = json.loads(path.read_text())['rows'][ref['row_index']]
    if old['weld_id'] != row['weld_id'] or old['weld_id'] != ref['weld_id']:
        raise ValueError('Search archive identity mismatch')
    return old[ref['field']]


def link_candidate(weld_id, candidate, source, out, name):
    src = Path(source) / candidate['path_file']
    if sha256(src) != candidate['path_sha256']:
        raise ValueError('Candidate changed')
    path = json.loads(src.read_text())
    if (path.get('weld_id', weld_id) != weld_id or not path.get('passed') or
            path.get('length_mm') != 2000. or path.get('fixed_distance_mm') != 2000.):
        raise ValueError('Candidate identity/distance mismatch')
    target = Path(out) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Sealed archives are immutable; never replace or edit a linked target.
    os.link(src, target)
    result = copy.deepcopy(candidate)
    result['path_file'] = name
    return result
