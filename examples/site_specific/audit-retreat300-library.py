"""Independent catalog/receipt audit. Does not rerun FCL collision checking."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(report_file, baseline_file, catalog_file, root, seeds_file, allow_pending=False):
    report, baseline = read(report_file), read(baseline_file)
    require(len(report['layers']) == len(baseline['layers']) == 21, 'Expected21 task layers')
    fields = ('task', 'weld_id', 'angle', 'robot_id', 'arm')
    expected = {}
    with sqlite3.connect(Path(catalog_file).resolve().as_uri()+'?mode=ro', uri=True) as db:
        for layer, old in zip(report['layers'], baseline['layers']):
            require(all(layer[k] == old[k] for k in fields), 'Original catalog conditions changed')
            for row in db.execute('SELECT rowid,angle,weld,robot,roll,sign,branch,payload FROM candidates WHERE angle=? AND weld=? AND robot=?',
                                  (old['angle'], old['weld_id'], old['robot_id'])):
                rowid, angle, weld, robot, roll, sign, branch, payload = row
                expected[rowid] = dict(task=old['task'], catalog_key=dict(angle=angle,weld=weld,robot=robot,roll=roll,sign=sign,branch=branch),
                    payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    catalog_state=dict(json.loads(payload),robot_id=robot,roll_deg=roll,axis_sign=sign,catalog_branch=branch))
    require(len(expected) == 784, 'Original registered784-row population differs')
    registry = report['registry']
    require(len(registry) == len(expected) and len({r['catalog_rowid'] for r in registry}) == len(expected), 'Missing/duplicate candidate rows')
    for name, checksum in report['manifest']['bindings'].items():
        require(sha(name) == checksum, 'Changed sealed input: '+name)
    bindings = report['manifest']['bindings']
    require(bindings.get(str(Path(catalog_file).resolve())) == sha(catalog_file), 'Catalog not sealed')
    sys.path[:0] = [str(root), str(root/'experimental/welding_retreat_transfer_v1')]
    from integrated_retreat_v117 import check_path_contract
    certified = {}
    for row in registry:
        require(row['catalog_rowid'] in expected, 'Foreign candidate')
        for key, value in expected[row['catalog_rowid']].items():
            require(row[key] == value, 'Catalog row identity changed: '+key)
        require(row['status'] in ('pending','certified','search_unresolved','proven_invalid'), 'Unknown candidate status')
        if row['status'] == 'certified':
            require('action' in row, 'Certified row missing action')
            action = row['action']
            initial = action['initial']
            state = row['catalog_state']
            require(all(initial[k] == state[k] for k in ('robot_id','roll_deg','axis_sign','q_rad')), 'Certified action changed IK candidate')
            require(initial['positioner_deg'] == row['catalog_key']['angle'], 'Certified action changed angle')
            certified[action['id']] = (row['task'], action)
    seed_report = read(seeds_file)
    require(seed_report.get('complete') and seed_report.get('source_assets_frozen_before_and_after'), 'Seed report is not sealed')
    require(bindings.get(str(Path(seeds_file).resolve())) == sha(seeds_file), 'Seed report not bound')
    seeds = {r['task']:r for r in seed_report['rows'] if r['length_mm']==300. and r['passed']}
    require(set(seeds)==set(range(1,22)), 'Expected21 sealed300mm seeds')
    seen_certified = set()
    seen_seeds = set()
    layer_actions = set()
    for layer in report['layers']:
        require(bool(layer['actions']), 'Task has no300mm action')
        for action in layer['actions']:
            aid = action['id']
            if aid in certified:
                task, expected_action = certified[aid]
                require(task == layer['task'] and action == expected_action, 'Eligible catalog action metadata changed')
                require(aid not in seen_certified, 'Duplicate eligible catalog action')
                seen_certified.add(aid)
            else:
                task = layer['task']
                seed = seeds[task]
                require(aid == 'incumbent300-'+str(task) and action['source']=='certified300_seed', 'Foreign noncatalog action')
                require(task not in seen_seeds, 'Duplicate incumbent seed')
                for key in ('initial','path_file','path_sha256'):
                    require(action[key]==seed[key], 'Incumbent300 seed changed: '+key)
                seen_seeds.add(task)
            path_file = action['path_file']
            require(sha(path_file) == action['path_sha256'], 'Action artifact changed')
            path = read(path_file)
            require(path['weld_id'] == layer['weld_id'] and path['robot_id'] == layer['robot_id'] and path['positioner_deg'] == layer['angle'], 'Layer action identity differs')
            check_path_contract(path, action['initial'], 300.)
            require(path['online_validation']['endpoint_clearance_mm'] >= 5., 'Insufficient endpoint clearance')
            layer_actions.add(path_file)
    require(set(certified)==seen_certified, 'Certified catalog actions omitted from eligible layers')
    require(seen_seeds==set(seeds), 'Original300mm seeds omitted')
    counts = Counter(r['status'] for r in registry)
    attempted = len(registry)-counts['pending']
    require(report['coverage_all_rows']['registered'] == len(registry), 'Coverage count differs')
    require(report['coverage_all_rows']['attempted'] == attempted, 'Attempted count differs')
    require(bool(report['complete']) == (counts['pending'] == 0), 'Completeness differs')
    if not allow_pending:
        require(report['complete'], 'Library still has pending candidates')
        require(report.get('source_assets_frozen_before_and_after') is True, 'No final source asset seal')
        before=read(Path(report_file).parent/'source_before.json')['source_sha256']
        after=read(Path(report_file).parent/'source_after.json')
        require(after.get('unchanged') is True and before==after['source_sha256']==report['manifest']['source_sha256'], 'Source before/after receipts differ')
        for name, checksum in before.items():
            path=str((root/name).resolve())
            require(bindings.get(path)==checksum and sha(path)==checksum, 'Source receipt not bound/current: '+path)
    return dict(passed=True, complete=bool(report['complete']), registered=len(registry), attempted=attempted,
        statuses=dict(counts), eligible_action_paths=len(layer_actions), all21_covered=True,
        all_certified_rows_eligible=True, action_contract_length_mm=300., report_sha256=sha(report_file),
        catalog_sha256=sha(catalog_file), independent_artifact_audit=True, independent_collision_cold_run=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('report','baseline','catalog','root','seeds','out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--allow-pending', action='store_true')
    a=p.parse_args()
    result=audit(a.report,a.baseline,a.catalog,a.root,a.seeds,a.allow_pending)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
