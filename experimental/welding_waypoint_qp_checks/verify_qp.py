"""Cold source-mesh/source-FK validation for newly rescued QP paths only."""
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
from run_geometry import atomic,digest,initialize_worker
from run_initial import assert_frozen_geometry_cache
from tool_world import ROOT
from variable_world import VariableToolWorld


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'build/welding_waypoint_qp_v2');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Use1 to4 workers')
    out=args.out.resolve();rp=out/'report.json';report=json.loads(rp.read_text())
    if not report['summary']['complete']:raise RuntimeError('Complete QP experiment required')
    assert_frozen_geometry_cache();hashes={}
    for section in ('source_sha256','previous_code_sha256','code_sha256'):hashes.update(report['manifest'][section])
    for name,value in hashes.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Source/code changed: '+name)
    for name,value in report['output_sha256'].items():
        if digest(out/name)!=value:raise RuntimeError('Output changed: '+name)
    selected=[r for r in report['rows'] if r['best']];rows=[];begin=time.perf_counter()
    spec=importlib.util.spec_from_file_location('variable_cold',ROOT/'experimental/welding_variable_open_v1/verify.py')
    cold=importlib.util.module_from_spec(spec);spec.loader.exec_module(cold)
    if selected:
        scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text());welds={w['id']:w for w in scene['welds']}
        cold._ENGINE=cold.old_verify.fresh_source_world(scene);cold._ENGINE.__class__=VariableToolWorld
        fk_spec=importlib.util.spec_from_file_location('raw_fk',ROOT/'scripts/audit_contact_release_v17.py')
        cold._FK=importlib.util.module_from_spec(fk_spec);fk_spec.loader.exec_module(cold._FK)
        raw=json.loads((ROOT/'build/robot.json').read_text());cold._JOINTS=cold._FK.joints_from_raw(raw)[:6]
        cold._TCP={a['id']:cold._FK.source_tcp_home(raw,a) for a in scene['assets'] if a['kind']=='robot'}
        vo=out/'cold_verification';vo.mkdir(exist_ok=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as pool:
            fs=[pool.submit(cold.verify_point,r,welds[r['weld_id']],vo,out) for r in selected]
            rows=[f.result() for f in concurrent.futures.as_completed(fs)]
    for name,value in hashes.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Source/code changed during verification')
    result=dict(report_sha256=digest(rp),verified_rows=rows,verification_code_sha256={str(Path(__file__).relative_to(ROOT)):digest(__file__),
          'experimental/welding_variable_open_v1/verify.py':digest(ROOT/'experimental/welding_variable_open_v1/verify.py'),
          'scripts/audit_contact_release_v17.py':digest(ROOT/'scripts/audit_contact_release_v17.py')},
          summary=dict(new_selected_paths=len(selected),reverified_retreats=sum(r['passed'] for r in rows),all_passed=all(r['passed'] for r in rows),wall_s=time.perf_counter()-begin),
          scope='Newly successful paths only; raw source fixture rebuild, independent source XML FK agreement gate,0.5mm opening and q-edge midpoints. '
                'Same source contact predicate and assembly exemptions; no full-system continuous safety certificate.',system_safe=False)
    atomic(out/'cold_verification.json',result);print('QP_COLD_SUMMARY',json.dumps(result['summary']),flush=True)
    if not result['summary']['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
