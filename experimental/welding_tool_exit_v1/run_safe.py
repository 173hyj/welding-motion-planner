"""Public entry: immutable legacy cache, full input seals, no timing overwrite."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from run_initial import OUT as INITIAL_OUT, extended_source_hashes, assert_frozen_geometry_cache
from run_geometry import OUT as GEOMETRY_OUT, atomic, digest
from tool_world import HERE, ROOT


def cached_result(path, baseline):
    report=json.loads(Path(path).read_text());manifest=report['manifest']
    expected=manifest.get('source_sha256',manifest.get('baseline_sha256',{}))
    for name,value in expected.items():
        if baseline.get(name)!=value:raise RuntimeError('Completed result has different source: '+name)
    for name,value in manifest.get('code_sha256',{}).items():
        if digest(HERE/name)!=value:raise RuntimeError('Completed result used different code: '+name)
    if not report['summary']['complete']:return None
    return report['summary']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['initial','geometry','arm'],default='initial')
    parser.add_argument('--out',type=Path);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--geometry-out',type=Path,default=GEOMETRY_OUT);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();assert_frozen_geometry_cache();baseline=extended_source_hashes()
    out=(args.out or (INITIAL_OUT if args.stage=='initial' else GEOMETRY_OUT)).resolve()
    report_name={'initial':'initial_report.json','geometry':'geometry_report.json','arm':'arm_report.json'}[args.stage]
    report_path=out/report_name;seal_path=out/(args.stage+'_safe_input_seal.json')
    if args.resume and report_path.exists():
        if args.stage!='initial' and not seal_path.exists():
            raise RuntimeError('Historical run lacks full input seal; view its report or use a new --out')
        if seal_path.exists() and json.loads(seal_path.read_text())['source_sha256']!=baseline:
            raise RuntimeError('Actual input or imported source dependency changed')
        result=cached_result(report_path,baseline)
        if result is not None:
            print('COMPLETED_EVIDENCE_PRESERVED',json.dumps(result,ensure_ascii=False));return
    out.mkdir(parents=True,exist_ok=True)
    if report_path.exists() and not args.resume:raise FileExistsError('Use --resume or a fresh --out')
    if seal_path.exists():
        if json.loads(seal_path.read_text())['source_sha256']!=baseline:raise RuntimeError('Input seal mismatch')
    else:
        atomic(seal_path,dict(source_sha256=baseline,stage=args.stage))
    script={'initial':'run_initial.py','geometry':'run_geometry.py','arm':'run_arm.py'}[args.stage]
    argv=[sys.executable,str(HERE/script),'--workers',str(args.workers),'--out',str(out)]
    if args.resume:argv.append('--resume')
    if args.stage=='arm':argv.extend(['--geometry-out',str(args.geometry_out.resolve())])
    env=os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[name]='1'
    process=subprocess.run(argv,cwd=ROOT,env=env)
    if extended_source_hashes()!=baseline:raise RuntimeError('Source or historical evidence changed during execution')
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
