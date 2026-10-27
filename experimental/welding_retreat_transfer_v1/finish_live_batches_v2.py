"""Finish dependency-ordered cold audits of the two current live experiments."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def run(name,arguments):
    log=Path('/tmp')/(name+'.log')
    with log.open('w') as stream:
        code=subprocess.call([str(ROOT/'.venv/bin/python'),*arguments],cwd=ROOT,
                             stdout=stream,stderr=subprocess.STDOUT)
    print('PIPELINE_STAGE',name,'EXIT',code,flush=True)
    if code:raise RuntimeError(name+' failed; see '+str(log))


def main():
    primary='build/welding_retreat_200cm_legal_chart_v6'
    diagnostic='build/welding_transfer_200mm_chart_v10'
    completed=set();start=time.monotonic()
    while len(completed)<2:
        if time.monotonic()-start>7200:
            raise RuntimeError('Planning reports still missing; no completion claim')
        if 'primary' not in completed and (ROOT/primary/'report.json').exists():
            run('welding_retreat_200cm_legal_chart_cold_v6',[
                'experimental/welding_retreat_transfer_v1/verify_v3.py',
                '--retreat-out',primary,'--out','build/welding_retreat_200cm_legal_chart_cold_v6','--workers','3'])
            run('welding_transfer_200cm_legal_chart_v10',[
                'experimental/welding_retreat_transfer_v1/run_transfer_v10.py',
                '--retreat-out',primary,'--retreat-cold','build/welding_retreat_200cm_legal_chart_cold_v6/cold_verification.json',
                '--out','build/welding_transfer_200cm_legal_chart_v10','--workers','2','--iterations','2400','--ordering-rounds','5'])
            run('welding_retreat_transfer_200cm_legal_chart_cold_v10',[
                'experimental/welding_retreat_transfer_v1/verify_v3.py',
                '--retreat-out',primary,'--transfer-out','build/welding_transfer_200cm_legal_chart_v10',
                '--out','build/welding_retreat_transfer_200cm_legal_chart_cold_v10','--workers','3'])
            completed.add('primary')
        if 'diagnostic' not in completed and (ROOT/diagnostic/'report.json').exists():
            run('welding_retreat_transfer_200mm_chart_cold_v10',[
                'experimental/welding_retreat_transfer_v1/verify_v3.py',
                '--retreat-out','build/welding_retreat_200mm_chart_v1','--transfer-out',diagnostic,
                '--out','build/welding_retreat_transfer_200mm_chart_cold_v10','--workers','3'])
            completed.add('diagnostic')
        if completed:
            subprocess.check_call([str(ROOT/'.venv/bin/python'),str(HERE/'write_status.py')],cwd=ROOT)
        if len(completed)<2:time.sleep(5.)
    (ROOT/'build/welding_retreat_transfer_live_finish.json').write_text(json.dumps(dict(
        finished=list(sorted(completed)),elapsed_s=time.monotonic()-start),indent=2))


if __name__=='__main__':main()
