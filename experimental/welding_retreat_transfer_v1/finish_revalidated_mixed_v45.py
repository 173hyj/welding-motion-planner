"""Verify new mixed inventory and plan transfers without reusing broken historical proofs."""
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PREFIX='experimental/welding_retreat_transfer_v1/'


def main():
    retreat='build/welding_retreat_200cm_mixed_wrist_v44'
    started=time.monotonic()
    print('WAIT_FOR_FRESH_MIXED_FULL21_REPORT',flush=True)
    while not (ROOT/retreat/'report.json').exists():
        if time.monotonic()-started>43200:
            raise RuntimeError('Fresh inventory unfinished; no completion claim')
        time.sleep(5)
    def stage(name,args):
        print('STAGE_START',name,flush=True)
        with Path('/tmp',name+'.log').open('w') as log:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,
                                 stdout=log,stderr=subprocess.STDOUT)
        print('STAGE_END',name,'EXIT',code,flush=True)
        if code:
            raise RuntimeError(name+' failed')
    cold='build/welding_retreat_200cm_mixed_wrist_cold_v44'
    stage('welding_retreat_200cm_mixed_wrist_cold_v44',[
        PREFIX+'verify_v3.py','--retreat-out',retreat,'--out',cold,'--workers','1'])
    transfer='build/welding_transfer_200cm_revalidated_mixed_v45'
    stage('welding_transfer_200cm_revalidated_mixed_v45',[
        PREFIX+'run_transfer_v30.py','--retreat-out',retreat,
        '--retreat-cold',cold+'/cold_verification.json','--out',transfer,
        '--workers','1','--iterations','2400','--ordering-rounds','5'])
    stage('welding_retreat_transfer_200cm_revalidated_mixed_cold_v45',[
        PREFIX+'verify_v3.py','--retreat-out',retreat,'--transfer-out',transfer,
        '--out','build/welding_retreat_transfer_200cm_revalidated_mixed_cold_v45','--workers','1'])
    stage('welding_revalidated_status_v45',[PREFIX+'write_recovery_status_v45.py'])
    print('FINISHED_BATCH_ONLY',time.monotonic()-started,flush=True)


if __name__=='__main__':
    main()
