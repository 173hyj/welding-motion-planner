"""Seal full mixed-domain inventory, verify, plan true transfers and verify again."""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'experimental/welding_retreat_transfer_v1/'


def main():
    inputs = ['welding_retreat_200cm_mixed_wrist_v41','welding_retreat_200cm_cell_inventory_v20',
              'welding_retreat_200cm_positive_angles_v35','welding_retreat_200cm_other_park_v12']
    started = time.monotonic()
    print('WAIT_FOR_FULL_MIXED_AND_CELL_REPORTS',flush=True)
    while not all((ROOT/'build'/f/'report.json').exists() for f in inputs):
        if time.monotonic()-started>43200:
            raise RuntimeError('Required reports unfinished; no completion claim')
        time.sleep(5)
    def stage(name,args):
        print('STAGE_START',name,flush=True)
        with Path('/tmp',name+'.log').open('w') as log:
            code = subprocess.call([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,
                                   stdout=log,stderr=subprocess.STDOUT)
        print('STAGE_END',name,'EXIT',code,flush=True)
        if code:
            raise RuntimeError(name+' failed')
    retreat = 'build/welding_retreat_200cm_merged_mixed_v42'
    merge = [PREFIX+'merge_retreat_inventory_v19.py','--out',retreat]
    for f in inputs:
        merge += ['--input','build/'+f]
    stage('welding_retreat_200cm_merged_mixed_v42',merge)
    cold = 'build/welding_retreat_200cm_merged_mixed_cold_v42'
    stage('welding_retreat_200cm_merged_mixed_cold_v42',[
        PREFIX+'verify_v3.py','--retreat-out',retreat,'--out',cold,'--workers','1'])
    seed_file = 'build/welding_transfer_200cm_merged_mixed_seed_v42.json'
    seed = [PREFIX+'snapshot_transfer_seeds_v28.py','--retreat-out',retreat,
            '--retreat-cold',cold+'/cold_verification.json','--out',seed_file]
    for transfer,proof in [
        ('welding_transfer_200cm_actual_fragments_v33','welding_retreat_transfer_200cm_actual_fragments_cold_v33'),
        ('welding_transfer_200cm_other_park_v12','welding_retreat_transfer_200cm_other_park_cold_v12')]:
        seed += ['--source','build/'+transfer,'build/'+proof+'/cold_verification.json']
    stage('welding_transfer_200cm_merged_mixed_seed_v42',seed)
    transfer = 'build/welding_transfer_200cm_merged_mixed_v42'
    stage('welding_transfer_200cm_merged_mixed_v42',[
        PREFIX+'run_transfer_v30.py','--retreat-out',retreat,
        '--retreat-cold',cold+'/cold_verification.json','--seed-snapshot',seed_file,
        '--out',transfer,'--workers','1','--iterations','2400','--ordering-rounds','5'])
    stage('welding_retreat_transfer_200cm_merged_mixed_cold_v42',[
        PREFIX+'verify_v3.py','--retreat-out',retreat,'--transfer-out',transfer,
        '--out','build/welding_retreat_transfer_200cm_merged_mixed_cold_v42','--workers','1'])
    stage('welding_retreat_transfer_200cm_write_status_v42',[PREFIX+'write_status.py'])
    print('FINISHED_BATCH_ONLY',time.monotonic()-started,flush=True)


if __name__=='__main__':
    main()
