"""Finish complete inventory, positive-wrist expansion, real transfers and cold audit."""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'experimental/welding_retreat_transfer_v1/'


def main():
    inputs = ['welding_retreat_200cm_positive_angles_v35',
              'welding_retreat_200cm_positive_wrist_v31',
              'welding_retreat_200cm_other_park_v12',
              'welding_retreat_200cm_cell_inventory_v20',
              'welding_retreat_200cm_analytic_positioner_v11']
    started = time.monotonic()
    print('WAIT_FOR_FULL_CELL_AND_EXPANDED_WRIST_REPORTS', flush=True)
    while not all((ROOT/'build'/f/'report.json').exists() for f in inputs):
        if time.monotonic()-started > 43200:
            raise RuntimeError('Required reports not finished; no completion claim')
        time.sleep(5)
    def stage(name, args):
        print('STAGE_START', name, flush=True)
        with Path('/tmp', name+'.log').open('w') as log:
            code = subprocess.call([str(ROOT/'.venv/bin/python'), *args], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        print('STAGE_END', name, 'EXIT', code, flush=True)
        if code:
            raise RuntimeError(name+' failed')
    merge = [PREFIX+'merge_retreat_inventory_v19.py', '--out',
             'build/welding_retreat_200cm_merged_positive_angles_v36']
    for folder in inputs:
        merge += ['--input', 'build/'+folder]
    stage('welding_retreat_200cm_merged_positive_angles_v36', merge)
    retreat = 'build/welding_retreat_200cm_remaining_positive_angles_v37'
    cold = 'build/welding_retreat_200cm_remaining_positive_angles_cold_v37'
    stage('welding_retreat_200cm_remaining_positive_angles_v37', [
        PREFIX+'probe_positive_angles_v35.py', '--retreat-out',
        'build/welding_retreat_200cm_merged_positive_angles_v36', '--out', retreat,
        '--negative-evidence', 'build/welding_retreat_200cm_positive_wrist_v31',
        '--negative-evidence', 'build/welding_retreat_200cm_positive_angles_v35'])
    stage('welding_retreat_200cm_remaining_positive_angles_cold_v37', [
        PREFIX+'verify_v3.py', '--retreat-out', retreat, '--out', cold, '--workers', '1'])
    seed_file = 'build/welding_transfer_200cm_remaining_positive_angles_seed_v37.json'
    seed = [PREFIX+'snapshot_transfer_seeds_v28.py', '--retreat-out', retreat,
            '--retreat-cold', cold+'/cold_verification.json', '--out', seed_file]
    for transfer, proof in [
        ('welding_transfer_200cm_other_park_v12', 'welding_retreat_transfer_200cm_other_park_cold_v12'),
        ('welding_transfer_200cm_actual_fragments_v33', 'welding_retreat_transfer_200cm_actual_fragments_cold_v33'),
        ('welding_transfer_200cm_coarse_search_v30', 'welding_retreat_transfer_200cm_coarse_search_cold_v30')]:
        if (ROOT/'build'/transfer/'report.json').exists() and (ROOT/'build'/proof/'cold_verification.json').exists():
            seed += ['--source', 'build/'+transfer, 'build/'+proof+'/cold_verification.json']
    stage('welding_transfer_200cm_remaining_positive_angles_seed_v37', seed)
    transfer = 'build/welding_transfer_200cm_remaining_positive_angles_v37'
    stage('welding_transfer_200cm_remaining_positive_angles_v37', [
        PREFIX+'run_transfer_v30.py', '--retreat-out', retreat,
        '--retreat-cold', cold+'/cold_verification.json', '--seed-snapshot', seed_file,
        '--out', transfer, '--workers', '1', '--iterations', '2400', '--ordering-rounds', '5'])
    stage('welding_retreat_transfer_200cm_remaining_positive_angles_cold_v37', [
        PREFIX+'verify_v3.py', '--retreat-out', retreat, '--transfer-out', transfer,
        '--out', 'build/welding_retreat_transfer_200cm_remaining_positive_angles_cold_v37', '--workers', '1'])
    stage('welding_retreat_transfer_200cm_write_status_v37', [PREFIX+'write_status.py'])
    print('FINISHED_BATCH_ONLY', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
