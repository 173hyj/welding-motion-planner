"""Fresh cold -> actual mixed-domain transfers -> final cold; no historic probe."""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PREFIX='experimental/welding_retreat_transfer_v1/'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retreat-out')
    parser.add_argument('--retreat-cold',help='Optional fresh certificate; transfer runner rechecks exact report/path hashes')
    parser.add_argument('--tag',default='revalidated_merged_v57')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Simple output tag required')
    started=time.monotonic()
    def stage(name,command):
        print('STAGE_START',name,command,flush=True)
        with Path('/tmp',name+'.log').open('x') as log:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),*command],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT)
        print('STAGE_END',name,'EXIT',code,flush=True)
        if code:raise RuntimeError(name+' failed; incomplete goal')
    if args.retreat_out:
        retreat=args.retreat_out
    else:
        live='build/welding_retreat_200cm_mixed_wrist_v44'
        print('WAIT_FOR_FRESH_MIXED_FULL21_REPORT',flush=True)
        while not (ROOT/live/'report.json').exists():time.sleep(5)
        summary=json.loads((ROOT/live/'report.json').read_text())['summary']
        if not summary['complete'] or summary['points']!=21:raise ValueError('Unfinished all21 inventory')
        merged='build/welding_retreat_200cm_merged_inventory_'+args.tag
        stage('welding_merged_inventory_'+args.tag,[PREFIX+'merge_retreat_inventory_v19.py',
            '--input','build/welding_retreat_200cm_same_pose_v55','--input',live,'--out',merged])
        retreat='build/welding_retreat_200cm_'+args.tag
        stage('welding_retreat_200cm_'+args.tag,[PREFIX+'snapshot_mixed_v46.py',
            '--baseline',merged,'--running-out',live,'--out',retreat,'--extra-code',
            PREFIX+'run_transfer_v52.py',PREFIX+'mixed_bridge_v52.py'])
    cold='build/welding_retreat_200cm_'+args.tag+'_cold'
    if args.retreat_cold:
        cold_file=args.retreat_cold
    else:
        stage('welding_retreat_200cm_'+args.tag+'_cold',[
            PREFIX+'verify_v3.py','--retreat-out',retreat,'--out',cold,'--workers','1'])
        cold_file=cold+'/cold_verification.json'
    seed='build/welding_transfer_seeds_'+args.tag+'.json'
    source_tag='same_pose_expanded_v56' if (ROOT/'build/welding_retreat_transfer_200cm_same_pose_expanded_v56_cold/cold_verification.json').exists() else 'compatible_expanded_v53'
    stage('welding_transfer_seeds_'+args.tag,[PREFIX+'snapshot_transfer_seeds_v28.py',
        '--retreat-out',retreat,'--retreat-cold',cold_file,'--source',
        'build/welding_transfer_200cm_'+source_tag,
        'build/welding_retreat_transfer_200cm_'+source_tag+'_cold/cold_verification.json',
        '--out',seed])
    transfer='build/welding_transfer_200cm_'+args.tag
    stage('welding_transfer_200cm_'+args.tag,[PREFIX+'run_transfer_v52.py',
        '--retreat-out',retreat,'--retreat-cold',cold_file,
        '--out',transfer,'--seed-snapshot',seed,'--workers','1','--iterations','2400','--ordering-rounds','5'])
    final='build/welding_retreat_transfer_200cm_'+args.tag+'_cold'
    stage('welding_retreat_transfer_200cm_'+args.tag+'_cold',[
        PREFIX+'verify_v3.py','--retreat-out',retreat,'--transfer-out',transfer,
        '--out',final,'--workers','1'])
    print('FINISHED_BATCH_ONLY',time.monotonic()-started,'full goal status not changed',flush=True)


if __name__=='__main__':main()
