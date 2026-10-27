"""Recover completed source-certified tail edge receipts into a fresh prior graph."""
import argparse
import copy
import json
from pathlib import Path
from experimental.welding_candidate_selection_v1.pool import atomic,digest
from experimental.welding_candidate_selection_v1.run import verify_bindings


def checkpoint(previous,attempt,snapshot,out):
    previous=Path(previous)/'report.json';attempt=Path(attempt);snapshot=Path(snapshot);out=Path(out).resolve()
    if out.exists():raise FileExistsError('Fresh checkpoint output required')
    if not snapshot.is_file():raise ValueError('Archived attempt source snapshot required')
    prior=json.loads(previous.read_text());bindings=dict(prior['manifest']['bindings'])
    # Prior adapter must not silently require replacing the attempted source in place.
    current_tail=Path(__file__).with_name('tail.py').resolve()
    if str(current_tail) in bindings:raise ValueError('Previous report binds live tail.py; archive and rebind explicitly first')
    bindings[str(previous.resolve())]=digest(previous);bindings[str(snapshot.resolve())]=digest(snapshot)
    own=Path(__file__).resolve();bindings[str(own)]=digest(own)
    verify_bindings(bindings)
    records={tuple(row['key']):copy.deepcopy(row) for row in prior['edges']};captured=[]
    for receipt in sorted((attempt/'edges').glob('*.json')):
        row=json.loads(receipt.read_text());key=tuple(row['key'])
        if len(key)!=3 or not -1<=key[0]<=13 or min(key[1:])<0:raise ValueError('Malformed cached edge key')
        if row.get('passed'):
            file=Path(row['file'])
            if digest(file)!=row['sha256']:raise ValueError('Completed edge hash mismatch: '+str(file))
            bindings[str(file.resolve())]=row['sha256']
        # A later failed refresh must not discard a previously certified edge.
        if not row.get('passed') and records.get(key,{}).get('passed'):
            adopted=False
        else:records[key]=row;adopted=True
        bindings[str(receipt.resolve())]=digest(receipt)
        captured.append(dict(key=list(key),receipt=str(receipt.resolve()),receipt_sha256=digest(receipt),passed=bool(row.get('passed')),adopted=adopted))
    verify_bindings({r['file']:r['sha256'] for r in records.values() if r.get('passed')})
    result=copy.deepcopy(prior);result['edges']=list(records.values());result['manifest']['bindings']=bindings
    result['checkpoint_provenance']=dict(previous=str(previous.resolve()),attempt=str(attempt.resolve()),source_snapshot=str(snapshot.resolve()),source_snapshot_sha256=digest(snapshot),captured_receipts=captured,scope='Recovered completed receipts only. Tail optimizer rechecks full endpoints against expanded action pool before planning.')
    out.mkdir();atomic(out/'report.json',result);verify_bindings(bindings)
    return dict(previous=str(out),total_edges=len(records),passed=sum(bool(r.get('passed')) for r in records.values()),captured=len(captured),captured_positive=sum(r['passed'] for r in captured),source_snapshot_sha256=digest(snapshot))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','attempt','snapshot','out'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();print(json.dumps(checkpoint(args.previous,args.attempt,args.snapshot,args.out),indent=2))

if __name__=='__main__':main()
