"""Preserve old loss audit, bind surviving source assets, independently reverify selected paths."""
import argparse
import hashlib
import json
from pathlib import Path
import mmap
import ijson
from recovery_provenance_v43 import classify_seals,validate_inventory

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')));tmp.replace(path)


def field(path,key):
    with path.open('rb') as f:
        return next(ijson.items(f,key,use_float=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();source,out=args.source.resolve(),args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh recovery evidence required')
    report=source/'report.json';manifest=field(report,'manifest');summary=field(report,'summary')
    if summary.get('complete') is not True:
        raise ValueError('Completed original report required')
    # These generated reports place all original rows before the huge duplicated
    # search_records list. Standard json preserves Python's archived Infinity
    # diagnostics, which ijson correctly refuses as non-standard JSON.
    with report.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        start=data.find(b'"rows":')
        end=data.find(b',"search_records":',start)
        if start<0 or end<0:
            raise ValueError('Expected archived rows before duplicated search_records')
        rows=json.loads(data[start+len(b'"rows":'):end])
    validate_inventory(rows,summary['required_straight_distance_mm'])
    seals={**manifest['source_sha256'],**manifest['code_sha256']}
    audit=classify_seals(seals,lambda n:digest(ROOT/n) if (ROOT/n).is_file() else None)
    # Runtime assets cannot be missing. Lost historical result chains are
    # disclosed below and do not provide evidence for this new verification.
    for n in audit['missing']:
        if not n.endswith('.json'):
            raise ValueError('Missing runtime code/asset requires exact recovery: '+n)
    out.mkdir(parents=True)
    audit.update(original_report=str(report.relative_to(ROOT)),original_report_sha256=digest(report),
       reason='Accidental cleanup removed inherited historical result dependencies. '
              'Old manifests left unchanged; current source assets match their original hashes. '
              'This is a new independent trajectory verification chain, not restoration of old certificates.',
       prior_negative_results_not_independently_reverified=True,
       prior_timing_not_recertified=True,prior_cold_not_reused=True)
    atomic(out/'dependency_loss_audit.json',audit)
    sources=dict(audit['matched'])
    sources[str(report.relative_to(ROOT))]=audit['original_report_sha256']
    sources[str((out/'dependency_loss_audit.json').relative_to(ROOT))]=digest(out/'dependency_loss_audit.json')
    outputs={}
    for i,row in enumerate(rows):
        if not row['best']:
            row['negative_evidence_historical_chain_incomplete']=True
            continue
        path=source/row['best']['path_file'];sha=digest(path)
        if sha!=row['best']['path_sha256']:
            raise ValueError('Retained positive bytes changed')
        trajectory=json.loads(path.read_text())
        if (not trajectory['passed'] or trajectory['length_mm']!=2000.
                or trajectory['q_start_rad']!=row['initial']['q_rad']
                or trajectory['other_q_rad']!=row['initial']['other_q_rad']):
            raise ValueError('Positive path distance/state identity mismatch')
        sources[str(path.relative_to(ROOT))]=sha
        name=f'paths/revalidation_retreat_{i:02d}.json';dest=out/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        if digest(dest)!=sha:
            raise ValueError('Path copy differs')
        outputs[name]=sha
        row['best'].update(path_file=name,path_sha256=sha)
        row['fresh_cold_required_after_dependency_loss']=True
    codes={str(p.relative_to(ROOT)):digest(p) for p in
           [Path(__file__),Path(__file__).with_name('recovery_provenance_v43.py')]}
    for n,s in {**sources,**codes}.items():
        if digest(ROOT/n)!=s:
            raise ValueError('Source changed during new chain preparation')
    manifest=dict(source_sha256=sources,code_sha256=codes,settings=dict(distance_mm=2000.,
       historical_chain_intact=False,dependency_loss_disclosed=True,require_independent_fresh_cold=True))
    atomic(out/'manifest.json',manifest)
    summary=dict(complete=True,inventory_prepared=True,points=21,passed=sum(bool(r['best']) for r in rows),
          required_straight_distance_mm=2000.,fresh_cold_passed=False,historical_chain_intact=False,wall_s=0.)
    atomic(out/'report.json',dict(manifest=manifest,summary=summary,rows=rows,output_sha256=outputs,
          scope='New source-matched independent verification inventory after disclosed historical '
                'dependency loss. Does not restore previous certificates, search timing or negative '
                'evidence. Preserved positive path bytes must pass new independent full-source cold '
                'and fresh endpoint transfer planning. Full original 21-weld identities retained.'))
    print('NEW_EVIDENCE_INVENTORY',summary,flush=True)


if __name__=='__main__':
    main()
