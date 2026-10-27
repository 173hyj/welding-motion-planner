"""Fresh generation with complete original asset/dependency provenance."""
import json
from pathlib import Path
import sys
from . import run

def main():
    from run_initial import extended_source_hashes,assert_frozen_geometry_cache
    assert_frozen_geometry_cache()
    baseline=extended_source_hashes()
    out=Path(sys.argv[sys.argv.index('--out')+1])
    if out.exists():raise FileExistsError('Fresh sealed generation directory required')
    out.mkdir(parents=True)
    run.atomic(out/'source_before.json',dict(source_sha256=baseline,geometry_cache_asserted=True))
    run.main()
    after=extended_source_hashes()
    if baseline!=after:raise ValueError('Source assets/dependencies changed during generation')
    report=json.loads((out/'generation_report.json').read_text())
    report['manifest']['bindings'].update({str(run.ROOT/name):sha for name,sha in baseline.items()})
    report['source_assets_frozen_before_and_after']=True
    report['accepted']=len(report['rows'])==42 and all(row['passed'] for row in report['rows'])
    run.atomic(out/'generation_report.json',report)
    run.atomic(out/'source_after.json',dict(source_sha256=after,unchanged=True))
    print('SEALED_GENERATION_COMPLETE',report['accepted'],len(report['rows']),flush=True)
if __name__=='__main__':main()
