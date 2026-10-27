"""Fresh short generation, source seal verified before publication."""
import json
from pathlib import Path
import sys
from . import run

def main():
 from run_initial import extended_source_hashes,assert_frozen_geometry_cache
 assert_frozen_geometry_cache();before=extended_source_hashes()
 out=Path(sys.argv[sys.argv.index('--out')+1])
 if out.exists():raise FileExistsError('Fresh sealed directory required')
 out.mkdir(parents=True);run.atomic(out/'source_before.json',dict(source_sha256=before,geometry_cache_asserted=True))
 run.main()
 assert_frozen_geometry_cache();after=extended_source_hashes()
 if before!=after:raise ValueError('Source changed during generation')
 report=json.loads((out/'generation_report.json').read_text());bindings=report['manifest']['bindings']
 for name,sha in bindings.items():
  if run.digest(name)!=sha:raise ValueError('Inherited binding changed '+name)
 for name,sha in before.items():
  name=str(run.ROOT/name)
  if name in bindings and bindings[name]!=sha:raise ValueError('Conflicting frozen binding '+name)
  bindings[name]=sha
 report['source_assets_frozen_before_and_after']=True
 report['accepted']=len(report['rows'])==42 and all(r['passed'] for r in report['rows'])
 report['approved_lengths_mm']=[200,300]
 run.atomic(out/'generation_report.json',report);run.atomic(out/'source_after.json',dict(source_sha256=after,unchanged=True))
 print('SEALED_GENERATION_COMPLETE',report['accepted'],len(report['rows']),flush=True)
if __name__=='__main__':main()
