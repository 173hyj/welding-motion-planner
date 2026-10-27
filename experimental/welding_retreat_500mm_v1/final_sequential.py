"""Final runner rejects provisional generation reports before any source work."""
import json
from pathlib import Path
import sys
from . import sequential_checked as sequential

def require_seal(report):
    if report.get('source_assets_frozen_before_and_after') is not True or report.get('accepted') is not True:
        raise ValueError('Completed before/after asset seal required; provisional reports forbidden')
    return sequential.require_generation(report)

def main():
    generation=Path(sys.argv[sys.argv.index('--generation')+1])
    report=json.loads(generation.read_text());require_seal(report)
    before=json.loads((generation.parent/'source_before.json').read_text())['source_sha256']
    after=json.loads((generation.parent/'source_after.json').read_text())
    if after.get('unchanged') is not True or before!=after['source_sha256']:raise ValueError('Before/after source receipt mismatch')
    from .run import ROOT,digest
    for name,sha in before.items():
        if report['manifest']['bindings'].get(str(ROOT/name))!=sha or digest(ROOT/name)!=sha:raise ValueError('Source receipt binding mismatch '+name)
    original=sequential.require_generation
    def guarded(report):
        if report.get('source_assets_frozen_before_and_after') is not True or report.get('accepted') is not True:raise ValueError('Unsealed generation')
        return original(report)
    sequential.require_generation=guarded
    sequential.main()
if __name__=='__main__':main()
