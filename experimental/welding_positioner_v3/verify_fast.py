"""Run the unchanged independent raw-source verifier on the accelerated result."""
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experimental/welding_positioner_v2'
OUT = ROOT/'build/welding_positioner_static_v3'
sys.path[:0] = [str(BASE), str(ROOT/'experimental/welding_endpoints_v1'), str(ROOT)]


def main():
    started = time.perf_counter()
    # Audit references immutable original/cached geometry paths. Copy its
    # metadata only, never write over prior search or verification evidence.
    shutil.copy2(ROOT/'build/welding_positioner_static_v2/geometry_audit.json', OUT/'geometry_audit.json')
    spec = importlib.util.spec_from_file_location('independent_verification_v2', BASE/'verify.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    verifier.OUT = OUT
    verifier.main()
    result = json.loads((OUT/'selected_independent_verification.json').read_text())
    assert result['all_selected_passed']
    (OUT/'verification_timing.json').write_text(json.dumps(dict(independent_verify_wall_s=time.perf_counter()-started), indent=2))


if __name__ == '__main__': main()
