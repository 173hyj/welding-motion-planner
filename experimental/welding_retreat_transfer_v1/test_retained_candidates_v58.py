import importlib
import json
import hashlib
import tempfile
import unittest
from pathlib import Path


class RetainedCandidateTests(unittest.TestCase):
    def setUp(self):
        try:self.module=importlib.import_module('retained_candidates_v58')
        except ModuleNotFoundError:self.fail('Original candidate propagation missing')

    def test_retained_original_keeps_exact_bytes_and_valid_local_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';out=root/'out';source.mkdir()
            raw=json.dumps(dict(weld_id='w',passed=True,length_mm=2000.,fixed_distance_mm=2000.)).encode()
            (source/'old.json').write_bytes(raw)
            candidate=dict(path_file='old.json',path_sha256=hashlib.sha256(raw).hexdigest(),driver_mm=-10.)
            result=self.module.copy_candidate('w',candidate,source,out,'paths/retained_00.json')
            self.assertEqual((out/result['path_file']).read_bytes(),raw)
            self.assertEqual(result['path_sha256'],candidate['path_sha256'])
            self.assertEqual(candidate['path_file'],'old.json')
            (source/'old.json').write_bytes(raw+b' ')
            with self.assertRaises(ValueError):self.module.copy_candidate('w',candidate,source,out,'paths/bad.json')

    def test_snapshot_defer_does_not_replace_original_by_selected_candidate(self):
        try:m=importlib.import_module('snapshot_mixed_v59')
        except ModuleNotFoundError:self.fail('Retention-aware snapshot missing')
        row=dict(best={'path_file':'selected.json'},retained_existing_candidate={'path_file':'original.json'})
        result=m.defer_row(row)
        self.assertEqual(result['retained_existing_candidate']['path_file'],'original.json')
        self.assertIsNone(result['best'])


if __name__=='__main__':unittest.main()
