import json
import tempfile
import unittest
from pathlib import Path
try:
    from run_safe import cached_result
except ModuleNotFoundError:
    cached_result=None


class SafeEntryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cached_result,'safe completed-run read protection missing')

    def test_completed_resume_keeps_original_timing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            path.write_text(json.dumps(dict(manifest={'source_sha256':{'input':'hash'},'code_sha256':{}},
                summary={'complete':True,'wall_s':66.727})))
            before=path.read_bytes()
            result=cached_result(path,{'input':'hash'})
            self.assertEqual(result['wall_s'],66.727)
            self.assertEqual(path.read_bytes(),before)

    def test_changed_input_cannot_reuse_completed_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            path.write_text(json.dumps(dict(manifest={'source_sha256':{'input':'old'},'code_sha256':{}},
                summary={'complete':True,'passed_points':21})))
            before=path.read_bytes()
            with self.assertRaises(RuntimeError):cached_result(path,{'input':'new'})
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
