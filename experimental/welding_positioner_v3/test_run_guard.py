import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_fast import completed_run


class RunGuardTests(unittest.TestCase):
    def test_completed_run_is_detected_without_changing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/'report.json'
            contents = json.dumps(dict(phase='complete', summary=dict(total_passed=21)))
            path.write_text(contents)
            self.assertEqual(completed_run(folder)['summary']['total_passed'], 21)
            self.assertEqual(path.read_text(), contents)

    def test_partial_run_can_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertIsNone(completed_run(folder))
            (folder/'report.json').write_text(json.dumps(dict(phase='stationary')))
            self.assertIsNone(completed_run(folder))


if __name__ == '__main__': unittest.main()
