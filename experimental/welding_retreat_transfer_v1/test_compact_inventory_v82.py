import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
try:
    import compact_inventory_v82 as compact
except ImportError:
    compact = None


class CompactInventoryTests(unittest.TestCase):
    def test_archive_reference_roundtrip_preserves_execution_fields(self):
        self.assertIsNotNone(compact, 'compact inventory feature missing')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = dict(weld_id='w', initial={'q_rad': [1]}, original_initial={'q_rad': [2]},
                       best=None, alternative_pose_trials=[{'large': list(range(1000))}])
            source = root / 'report.json'
            source.write_text(json.dumps({'rows': [row]}))
            before = copy.deepcopy(row)
            rows = compact.compact_rows([row], source, root)
            self.assertEqual(row, before)
            self.assertNotIn('alternative_pose_trials', rows[0])
            self.assertEqual({k:v for k,v in rows[0].items() if k != 'search_archive_reference'},
                             {k:v for k,v in row.items() if k != 'alternative_pose_trials'})
            self.assertEqual(compact.resolve_trials(rows[0], root), row['alternative_pose_trials'])
            source.write_text(source.read_text() + ' ')
            with self.assertRaises(ValueError):
                compact.resolve_trials(rows[0], root)

    def test_rejects_wrong_report_identity(self):
        self.assertIsNotNone(compact, 'compact inventory feature missing')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'report.json'
            source.write_text(json.dumps({'rows': [{'weld_id':'other', 'alternative_pose_trials':[]}]}))
            with self.assertRaises(ValueError):
                compact.compact_rows([{'weld_id':'w','alternative_pose_trials':[]}], source, root)

    def test_exact_local_path_link_and_overwrite_or_changed_source_rejected(self):
        self.assertIsNotNone(compact, 'compact inventory feature missing')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / 'source.json'
            src.write_text(json.dumps(dict(weld_id='w', passed=True, length_mm=2000., fixed_distance_mm=2000.)))
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            candidate = dict(path_file='source.json', path_sha256=sha, length_mm=2000.)
            out = root / 'new'
            result = compact.link_candidate('w', candidate, root, out, 'paths/retained.json')
            self.assertEqual((out/result['path_file']).read_bytes(), src.read_bytes())
            self.assertEqual((out/result['path_file']).stat().st_ino, src.stat().st_ino)
            with self.assertRaises(FileExistsError):
                compact.link_candidate('w', candidate, root, out, 'paths/retained.json')
            with self.assertRaises(ValueError):
                compact.link_candidate('wrong', candidate, root, out, 'paths/wrong.json')
            with self.assertRaises(ValueError):
                compact.link_candidate('w', dict(candidate, path_sha256='bad'), root, out, 'paths/bad.json')


if __name__ == '__main__':
    unittest.main()
