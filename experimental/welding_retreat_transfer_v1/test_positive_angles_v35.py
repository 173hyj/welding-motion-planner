import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class PositiveAngleTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('positive_angles_v35')
        except ModuleNotFoundError:
            self.fail('positive-angle expansion and sealed negative reuse are missing')

    def test_nearby_angles_first_and_duplicates_removed(self):
        m = self.module()
        self.assertEqual(m.angle_sequence(120., [0., 120., 150., 90., 120., -120.]),
                         [120., 150., 90., 0., -120.])
        with self.assertRaises(ValueError):
            m.angle_sequence(0., [float('nan')])

    def test_skip_sealed_failed_angle_and_stop_at_first_positive(self):
        m = self.module()
        calls = []
        def solve(angle):
            calls.append(angle)
            return dict(best={'length_mm': 2000.} if angle == 150. else None)
        best, records = m.search_until_positive('w1', [120., 90., 150., 0.],
                        {('w1', 120.): 'old.json'}, solve)
        self.assertEqual(calls, [90., 150.])
        self.assertEqual(best['best']['length_mm'], 2000.)
        self.assertEqual(records[0]['skipped_certified_negative'], 'old.json')
        self.assertEqual(len(records), 3)

    def evidence(self, base):
        root = Path(base)
        out = root/'old'; out.mkdir()
        (root/'core.py').write_text('original core')
        (root/'source.mesh').write_text('original mesh')
        (out/'point.json').write_text(json.dumps(dict(weld_id='w1', best=None, initial={'positioner_deg':120.},
                    passed=False, required_straight_distance_mm=2000.)))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        codes = {'core.py': sha(root/'core.py')}
        report = dict(manifest=dict(source_sha256={'source.mesh': sha(root/'source.mesh')},
                code_sha256=codes, settings=dict(distance_mm=2000., posture_roll_step_deg=5.,
                    source_wrist_search_low_rad=1e-6, both_robot_assignment=True,
                    full_direction_bank=True, no_midpath_wrap=True, cold_uses_full_source_limits=True)),
                summary=dict(complete=True),
                search_records=[dict(weld_id='w1', positioner_deg=120.,
                    new_positive_found=False, report_file='point.json')],
                output_sha256={'point.json': sha(out/'point.json')})
        (out/'report.json').write_text(json.dumps(report))
        return root, out, codes, report

    def test_only_completed_matching_sealed_failure_can_skip(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as base:
            root, out, codes, report = self.evidence(base)
            failures = m.read_certified_failures(out, root, codes)
            self.assertEqual(set(failures), {('w1', 120.)})
            report['summary']['complete'] = False
            (out/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                m.read_certified_failures(out, root, codes)

    def test_tampered_result_or_source_cannot_skip(self):
        m = self.module()
        for target in ['old/point.json', 'source.mesh', 'core.py']:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as base:
                root, out, codes, _ = self.evidence(base)
                (root/target).write_text('tampered')
                with self.assertRaises(ValueError):
                    m.read_certified_failures(out, root, codes)

    def test_different_search_domain_cannot_skip(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as base:
            root, out, codes, report = self.evidence(base)
            report['manifest']['settings']['full_direction_bank'] = False
            (out/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                m.read_certified_failures(out, root, codes)

    def test_positive_record_cannot_be_reused_as_negative(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as base:
            root, out, codes, report = self.evidence(base)
            report['search_records'][0]['new_positive_found'] = True
            (out/'report.json').write_text(json.dumps(report))
            self.assertEqual(m.read_certified_failures(out, root, codes), {})

    def test_negative_angle_identity_must_match_result(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as base:
            root, out, codes, report = self.evidence(base)
            report['search_records'][0]['positioner_deg'] = 150.
            (out/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                m.read_certified_failures(out, root, codes)

    def test_layout_rejection_still_gets_complete_distance_identity(self):
        m = self.module()
        result = dict(weld_id='w1', best=None, passed=False,
                      reason='positioner_layout_blocked', initial={'positioner_deg':90.})
        sealed = m.normalize_result(result, 'w1', 90.)
        self.assertEqual(sealed['required_straight_distance_mm'], 2000.)
        with self.assertRaises(ValueError):
            m.normalize_result(result, 'w1', 120.)


if __name__ == '__main__':
    unittest.main()
