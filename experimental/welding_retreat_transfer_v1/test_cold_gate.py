import unittest
import run_transfer


class ColdGateTests(unittest.TestCase):
    def test_failed_retreat_not_a_route_node(self):
        rows = [dict(weld_id='a', best=dict(path_sha256='hash-a')),
                dict(weld_id='b', best=dict(path_sha256='hash-b'))]
        cold = dict(retreat_report_sha256='report', retreat_rows=[
            dict(weld_id='a', passed=True, planned_path_sha256='hash-a'),
            dict(weld_id='b', passed=False, planned_path_sha256='hash-b')])
        self.assertEqual(run_transfer.cold_retreat_ids(rows, cold, 'report'), {'a'})

    def test_wrong_report_or_path_identity_rejected(self):
        rows = [dict(weld_id='a', best=dict(path_sha256='hash-a'))]
        cold = dict(retreat_report_sha256='report', retreat_rows=[
            dict(weld_id='a', passed=True, planned_path_sha256='hash-wrong')])
        with self.assertRaises(RuntimeError):
            run_transfer.cold_retreat_ids(rows, cold, 'report')
        with self.assertRaises(RuntimeError):
            run_transfer.cold_retreat_ids(rows, cold, 'other-report')


if __name__ == '__main__':
    unittest.main()
