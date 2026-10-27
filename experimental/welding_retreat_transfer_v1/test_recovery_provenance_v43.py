import importlib
import unittest


class RecoveryProvenanceTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('recovery_provenance_v43')
        except ModuleNotFoundError:
            self.fail('Explicit dependency-loss recovery provenance missing')

    def test_loss_is_kept_as_explicit_audit_and_never_passed(self):
        m = self.module()
        result = m.classify_seals({'source.xml':'one','old/report.json':'two'},
                                  lambda p: 'one' if p=='source.xml' else None)
        self.assertEqual(result['matched'],{'source.xml':'one'})
        self.assertEqual(result['missing'],{'old/report.json':'two'})
        self.assertFalse(result['historical_chain_intact'])
        self.assertTrue(result['requires_fresh_cold'])

    def test_changed_existing_source_refuses_recovery(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.classify_seals({'source.xml':'one'},lambda p:'different')

    def test_all_original_inventory_identities_and_distance_required(self):
        m = self.module()
        rows = [dict(weld_id=str(i),best=None) for i in range(21)]
        m.validate_inventory(rows,2000.)
        with self.assertRaises(ValueError):
            m.validate_inventory(rows,200.)
        with self.assertRaises(ValueError):
            m.validate_inventory(rows[:-1],2000.)
        rows[-1]['weld_id']='0'
        with self.assertRaises(ValueError):
            m.validate_inventory(rows,2000.)


if __name__=='__main__':
    unittest.main()
