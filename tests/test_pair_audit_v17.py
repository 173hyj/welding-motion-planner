"""Pair coverage contract for the independent policy-complement audit."""
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit_pairs',ROOT/'scripts/audit_collision_exemptions_v17.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class PairAuditTests(unittest.TestCase):
    def setUp(self):
        self.main=[str(k)for k in range(240,246)]
        self.guns=['gun:507_body','gun:507_electrode_9','gun:510_body','gun:510_electrode_1717','gun:508_body','gun:509_body']
        self.keys=['239',*self.main,'246','247',*self.guns]

    def test_every_self_pair_checked_or_audited_exactly_once(self):
        checked={frozenset((a,b))for a,b in combinations(self.main,2)if abs(int(a)-int(b))>1}
        checked|={frozenset((a,b))for a in self.main[:-1]for b in self.guns}
        omitted=[frozenset((a,b))for a,b,_ in audit.omitted_self_pairs(self.keys)]
        all_pairs={frozenset(pair)for pair in combinations(self.keys,2)}
        self.assertEqual(len(omitted),len(set(omitted)))
        self.assertFalse(checked.intersection(omitted))
        self.assertEqual(checked|set(omitted),all_pairs)

    def test_counterweight_and_base_are_covered(self):
        rows={frozenset((a,b)):policy for a,b,policy in audit.omitted_self_pairs(self.keys)}
        self.assertIn(frozenset(('240','246')),rows)
        self.assertIn(frozenset(('239','gun:509_body')),rows)
        self.assertIn(frozenset(('246','gun:510_body')),rows)

    def test_flange_and_gun_internals_covered(self):
        rows={frozenset((a,b)):policy for a,b,policy in audit.omitted_self_pairs(self.keys)}
        for gun in self.guns:self.assertEqual(rows[frozenset(('245',gun))],'flange245_gun_exemption')
        for a,b in combinations(self.guns,2):self.assertEqual(rows[frozenset((a,b))],'internal_gun_pair_exemption')


if __name__=='__main__':unittest.main()
