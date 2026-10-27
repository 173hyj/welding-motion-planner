import copy
import importlib
import unittest


class InventoryReselectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('probe_mixed_wrist_v65')
        except ModuleNotFoundError:
            self.fail('Retention-aware faster inventory entry is missing')

    def test_new_positive_keeps_original_local_candidate(self):
        original = dict(weld_id='w', best=None,
            retained_existing_candidate=dict(path_file='paths/original.json', path_sha256='original'))
        positive = dict(weld_id='w', best=dict(path_file='paths/new.json', path_sha256='new'), passed=True)
        before = copy.deepcopy(original)
        selected = self.module.replace_with_positive(original, positive)
        self.assertEqual(selected['best'], positive['best'])
        self.assertEqual(selected['retained_existing_candidate'], original['retained_existing_candidate'])
        self.assertEqual(original, before)
        selected['retained_existing_candidate']['path_file'] = 'changed'
        self.assertEqual(original, before)

    def test_failed_or_different_identity_cannot_replace_inventory(self):
        original = dict(weld_id='w', best=None)
        for result in [dict(weld_id='other', best={'path_file': 'p'}, passed=True),
                       dict(weld_id='w', best=None, passed=False)]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.module.replace_with_positive(original, result)


if __name__ == '__main__':
    unittest.main()
