import copy
import importlib
import unittest

try:
    seed = importlib.import_module('snapshot_transfer_seeds_v28')
except ModuleNotFoundError:
    seed = None


class TransferSeedTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(seed, 'Cold-certified seed remapping is not implemented')
        self.old_nodes = [dict(weld_id='a', state=[0.] * 15), dict(weld_id='b', state=[1.] * 15)]
        self.new_nodes = [dict(weld_id='new', state=[2.] * 15), *copy.deepcopy(self.old_nodes)]
        self.edge = dict(from_index=0, to_index=1, from_weld_id='a', to_weld_id='b', passed=True,
                         path_file='paths/edge.json', path_sha256='bound-path')
        self.report = dict(nodes=self.old_nodes, selected_edges=[self.edge])
        self.cold = dict(transfer_rows=[dict(from_weld_id='a', to_weld_id='b', passed=True,
                                            planned_path_sha256='bound-path')])

    def test_remaps_by_weld_identity_when_new_node_changes_indices(self):
        result = seed.remap_certified_edges(self.new_nodes, self.report, self.cold)
        self.assertEqual([(r['from_index'], r['to_index']) for r in result], [(1, 2)])
        self.assertEqual(result[0]['source_path_sha256'], 'bound-path')

    def test_changed_physical_endpoint_is_not_reused(self):
        self.new_nodes[1]['state'][6] += .01
        self.assertEqual(seed.remap_certified_edges(self.new_nodes, self.report, self.cold), [])

    def test_uncertified_or_different_path_cannot_be_seed(self):
        for alteration in [dict(passed=False), dict(planned_path_sha256='different')]:
            cold = copy.deepcopy(self.cold)
            cold['transfer_rows'][0].update(alteration)
            self.assertEqual(seed.remap_certified_edges(self.new_nodes, self.report, cold), [])

    def test_ambiguous_inventory_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            seed.remap_certified_edges(self.new_nodes + [self.new_nodes[1]], self.report, self.cold)


if __name__ == '__main__':
    unittest.main()
