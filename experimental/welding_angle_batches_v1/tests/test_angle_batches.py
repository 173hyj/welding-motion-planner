import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).absolute().parents[1] / 'grouping.py'


class AngleBatchTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), 'angle grouping implementation is missing')
        spec = importlib.util.spec_from_file_location('angle_grouping', PATH)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)

    def test_all_welds_at_one_stop(self):
        available = {a: {'w1', 'w2', 'w3'} for a in range(65, 116, 5)}
        plans = self.api.propose_batches(available)
        self.assertEqual(len(plans), 11)
        self.assertEqual(plans[0]['angles'], (65.0,))
        self.assertEqual(plans[0]['groups'], [{'angle': 65.0, 'weld_ids': ['w1', 'w2', 'w3']}])
        self.assertEqual(plans[0]['stops'], 1)

    def test_enumerates_all_minimum_covers(self):
        covers = self.api.minimal_angle_covers({0: {'a'}, 10: {'b'}, 20: {'a'}, 30: {'b'}})
        self.assertEqual(covers, [(0.0, 10.0), (0.0, 30.0), (10.0, 20.0), (20.0, 30.0)])

    def test_missing_required_weld_has_no_plan(self):
        self.assertEqual(self.api.propose_batches({0: {'a'}}, required={'a', 'b'}), [])

    def test_angles_do_not_wrap(self):
        result = self.api.evaluate_plan([{'angle': 180, 'weld_ids': ['a']}], start_angle=-180)
        self.assertEqual(result['total_travel_deg'], 360)
        self.assertEqual(result['rotations_count'], 1)

    def test_route_is_shortest_and_deterministic(self):
        plans = self.api.propose_batches({-20: {'a'}, 10: {'b'}, 30: {'c'}}, start_angle=0, stop_penalty_s=2)
        self.assertEqual(plans[0]['angles'], (-20.0, 10.0, 30.0))
        self.assertEqual(plans[0]['total_travel_deg'], 70)
        self.assertEqual(plans[0]['stop_penalty_s'], 6)

    def test_starting_at_stop_is_not_a_rotation(self):
        plan = self.api.propose_batches({0: {'a'}, 10: {'b'}})[0]
        self.assertEqual(plan['rotations_count'], 1)

    def test_empty_required_is_empty_plan(self):
        self.assertEqual(self.api.minimal_angle_covers({0: {'a'}}, required=set()), [()])

    def test_enumeration_limit_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.api.minimal_angle_covers({0: {'a'}, 1: {'a'}}, max_covers=1)


if __name__ == '__main__':
    unittest.main()
