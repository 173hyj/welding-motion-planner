import copy
import importlib
import unittest


class MixedSnapshotTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('snapshot_mixed_v46')
        except ModuleNotFoundError:
            self.fail('Mixed completed-path snapshot is missing')
        self.signs = {'r0':-1, 'r1':1}
        self.parks = {'r0':[0.,0.,0.,0.,0.,-1.], 'r1':[0.,0.,0.,0.,0.,1.]}
        self.row = dict(weld_id='w', initial=dict(robot_id='r1',other_robot_id='r0',
            q_rad=self.parks['r1'],other_q_rad=[0.,0.,0.,0.,0.,1.],positioner_deg=0.),
            best=dict(path_file='old.json',path_sha256='old',driver_mm=-10.,amplitude_mm=10.))
        self.path = dict(weld_id='w',passed=True,fixed_distance_mm=2000.,length_mm=2000.,
            robot_id='r1',other_robot_id='r0',q_start_rad=self.parks['r1'],
            q_end_rad=self.parks['r1'],other_q_rad=self.parks['r0'],positioner_deg=0.,
            driver_mm=-10.,frames=[dict(q_rad=self.parks['r1'])])

    def test_new_park_is_exact_and_old_input_preserved(self):
        result = self.module.select_row(self.row,self.path,self.signs,self.parks,'new.json','new')
        self.assertEqual(result['initial']['other_q_rad'],self.parks['r0'])
        self.assertEqual(result['original_initial'],self.row['initial'])
        self.assertEqual(result['best']['path_sha256'],'new')
        self.assertEqual(self.row['best']['path_sha256'],'old')

    def test_mixed_midpath_branch_is_rejected(self):
        path=copy.deepcopy(self.path)
        path['frames'].append(dict(q_rad=self.parks['r0']))
        with self.assertRaises(ValueError):
            self.module.select_row(self.row,path,self.signs,self.parks,'new.json','new')

    def test_different_weld_or_short_retreat_is_rejected(self):
        for change in ({'weld_id':'other'},{'length_mm':200.},{'q_start_rad':self.parks['r0']}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.module.select_row(self.row,dict(self.path,**change),self.signs,self.parks,'new.json','new')

    def test_unselected_positive_preserves_candidate_without_failure_claim(self):
        result=self.module.defer_row(self.row)
        self.assertIsNone(result['best'])
        self.assertEqual(result['retained_existing_candidate'],self.row['best'])
        self.assertEqual(result['selection_status'],'awaiting_branch_reselection_not_failure')
        self.assertIsNotNone(self.row['best'])


if __name__=='__main__':
    unittest.main()
