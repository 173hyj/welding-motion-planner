import unittest
try:
    from run_arm import qualify_full_pairs
except ModuleNotFoundError:
    qualify_full_pairs = None


class FullPairTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(qualify_full_pairs,'complete arm/tool collision qualification missing')
        self.cert = dict(blocking_pairs=[],certificates={'cap':{'passed':True}})

    def test_only_exact_active_electrode_product_pair_is_removed(self):
        intended = dict(checked_robot_id='r',link='gun:cap',asset_id='p')
        body = dict(checked_robot_id='r',link='gun:body',asset_id='p')
        other = dict(checked_robot_id='other',link='gun:cap',asset_id='p')
        fixed = dict(checked_robot_id='r',link='gun:cap',asset_id='fixture')
        self.assertEqual(qualify_full_pairs([intended,body,other,fixed],'r','p',self.cert,'opening'),[body,other,fixed])

    def test_no_contact_permission_after_separation(self):
        pair = dict(checked_robot_id='r',link='gun:cap',asset_id='p')
        self.assertEqual(qualify_full_pairs([pair],'r','p',self.cert,'separated'),[pair])


if __name__ == '__main__': unittest.main()
