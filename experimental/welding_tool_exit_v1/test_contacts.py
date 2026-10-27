import unittest
import numpy as np
try:
    from tool_world import patch_certificate, closed_contact_passed
except ModuleNotFoundError:
    patch_certificate = closed_contact_passed = None


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(patch_certificate, 'source tool contact validation not implemented')
        self.vertices = np.array([[0,0,0],[1,0,0],[0,1,0]], float)
        self.faces = np.array([[0,1,2]])
        self.contacts = [dict(moving_face=0,product_face=0,penetration_depth_mm=.2,
                              contact_xyz_mm=[0,0,0])]
        self.product = type('Mesh', (), {'vertices': self.vertices, 'faces': self.faces})()

    def certify(self, **overrides):
        args = dict(vertices=self.vertices, faces=self.faces, transform=np.eye(4),
                    origin=np.zeros(3), contacts=self.contacts, product_mesh=self.product,
                    truncated=False, allowed_planes=[dict(normal=[0,0,1], offset_mm=0.)])
        args.update(overrides)
        return patch_certificate(**args)

    def test_local_plane_contact_passes(self):
        self.assertTrue(self.certify()['passed'])

    def test_contact_full_source_triangle_outside_patch_rejected(self):
        vertices = self.vertices.copy(); vertices[2,1] = 26.
        self.assertFalse(self.certify(vertices=vertices)['passed'])

    def test_new_off_plane_product_contact_rejected(self):
        self.assertFalse(self.certify(allowed_planes=[dict(normal=[1,0,0],offset_mm=0.)])['passed'])

    def test_excessive_depth_or_truncated_contacts_rejected(self):
        contacts = [dict(self.contacts[0], penetration_depth_mm=2.01)]
        self.assertFalse(self.certify(contacts=contacts)['passed'])
        self.assertFalse(self.certify(truncated=True)['passed'])

    def test_both_electrodes_must_touch_designated_product(self):
        good = dict(blocking_pairs=[],certificates={'a':{'passed':True}},electrode_gaps_mm={'b':2.})
        self.assertFalse(closed_contact_passed(good, ['a','b']))
        good['electrode_gaps_mm']['b'] = .01
        self.assertTrue(closed_contact_passed(good, ['a','b']))


if __name__ == '__main__': unittest.main()
