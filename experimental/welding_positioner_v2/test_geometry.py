import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).parent))
from geometry import duplicate_source_faces, source_face_group_mask


class GeometryTests(unittest.TestCase):
    def test_only_matching_triangles_removed_even_when_centers_match(self):
        original = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
        different = np.array([[-1., 0., 0.], [4., 0., 0.], [0., 3., 0.]])
        vertices = np.vstack([original[[2, 0, 1]] + .001, different, original + [0, 0, .05]])
        layout = trimesh.Trimesh(vertices=vertices, faces=np.arange(9).reshape(-1, 3), process=False)
        source = trimesh.Trimesh(vertices=original, faces=[[0, 1, 2]], process=False)
        mask, audit = duplicate_source_faces(layout, source, tolerance_mm=.02)
        np.testing.assert_array_equal(mask, [True, False, False])
        self.assertEqual(audit['duplicate_faces'], 1)
        self.assertLess(audit['maximum_vertex_difference_mm'], .02)

    def test_rigid_motion_preserves_duplicate_decisions(self):
        mesh = trimesh.creation.box()
        moved = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [17, 33, 50], degrees=True).as_matrix()
        transform[:3, 3] = [17000, -3000, -12000]
        moved.apply_transform(transform)
        mask, audit = duplicate_source_faces(moved, moved.copy(), tolerance_mm=.02)
        self.assertTrue(mask.all())
        self.assertEqual(audit['duplicate_faces'], len(mesh.faces))

    def test_same_centroid_different_source_triangle_does_not_hide_match(self):
        wrong = np.array([[-1., 0., 0.], [4., 0., 0.], [0., 3., 0.]])
        right = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
        source = trimesh.Trimesh(vertices=np.vstack([wrong, right]), faces=[[0, 1, 2], [3, 4, 5]], process=False)
        layout = trimesh.Trimesh(vertices=right, faces=[[0, 1, 2]], process=False)
        mask, _ = duplicate_source_faces(layout, source, tolerance_mm=.02)
        self.assertTrue(mask[0])

    def test_only_declared_instance_occurrence_assigned_to_interface(self):
        raw = dict(nodes=[
            dict(id=0, children=[1, 2], meshes=[], type='PartitionNodeData'),
            dict(id=1, children=[3], meshes=[], type='InstanceNodeData'),
            dict(id=2, children=[3], meshes=[], type='InstanceNodeData'),
            dict(id=3, children=[], meshes=[0], type='TriStripSetShapeNodeData')],
            meshes=[dict(indices=[0, 1, 2, 0, 0, 1])])
        np.testing.assert_array_equal(source_face_group_mask(raw, {1}), [True, False])


if __name__ == '__main__':
    unittest.main()
