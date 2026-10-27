import struct
import unittest
import numpy as np
import trimesh

from materials import split_mesh_by_color, pack_msh
from scene import world_pose, source_matrix
from timeline import pair_at


class MaterialTests(unittest.TestCase):
    def test_exact_two_colors_no_triangles_lost(self):
        v=np.array([[0,0,0],[1000,0,0],[0,1000,0],[0,0,1000],[1000,0,1000],[0,1000,1000]],dtype=float)
        m=trimesh.Trimesh(v,[[0,1,2],[3,4,5]],process=False)
        m.visual.vertex_colors=np.array([[255,255,0,255]]*3+[[0,255,255,128]]*3,dtype=np.uint8)
        parts=split_mesh_by_color(m)
        self.assertEqual(sum(len(x['faces']) for x in parts),2)
        self.assertEqual({tuple(x['rgba']) for x in parts},{(255,255,0,255),(0,255,255,128)})
        for p in parts:
            self.assertLessEqual(p['vertices'].max(),1)
            blob=pack_msh(p['vertices'],p['faces'])
            nv,nn,nt,nf=struct.unpack('<4i',blob[:16]);self.assertEqual((nv,nn,nt,nf),(3,0,0,1))
            self.assertEqual(len(blob),16+nv*12+nf*12)

    def test_mixed_vertex_color_face_is_not_silently_averaged(self):
        m=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],process=False)
        m.visual.vertex_colors=[[255,0,0,255],[0,255,0,255],[0,0,255,255]]
        with self.assertRaisesRegex(ValueError,'mixed'):split_mesh_by_color(m)


class PoseTests(unittest.TestCase):
    def test_mm_y_up_conversion_preserves_rotation_and_translation(self):
        t=np.eye(4);t[:3,3]=[2000,3000,4000]
        result=source_matrix(t.ravel().tolist())
        np.testing.assert_allclose(result[:3,3],[2,-4,3],atol=1e-12)
        pos,quat=world_pose(t.ravel().tolist());np.testing.assert_allclose(pos,[2,-4,3])
        from scipy.spatial.transform import Rotation
        np.testing.assert_allclose(Rotation.from_quat(np.roll(quat,-1)).as_matrix(),result[:3,:3],atol=1e-12)

    def test_equal_time_seam_selects_new_phase(self):
        times=np.array([0.,1.,1.,2.]);i,j,a=pair_at(times,1.)
        self.assertEqual((i,j,a),(2,3,0.))
        self.assertEqual(pair_at(times,2.),(3,3,0.))
        self.assertEqual(pair_at(times,.5),(0,1,.5))


if __name__=='__main__':unittest.main()
