import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SceneTests(unittest.TestCase):
    def test_fixture_world_pose_includes_parent_assembly(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from build_scene import assemble
        from import_project import xml_root
        import numpy as np
        ps=xml_root((ROOT/'source/psz/StandaloneStudy_PsState.xml').read_bytes())
        objects={o.get('ExternalId'):o for o in ps.find('Objects')}
        scene=assemble(export=False)
        for asset in scene['assets']:
            layout=objects[objects[asset['id']].findtext('layout')]
            absolute=layout.find('NodeInfo/absoluteLocation')
            expected=[float(absolute.findtext(k)) for k in ('x','y','z')]
            actual=np.array(asset['transform']).reshape(4,4)[:3,3]
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0,
                                       err_msg=f"missing hierarchy for {asset['name']}")

    def test_source_coordinate_registration_within_csv_precision(self):
        self.assertTrue((ROOT / "build_scene.py").exists(), "scene export not implemented")
        import sys
        sys.path.insert(0, str(ROOT))
        from build_scene import assemble
        scene = assemble(export=False)
        self.assertEqual(len(scene["welds"]), 29)
        self.assertLess(max(w["registration_error_mm"] for w in scene["welds"]), 0.1)
        self.assertEqual(sum(a["kind"] == "robot" for a in scene["assets"]), 2)


if __name__ == "__main__":
    unittest.main()
