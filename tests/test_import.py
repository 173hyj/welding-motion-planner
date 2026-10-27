import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ImportTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "import_project.py"
        self.assertTrue(path.exists(), "source importer has not been implemented")
        spec = importlib.util.spec_from_file_location("import_project", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_rejects_traversal_and_drive_names(self):
        for name in ("../escape", "EngData\\..\\escape", "/escape", "C:\\escape"):
            with self.assertRaises(ValueError):
                self.module.safe_name(name)

    def test_normalizes_nested_zip_windows_separators(self):
        self.assertEqual(self.module.safe_name("EngData\\robot\\link.jt"), "EngData/robot/link.jt")

    def test_decodes_non_utf8_zip_names(self):
        name = "上航发/AP31-LZ".encode("gbk").decode("cp437")
        self.assertEqual(self.module.decoded_name(name, 0), "上航发/AP31-LZ")

    def test_parses_declared_chinese_xml_encoding(self):
        data = '<?xml version="1.0" encoding="gb2312"?><root><name>机器人</name></root>'.encode("gb2312")
        self.assertEqual(self.module.xml_root(data).findtext("name"), "机器人")


if __name__ == "__main__":
    unittest.main()
