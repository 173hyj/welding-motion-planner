import json
import pathlib
import unittest
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PlanTests(unittest.TestCase):
    def test_diagnostic_plan_contains_actual_frames_and_explicit_status(self):
        self.assertTrue((ROOT / "build_plan.py").exists(), "plan generator not implemented")
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "plan.json"
            subprocess.run([".venv/bin/python", "build_plan.py", "--output", str(output)], cwd=ROOT, check=True)
            plan = json.loads(output.read_text())
        self.assertEqual(len(plan["frames"]), 25)
        self.assertTrue(plan["status"]["diagnostic_only"])
        self.assertEqual(len(plan["welds"]), 29)
        self.assertEqual(len(plan["frames"][0]["robots"]), 2)


if __name__ == "__main__":
    unittest.main()
