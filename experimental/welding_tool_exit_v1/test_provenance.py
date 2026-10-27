import unittest
from pathlib import Path
import r2_safe_park
from run_initial import extended_source_hashes,ROOT


class ProvenanceTests(unittest.TestCase):
    def test_guard_hashes_actual_collision_dependency_and_source_meshes(self):
        try:
            hashes=extended_source_hashes()
        except FileNotFoundError as exc:
            self.fail('Evidence guard targets a nonexistent input: '+str(exc))
        self.assertIn(str(Path(r2_safe_park.__file__).relative_to(ROOT)),hashes)
        self.assertTrue(any(k.endswith('.glb') and 'robot' in k for k in hashes))
        self.assertIn('collision_audit.py',hashes)


if __name__=='__main__':unittest.main()
