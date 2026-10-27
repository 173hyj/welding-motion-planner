import unittest
from experimental.welding_retreat_500mm_v1.sequential_checked import require_generation,merge_bindings
class SealTest(unittest.TestCase):
    def test_unsealed_all42_cannot_pass(self):
        rows=[dict(passed=True,length_mm=d,task=t) for d in (500.,1000.) for t in range(1,22)]
        with self.assertRaises(ValueError):require_generation(dict(complete=True,rows=rows))
    def test_conflicting_binding_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):merge_bindings({'geometry':'old'},{'geometry':'new'})
if __name__=='__main__':unittest.main()
