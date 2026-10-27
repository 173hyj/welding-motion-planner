"""Prevent a cold-bound route from being displayed with another scene."""
import hashlib,importlib,json,tempfile,unittest
from pathlib import Path

MODULE='verified_route_loader_v23'


class SceneBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);(self.root/'build').mkdir()
        self.corrected=self.root/'build/corrected.json';self.corrected.write_text('{"world_position":45}')
        self.old=self.root/'build/old.json';self.old.write_text('{"world_position":0}')
        self.metadata=dict(source_scene='build/corrected.json',source_scene_sha256=hashlib.sha256(self.corrected.read_bytes()).hexdigest())
    def resolver(self):
        fn=getattr(importlib.import_module(MODULE),'bound_scene',None)
        self.assertTrue(callable(fn),'Replay scene has no binding validation')
        return fn
    def test_default_resolves_the_exact_cold_bound_scene(self):
        self.assertEqual(self.resolver()(self.root,None,self.metadata),self.corrected.resolve())
    def test_another_scene_is_rejected_before_builder_load(self):
        with self.assertRaisesRegex(ValueError,'scene'):
            self.resolver()(self.root,self.old,self.metadata)
    def test_mutated_bound_scene_is_rejected(self):
        self.corrected.write_text('{"world_position":90}')
        with self.assertRaisesRegex(ValueError,'scene'):
            self.resolver()(self.root,None,self.metadata)


if __name__=='__main__':unittest.main()
