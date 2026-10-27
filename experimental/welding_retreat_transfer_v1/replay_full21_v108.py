"""Visual shell meshes support source thin sheets without changing triangles/colors."""
import hashlib
import json
from pathlib import Path
import replay_full21_v107 as base


class ThinSourceScene(base.MovingScene):
    def mesh(self,url):
        result=super().mesh(url)
        # Visual-only zero-mass geometry: thin source color partitions have
        # no solid volume. This affects inertia calculation, never mesh shape.
        for mesh in self.asset_xml.findall('mesh'):mesh.set('inertia','shell')
        return result


def main():
    path=Path(__file__).resolve()
    evidence=base.ROOT/'build/welding_full21_replay_v105/renderer_v108.json'
    audit=json.loads(evidence.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['renderer_sha256']:
        raise ValueError('Renderer code changed')
    base.MovingScene=ThinSourceScene
    base.main()


if __name__=='__main__':main()
