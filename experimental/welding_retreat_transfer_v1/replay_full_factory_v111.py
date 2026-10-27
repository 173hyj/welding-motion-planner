"""Full original factory background with cold-certified source-FK welding playback."""
import hashlib
import json
import numpy as np
from pathlib import Path
import replay_full21_v110 as base


class FullFactoryScene(base.MovingScene):
    def __init__(self,source,include_layout=True):
        super().__init__(source,True)
        self.scene['assets']=[a for a in self.scene['assets'] if a['kind']!='layout']
        factory=dict(next(a for a in source['assets'] if a['kind']=='layout'))
        factory.update(url='welding_full_factory_v109/factory.glb',transform=np.eye(4).ravel().tolist())
        self.scene['assets'].append(factory)

    def mesh(self,url):
        result=super().mesh(url)
        for mesh in self.asset_xml.findall('mesh'):mesh.set('inertia','shell')
        return result


def main():
    folder=base.ROOT/'build/welding_full21_replay_v105'
    audit=json.loads((folder/'renderer_v111.json').read_text())
    for name,value in audit['code_sha256'].items():
        if hashlib.sha256((base.ROOT/name).read_bytes()).hexdigest()!=value:raise ValueError('Renderer source changed '+name)
    report=base.ROOT/'build/welding_full_factory_v109/report.json'
    info=json.loads(report.read_text())
    if not info['complete'] or not info['full_factory_not_cropped']:raise ValueError('Full factory required')
    if hashlib.sha256(report.read_bytes()).hexdigest()!=audit['factory_report_sha256']:raise ValueError('Factory report changed')
    path=report.parent/'factory.glb'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info['output_sha256']:raise ValueError('Factory geometry changed')
    base.MovingScene=FullFactoryScene
    base.main()


if __name__=='__main__':main()
