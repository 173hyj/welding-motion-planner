"""Preserve the complete factory; remove only verified duplicate fixture triangles."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_positioner_v2')]
from geometry import duplicate_source_faces


def main():
    out=ROOT/'build/welding_full_factory_v109'
    if out.exists():raise FileExistsError('Fresh output required')
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    asset=next(a for a in scene['assets'] if a['kind']=='layout')
    fixture=next(a for a in scene['assets'] if a['name']=='ST1-LZ-OP10L')
    paths=[ROOT/'build'/asset['url'],ROOT/'build/welding_positioner_static_v2/DONG.glb',ROOT/'build/welding_positioner_static_v2/JING.glb',Path(__file__)]
    seals={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    layout=trimesh.load(paths[0],force='mesh',process=False)
    layout.apply_transform(np.asarray(asset['transform']).reshape(4,4))
    duplicate=np.zeros(len(layout.faces),bool);audits=[]
    for p in paths[1:3]:
        source=trimesh.load(p,force='mesh',process=False)
        source.apply_transform(np.asarray(fixture['transform']).reshape(4,4))
        mask,audit=duplicate_source_faces(layout,source)
        duplicate|=mask;audits.append(dict(source=str(p.relative_to(ROOT)),**audit))
        print('FULL_FACTORY_DUPLICATES',p.name,audit['duplicate_faces'],flush=True)
    remaining=layout.submesh([np.flatnonzero(~duplicate)],append=True,repair=False)
    if len(remaining.faces)+int(duplicate.sum())!=len(layout.faces):raise ValueError('Triangle conservation failed')
    out.mkdir();remaining.export(out/'factory.glb')
    report=dict(complete=True,full_factory_not_cropped=True,source_triangles=len(layout.faces),
        remaining_background_triangles=len(remaining.faces),duplicate_triangles=int(duplicate.sum()),
        source_sha256=seals,output_sha256=hashlib.sha256((out/'factory.glb').read_bytes()).hexdigest(),
        duplicate_audits=audits,colors_preserved=True,background_transform=np.eye(4).ravel().tolist(),
        scope='Full original factory bounds preserved; only exact tolerance-qualified duplicate fixture faces removed. DONG/JING original instances remain separately rendered.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('FULL_FACTORY_READY',report['remaining_background_triangles'],flush=True)


if __name__=='__main__':main()
