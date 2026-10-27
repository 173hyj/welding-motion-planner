#!/usr/bin/env python3
"""Produce source-derived display LOD; never replace exact planning meshes."""
import hashlib
import json
from pathlib import Path
import trimesh

ROOT=Path(__file__).resolve().parent


def main():
    target=ROOT/'build/display_meshes';target.mkdir(exist_ok=True)
    records=[]
    for source in sorted((ROOT/'build/meshes').glob('*.glb')):
        scene=trimesh.load(source,force='scene',process=False)
        total=sum(len(mesh.faces) for mesh in scene.geometry.values())
        budget=150000 if source.name.startswith('AP31') else 100000
        if total<=budget:continue
        for key,mesh in list(scene.geometry.items()):
            count=max(20,int(budget*len(mesh.faces)/total))
            if len(mesh.faces)>count:
                simplified=mesh.simplify_quadric_decimation(face_count=count)
                color=getattr(mesh.visual,'main_color',None)
                if color is not None:simplified.visual=trimesh.visual.ColorVisuals(simplified,face_colors=color)
                scene.geometry[key]=simplified
        output=target/source.name;output.write_bytes(scene.export(file_type='glb'))
        records.append({'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'display':str(output.relative_to(ROOT)), 'display_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                        'source_triangles':total,'display_triangles':sum(len(m.faces) for m in scene.geometry.values())})
        print(json.dumps(records[-1]),flush=True)
    (target/'manifest.json').write_text(json.dumps({'scope':'source-derived inspection LOD; full originals retained for planning','meshes':records},indent=2))


if __name__=='__main__':main()
