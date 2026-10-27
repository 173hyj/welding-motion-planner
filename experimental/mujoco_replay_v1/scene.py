"""Full original GLB geometry in a visual-only MuJoCo mocap model."""
import gc
import hashlib
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from materials import pack_msh, read_glb, split_mesh_by_color
from timeline import SOURCE_TO_MUJOCO, value_at_path

ROOT = Path(__file__).resolve().parents[2]


def source_matrix(values):
    source = np.asarray(values, dtype=float).reshape(4, 4)
    c = SOURCE_TO_MUJOCO
    result = np.eye(4)
    result[:3, :3] = c @ source[:3, :3] @ c.T
    result[:3, 3] = c @ source[:3, 3] / 1000.
    return result


def world_pose(values):
    matrix = source_matrix(values)
    return matrix[:3, 3], np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def numbers(values):
    return ' '.join(format(float(x), '.17g') for x in values)


class SceneBuilder:
    def __init__(self, root=ROOT, include_layout=False, scene_path=None):
        self.root = root
        self.include_layout = include_layout
        self.scene_path = Path(scene_path) if scene_path is not None else root / 'build/scene.json'
        if not self.scene_path.is_absolute():self.scene_path=root/self.scene_path
        self.scene = json.loads(self.scene_path.read_text())
        self.xml = ET.Element('mujoco', model='Shanghangfa original GLB visual replay')
        ET.SubElement(self.xml, 'compiler', angle='radian', inertiafromgeom='false', discardvisual='false')
        ET.SubElement(self.xml, 'option', gravity='0 0 0')
        visual = ET.SubElement(self.xml, 'visual')
        ET.SubElement(visual, 'global', offwidth='1440', offheight='900')
        ET.SubElement(visual, 'headlight', ambient='.22 .22 .22', diffuse='.4 .4 .4', specular='.03 .03 .03')
        default = ET.SubElement(self.xml, 'default')
        ET.SubElement(default, 'geom', contype='0', conaffinity='0', group='2', mass='0')
        self.asset_xml = ET.SubElement(self.xml, 'asset')
        ET.SubElement(self.asset_xml, 'texture', type='skybox', builtin='gradient',
                      rgb1='.12 .15 .18', rgb2='.45 .48 .5', width='128', height='768')
        self.world = ET.SubElement(self.xml, 'worldbody')
        ET.SubElement(self.world, 'light', pos='6 8 8', dir='0 0 -1', directional='true', diffuse='.3 .3 .3')
        self.blobs, self.cache, self.materials = {}, {}, {}
        self.mesh_manifest, self.bodies, self.bindings, self.bounds, self.focus_bounds = [], [], [], [], []

    def mesh(self, url):
        if url in self.cache:
            return self.cache[url]
        path = self.root / 'build' / url
        raw, scene = read_glb(path)
        groups, total = [], 0
        for node in scene.graph.nodes_geometry:
            transform, key = scene.graph[node]
            mesh = scene.geometry[key]
            total += len(mesh.faces)
            for part in split_mesh_by_color(mesh):
                vertices = part['vertices'] @ transform[:3, :3].T + transform[:3, 3] / 1000.
                vertices = vertices @ SOURCE_TO_MUJOCO.T
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                rgba = part['rgba']
                # glTF OPAQUE is the default for these original primitives.
                if rgba not in self.materials:
                    name = f'mat_{len(self.materials)}'
                    self.materials[rgba] = name
                    ET.SubElement(self.asset_xml, 'material', name=name,
                                  rgba=numbers([*(np.array(rgba[:3]) / 255.), 1.]), specular='0', shininess='0')
                name = f'mesh_{len(self.blobs)}'
                filename = name + '.msh'
                triangles = vertices[part['faces']]
                # No NORMAL attribute in source GLBs: use glTF flat face normals.
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
                flat_vertices = triangles.reshape(-1, 3)
                faces = np.arange(len(flat_vertices), dtype=np.int32).reshape(-1, 3)
                self.blobs[filename] = pack_msh(flat_vertices, faces, np.repeat(normals, 3, axis=0))
                ET.SubElement(self.asset_xml, 'mesh', name=name, file=filename, maxhullvert='4', inertia='legacy')
                groups.append(dict(name=name, material=self.materials[rgba], rgba=list(rgba),
                                   triangles=len(faces), lo=lo, hi=hi))
        if sum(g['triangles'] for g in groups) != total:
            raise ValueError(f'triangle loss: {url}')
        self.cache[url] = groups
        self.mesh_manifest.append(dict(url=url, sha256=hashlib.sha256(raw).hexdigest(),
                                       source_triangles=total, converted_triangles=total,
                                       colors=[g['rgba'] for g in groups], alpha_mode='OPAQUE',
                                       normals='flat face normals; source GLB has no NORMAL attribute'))
        print(f'  {url}: {total:,} triangles, {len(groups)} colors', flush=True)
        return groups

    def body(self, label, url, matrix, kind, path=None):
        name = f'body_{len(self.bodies)}'
        pos, quat = world_pose(matrix)
        attributes = dict(name=name, pos=numbers(pos), quat=numbers(quat))
        if path:
            attributes['mocap'] = 'true'
        body = ET.SubElement(self.world, 'body', **attributes)
        if path:
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='1', diaginertia='1 1 1')
            self.bindings.append(dict(name=name, label=label, path=path, kind=kind))
        self.bodies.append(dict(name=name, label=label, url=url, source_matrix=np.asarray(matrix).reshape(-1).tolist(), kind=kind))
        matrix = source_matrix(matrix)
        for group in self.mesh(url):
            ET.SubElement(body, 'geom', type='mesh', mesh=group['name'], material=group['material'])
            lo, hi = group['lo'], group['hi']
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            points = corners @ matrix[:3, :3].T + matrix[:3, 3]
            self.bounds.extend(points.tolist())
            if kind in ('robot', 'gun', 'product'):
                self.focus_bounds.extend(points.tolist())

    def compile(self, first_frame, selected_welds=()):
        started = time.perf_counter()
        for asset in self.scene['assets']:
            kind = asset['kind']
            if kind == 'layout' and not self.include_layout:
                continue
            if kind in ('robot', 'gun'):
                for key, part in asset[kind]['link_models'].items():
                    path = ('robots', asset['id'], 'links', key) if kind == 'robot' else ('guns', asset['id'], 'parts', key)
                    self.body(asset['name'] + ':' + key, part['url'], value_at_path(first_frame, path), kind, path)
            else:
                self.body(asset['name'], asset['url'], asset['transform'], kind)
        bounds = np.asarray(self.bounds)
        center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2.
        extent = float(np.linalg.norm(np.ptp(bounds, axis=0)))
        ET.SubElement(self.xml, 'statistic', center=numbers(center), extent=str(extent))
        focus = np.asarray(self.focus_bounds)
        self.camera_center = (focus.min(axis=0) + focus.max(axis=0)) / 2.
        self.camera_extent = float(np.linalg.norm(np.ptp(focus, axis=0)))
        for i, weld in enumerate(self.scene['welds']):
            selected = weld['id'] in selected_welds
            ET.SubElement(self.world, 'site', name=f'weld_{i}',
                          pos=numbers(SOURCE_TO_MUJOCO @ np.array(weld['world_xyz']) / 1000.),
                          size='.006' if selected else '.003',
                          rgba='1 .25 .05 1' if selected else '.6 .6 .6 .5', group='3')
        self.xml_text = ET.tostring(self.xml, encoding='unicode')
        model = mujoco.MjModel.from_xml_string(self.xml_text, assets=self.blobs)
        self.blobs.clear()
        gc.collect()
        for binding in self.bindings:
            binding['body_id'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, binding['name'])
            binding['mocap_id'] = int(model.body_mocapid[binding['body_id']])
        total = sum(m['source_triangles'] for m in self.mesh_manifest)
        if model.nmeshface != total:
            raise ValueError(f'MuJoCo changed mesh triangle count: {model.nmeshface} != {total}')
        if np.any(model.geom_contype) or np.any(model.geom_conaffinity):
            raise ValueError('visual model must not be presented as a collision simulator')
        self.manifest = dict(source_scene=str(self.scene_path),
                             scene_sha256=hashlib.sha256(self.scene_path.read_bytes()).hexdigest(),
                             meshes=self.mesh_manifest, bodies=self.bodies,
                             materials=[dict(source_rgba=list(rgba), mujoco_name=name,
                                             effective_rgba=[*(np.array(rgba[:3]) / 255.).tolist(), 1.])
                                        for rgba, name in self.materials.items()],
                             compiled_meshes=model.nmesh, geoms=model.ngeom,
                             moving_bodies=model.nmocap, unique_source_triangles=total,
                             compiled_unique_triangles=model.nmeshface,
                             instance_triangles=sum(int(model.mesh_facenum[x]) for x in model.geom_dataid if x >= 0),
                             include_layout=self.include_layout, build_seconds=time.perf_counter() - started,
                             placement_corrected=self.scene.get('validation',{}).get('placement_corrected',False),
                             old_collision_results_valid_for_this_scene=self.scene.get('validation',{}).get('old_collision_results_valid_for_this_scene',True),
                             camera_center=self.camera_center.tolist(), camera_extent=self.camera_extent,
                             collision_simulation=False, material_limit='source GLBs have vertex colors only; no original JT PBR/texture data; MuJoCo lighting differs')
        print(f'Compiled {model.nmesh} meshes / {model.nmocap} moving bodies / {total:,} unique triangles', flush=True)
        return model
