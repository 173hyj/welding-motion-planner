"""Lossless triangle/color partitioning for original GLB visual assets."""
import json
import struct

import numpy as np
import trimesh


def split_mesh_by_color(mesh):
    if mesh.visual.kind != 'vertex':
        raise ValueError('expected explicit source vertex colors')
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[faces]
    if np.any(colors != colors[:, :1, :]):
        raise ValueError('mixed vertex colors within a triangle; cannot use one material')
    values, groups = np.unique(colors[:, 0, :], axis=0, return_inverse=True)
    parts = []
    for i, rgba in enumerate(values):
        used, inverse = np.unique(faces[groups == i].reshape(-1), return_inverse=True)
        parts.append(dict(rgba=tuple(rgba.tolist()), vertices=vertices[used] / 1000.,
                          faces=inverse.reshape(-1, 3).astype(np.int32),
                          source_vertex_indices=used))
    assert sum(len(part['faces']) for part in parts) == len(faces)
    return parts


def pack_msh(vertices, faces, normals=None):
    vertices = np.asarray(vertices, dtype='<f4').reshape(-1, 3)
    faces = np.asarray(faces, dtype='<i4').reshape(-1, 3)
    normals = (np.empty((0, 3), dtype='<f4') if normals is None else
               np.asarray(normals, dtype='<f4').reshape(-1, 3))
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
        raise ValueError('nonfinite mesh data')
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('face index outside vertex array')
    if len(normals) not in (0, len(vertices)):
        raise ValueError('normals must correspond to vertices')
    return (struct.pack('<4i', len(vertices), len(normals), 0, len(faces)) +
            vertices.tobytes() + normals.tobytes() + faces.tobytes())


def read_glb(path):
    """Reject unsupported material conversions instead of inventing a look.

    Current source files have COLOR_0, no textures and no explicit material.
    glTF's default alphaMode is OPAQUE: COLOR_0 alpha=0 does not hide a face.
    Preserve that behavior while recording the original RGBA in the manifest.
    """
    raw = path.read_bytes()
    if raw[:4] != b'glTF':
        raise ValueError(f'not a GLB: {path}')
    size, kind = struct.unpack_from('<II', raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError('missing GLB JSON chunk')
    document = json.loads(raw[20:20 + size])
    if document.get('materials') or document.get('textures'):
        raise ValueError('explicit GLB PBR/texture material needs an exact converter')
    for mesh in document['meshes']:
        for primitive in mesh['primitives']:
            if 'COLOR_0' not in primitive['attributes']:
                raise ValueError('source primitive has no COLOR_0')
    scene = trimesh.load(path, process=False)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    return raw, scene
