"""Remove only geometrically duplicate layout faces, retaining their moving source."""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def source_face_group_mask(raw, interface_instance_ids):
    """Follow instance occurrences (JT is a DAG), keeping flatten's face order."""
    nodes = {n['id']: n for n in raw['nodes']}
    masks = []
    def visit(key, inside):
        node = nodes[key]
        inside = inside or key in interface_instance_ids
        for mesh_id in node['meshes']:
            faces = np.asarray(raw['meshes'][mesh_id]['indices']).reshape(-1, 3)
            count = np.count_nonzero((faces[:, 0] != faces[:, 1]) &
                                     (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]))
            masks.append(np.full(count, inside))
        children = node['children'][:1] if node['type'] == 'RangeLODNodeData' else node['children']
        for child in children:
            visit(child, inside)
    visit(0, False)
    return np.concatenate(masks)


def duplicate_source_faces(layout, source, tolerance_mm=.02, chunk_size=50000):
    reference = np.asarray(source.vertices)[source.faces]
    tree = cKDTree(reference.mean(axis=1))
    duplicate = np.zeros(len(layout.faces), dtype=bool)
    maximum = 0.
    for start in range(0, len(layout.faces), chunk_size):
        triangles = np.asarray(layout.vertices)[layout.faces[start:start + chunk_size]]
        distances, indices = tree.query(triangles.mean(axis=1), distance_upper_bound=tolerance_mm)
        possible = np.flatnonzero(np.isfinite(distances))
        if not len(possible):
            continue
        actual = triangles[possible]
        expected = reference[indices[possible]]
        errors = np.full(len(possible), np.inf)
        for permutation in itertools.permutations(range(3)):
            errors = np.minimum(errors, np.linalg.norm(actual - expected[:, permutation, :], axis=2).max(axis=1))
        # Several tessellations can share a centroid. The nearest centroid is
        # only an accelerator; search every neighbor before rejecting a match.
        for local in np.flatnonzero(errors > tolerance_mm):
            neighbors = tree.query_ball_point(actual[local].mean(axis=0), tolerance_mm)
            candidates = reference[neighbors]
            for permutation in itertools.permutations(range(3)):
                difference = np.linalg.norm(actual[local] - candidates[:, permutation, :], axis=2).max(axis=1)
                errors[local] = min(errors[local], float(difference.min()))
        matched = errors <= tolerance_mm
        duplicate[start + possible[matched]] = True
        if matched.any():
            maximum = max(maximum, float(errors[matched].max()))
    return duplicate, dict(layout_faces=len(layout.faces), source_faces=len(source.faces),
                          duplicate_faces=int(duplicate.sum()), tolerance_mm=tolerance_mm,
                          maximum_vertex_difference_mm=maximum,
                          rule='All three vertices of an entire source triangle must match, allowing only vertex permutation; centroid proximity alone never removes a face.')
