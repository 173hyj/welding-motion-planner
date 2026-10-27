"""Source-mesh scene with an explicit rotary assembly interface list."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import fcl
import numpy as np
import trimesh
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experimental/welding_positioner_v1')]
from positioner import SourcePositioner
from collision_audit import CollisionWorld
from build_scene import flatten
from geometry import duplicate_source_faces, source_face_group_mask

OUT = ROOT/'build/welding_positioner_static_v2'
INTERFACE_NODES = {14768: '主动侧悬挂.asm;10878;14:', 14808: '从动悬挂.asm;11054;14:'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_scene(scene, out=OUT):
    out.mkdir(exist_ok=True)
    fixture = SourcePositioner(scene)
    layout_asset = next(a for a in scene['assets'] if a['kind'] == 'layout')
    sources = [ROOT/'build/scene_placement_corrected_v1.json', ROOT/'build/fixtureL.json',
               ROOT/'build/AP31-M7-S1-00-00.json', ROOT/'build'/layout_asset['url'], Path(__file__), Path(__file__).with_name('geometry.py')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    audit_path = out/'geometry_audit.json'
    if audit_path.exists():
        cached = json.loads(audit_path.read_text())
        if cached['source_sha256'] == hashes and all(sha(ROOT/p) == digest for p, digest in cached['mesh_sha256'].items()):
            return cached['scene'], cached
    raw = json.loads((ROOT/'build/fixtureL.json').read_text())
    moving = flatten(raw, [2]); fixed = flatten(raw, [6196])
    assert len(moving.faces)+len(fixed.faces) == fixture.asset['triangles']
    layout = trimesh.load(ROOT/'build'/layout_asset['url'], force='mesh', process=False)
    layout.apply_transform(np.asarray(layout_asset['transform']).reshape(4, 4))
    original_faces = len(layout.faces)
    bases = np.array([np.asarray(a['transform']).reshape(4, 4)[:3, 3] for a in scene['assets'] if a['kind'] == 'robot'])
    triangles = layout.vertices[layout.faces]
    crop = np.all(triangles.max(1) >= bases.min(0)-5000, 1) & np.all(triangles.min(1) <= bases.max(0)+5000, 1)
    del triangles
    layout_raw = json.loads((ROOT/'build/AP31-M7-S1-00-00.json').read_text())
    for nid, name in INTERFACE_NODES.items():
        assert next(n for n in layout_raw['nodes'] if n['id'] == nid)['properties']['JT_PROP_NAME'] == name
    interfaces = source_face_group_mask(layout_raw, set(INTERFACE_NODES))
    assert len(interfaces) == original_faces
    interfaces = interfaces[crop]
    layout = layout.submesh([np.flatnonzero(crop)], append=True, repair=False)
    duplicate = np.zeros(len(layout.faces), bool); audits = []
    for name, source in [('moving', moving), ('fixed', fixed)]:
        mesh = source.copy(); mesh.apply_transform(fixture.base)
        mask, audit = duplicate_source_faces(layout, mesh)
        duplicate |= mask; audits.append(dict(component=name, **audit))
        print('DEDUP', name, audit['duplicate_faces'], flush=True)
    external = layout.submesh([np.flatnonzero(~duplicate & ~interfaces)], append=True, repair=False)
    interface = layout.submesh([np.flatnonzero(~duplicate & interfaces)], append=True, repair=False)
    adapted = copy.deepcopy(scene)
    adapted['assets'] = [a for a in adapted['assets'] if a['id'] not in (layout_asset['id'], fixture.asset['id'])]
    mesh_hashes = {}
    for name, mesh, template, T in [('DONG', moving, fixture.asset, fixture.base), ('JING', fixed, fixture.asset, fixture.base),
                                  ('external', external, layout_asset, np.eye(4)), ('hanging_interface', interface, layout_asset, np.eye(4))]:
        path = out/(name+'.glb'); mesh.export(path)
        asset = copy.deepcopy(template)
        asset.update(id=template['id']+':'+name, name=template['name']+':'+name,
                     url=str(path.relative_to(ROOT/'build')), transform=T.ravel().tolist(), triangles=len(mesh.faces))
        adapted['assets'].append(asset); mesh_hashes[str(path.relative_to(ROOT))] = sha(path)
    audit = dict(source_sha256=hashes, mesh_sha256=mesh_hashes, scene=adapted,
                 original_layout_faces=original_faces, cropped_layout_faces=len(layout.faces),
                 duplicate_faces=int(duplicate.sum()), retained_external_faces=len(external.faces),
                 retained_interface_faces=len(interface.faces), duplicate_sources=audits,
                 assembly_interfaces=dict(layout_instance_nodes=INTERFACE_NODES,
                     rule='Only DONG-versus-JING and DONG-versus-declared-hanging-interface are excluded as rotary assembly interfaces. Robot/gun-versus-interface and product-versus-interface remain checked.',
                     qualification='Assembly grouping follows original component names/ancestry; release of locking devices and rotary motion safety are not verified.'))
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return adapted, audit


class PositionerWorld:
    def __init__(self, scene, out=OUT):
        self.source_scene = scene
        self.positioner = SourcePositioner(scene)
        adapted, self.audit = prepare_scene(scene, out)
        self.world = CollisionWorld(adapted)
        self.objects = {oid: obj for oid, name, obj in self.world.static}
        self.moving_id = self.positioner.asset['id']+':DONG'
        self.fixed_id = self.positioner.asset['id']+':JING'
        layout = next(a for a in scene['assets'] if a['kind'] == 'layout')
        self.interface_id = layout['id']+':hanging_interface'
        self.relative_ids = {self.moving_id, *self.positioner.attached_product_ids}
        self.relative_env = [x for x in self.world.static if x[0] in self.relative_ids]
        self.fixed_env = [x for x in self.world.static if x[0] not in self.relative_ids]
        self.angle_cache = {}
        self.set_angle(0)

    def set_angle(self, angle):
        d = self.positioner.delta(angle)
        # CollisionWorld bakes every source asset transform into its BVH.
        # Applying base a second time would move DONG to the wrong workcell.
        self.objects[self.moving_id].setTransform(fcl.Transform(d[:3, :3], d[:3, 3]))
        for pid in self.positioner.attached_product_ids:
            self.objects[pid].setTransform(fcl.Transform(d[:3, :3], d[:3, 3]))
        return d

    def layout_collisions(self, angle):
        if angle in self.angle_cache:
            return self.angle_cache[angle]
        self.set_angle(angle)
        blocked = []; interfaces = []
        for moving_id in self.relative_ids:
            for oid, name, environment in self.fixed_env:
                request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
                result = fcl.CollisionResult()
                if not fcl.collide(self.objects[moving_id], environment, request, result):
                    continue
                contact = result.contacts[0] if result.contacts else None
                record = dict(part=moving_id, asset_id=oid, asset_name=name,
                              contact_xyz_mm=np.asarray(contact.pos).tolist() if contact else None)
                if moving_id == self.moving_id and oid in (self.fixed_id, self.interface_id):
                    interfaces.append(record)
                else:
                    blocked.append(record)
        self.angle_cache[angle] = dict(blocking_pairs=blocked, assembled_interface_pairs=interfaces)
        return self.angle_cache[angle]
