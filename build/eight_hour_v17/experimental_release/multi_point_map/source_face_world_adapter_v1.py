"""Independent derived experiment adapter; original CollisionWorld unchanged."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from source_face_radius_v1 import certify_source_faces, filter_certified_pairs
from r2_safe_park import additional_self_checks as _additional_self_checks

def _validate_state_inputs(world,robot_id,other_robot_id,q,other_q,driver_mm,phase,margin_mm):
    if robot_id==other_robot_id or robot_id not in world.models or other_robot_id not in world.models:
        raise ValueError('Expected two distinct source robot ids')
    if phase not in ('closed','opening','closing','release','retreat','return','transfer'):
        raise ValueError('Unknown collision-check phase')
    if not np.isfinite(driver_mm) or not np.isfinite(margin_mm) or margin_mm<0:
        raise ValueError('Invalid driver/margin')
    if phase=='closed' and driver_mm!=0.:raise ValueError('Closed phase requires exact closed driver')
    if phase in ('release','retreat','return') and driver_mm!=-185.:raise ValueError('Separated/release phase requires exact source-open driver')
    model=world.models[robot_id]
    low,high=model.gun_model.limits_mm
    if not low<=driver_mm<=high:raise ValueError('Driver outside source gun limits')
    qa,qb=np.asarray(q,dtype=float),np.asarray(other_q,dtype=float)
    if qa.shape!=(6,) or qb.shape!=(6,) or not np.all(np.isfinite(qa)) or not np.all(np.isfinite(qb)):
        raise ValueError('Expected finite six-joint states')
    if not model.limits_valid(qa) or not world.models[other_robot_id].limits_valid(qb):
        raise ValueError('Source joint limits invalid before any FK/update')
    return qa,qb

def _check_simultaneous_source_faces(world, *, robot_id, other_robot_id, q, other_q,
                                    driver_mm, weld, phase, source_parts,
                                    product_mesh, margin_mm=0.):
    """Fully enumerate original pairs first, then certify exact source cap pairs.

    source_parts must be the gun_mesh_parts source mapping used by this world;
    product_mesh must be the transformed source product mesh, unmodified.
    """
    q,other_q=_validate_state_inputs(world,robot_id,other_robot_id,q,other_q,driver_mm,phase,margin_mm)
    import fcl
    world.update(other_robot_id,other_q,0.)
    world.update(robot_id,q,driver_mm)
    original=[]
    for rid,qi,di in [(robot_id,q,driver_mm),(other_robot_id,other_q,0.)]:
        pairs=world.check(rid,qi,weld=None,gun_driver_mm=di,
                          margin_mm=margin_mm,max_pairs=sys.maxsize)
        pairs+=_additional_self_checks(world,rid)
        original.extend(dict(p,checked_robot_id=rid) for p in pairs)
    certificates={}; raw_contacts={}
    model=world.models[robot_id]
    if phase in ('closed','opening','closing') and weld is not None:
        product=next(ob for oid,name,ob in world.static if oid==weld['product_id'])
        for key in sorted({p['link'] for p in original if p.get('checked_robot_id')==robot_id
                           and p.get('asset_id')==weld['product_id'] and 'distance_mm' not in p}):
            if not key.startswith('gun:') or key[4:] not in model.gun_parts or model.gun_parts[key[4:]]['kind']!='electrode':
                continue
            pid=key[4:];link,kind,mesh=source_parts[pid]
            if kind!='electrode' or link!=model.gun_parts[pid]['link_key']:
                raise RuntimeError('Source gun part identity mismatch')
            T=model.links(q,world=True)['245']@model.gun_home@model.gun_model.transforms(driver_mm)[link]
            result=fcl.CollisionResult()
            fcl.collide(world.robots[robot_id][key],product,
                        fcl.CollisionRequest(num_max_contacts=10000,enable_contact=True),result)
            contacts=[{'moving_face':int(c.b1),'product_face':int(c.b2),
                       'penetration_depth_mm':float(c.penetration_depth),
                       'contact_xyz_mm':np.asarray(c.pos).tolist(),
                       'contact_normal':np.asarray(c.normal).tolist()} for c in result.contacts]
            raw_contacts[key]=contacts
            certificates[key]=certify_source_faces(vertices=mesh.vertices,faces=mesh.faces,transform=T,
                origin=weld['world_xyz'],tcp_xyz=model.tcp(q)[:3,3],contacts=contacts,
                product_face_count=len(product_mesh.faces),raw_collision=bool(result.is_collision),
                truncated=len(contacts)>=10000,electrode_is_source=True,designated_product=True)
    limits=bool(model.limits_valid(q) and world.models[other_robot_id].limits_valid(other_q))
    if not limits:
        for cert in certificates.values():
            cert['passed']=False;cert['failure_reasons'].append('source_joint_limits_invalid')
    filtered=filter_certified_pairs(original,robot_id=robot_id,product_id=weld['product_id'] if weld else None,
                                    certificates=certificates,phase=phase)
    return {'original_pairs':original,'all_original_pairs_enumerated_before_filter':True,
            **filtered,'certificates':certificates,'raw_contacts':raw_contacts,
            'both_source_joint_limits_valid':limits,'phase':phase,'margin_mm':margin_mm,
            'derived_state_passed':limits and not filtered['remaining_pairs'],
            'legacy_gate_changed':False,'strict_planned_task_count':0,
            'collision_verified':False,'system_safe':False,
            'source_to_alternative_park_verified':False}

class SourceFaceCollisionAdapter:
    """Bind authoritative immutable meshes once; callers cannot replace faces."""
    def __init__(self,world):
        import trimesh
        from gun_kinematics import gun_mesh_parts
        self.world=world
        scene_path=ROOT/'build/scene.json'
        self.scene=json.loads(scene_path.read_text())
        if world.scene!=self.scene:raise RuntimeError('World scene differs from frozen source scene')
        frozen=json.loads((Path(__file__).parent.parent/'point7_release_report_v3.json').read_text())['summary']['source_hashes_after']
        self.frozen=frozen
        self._verify_hashes()
        self.parts={};self.products={}
        for rid,model in world.models.items():
            asset=next(a for a in self.scene['assets'] if a['id']==model.gun_id)
            _,parts=gun_mesh_parts(json.loads((ROOT/asset['gun']['source_detail']).read_text()))
            self.parts[rid]=parts
        for weld in self.scene['welds']:
            pid=weld['product_id']
            if pid in self.products:continue
            asset=next(a for a in self.scene['assets'] if a['id']==pid)
            mesh=trimesh.load(ROOT/'build'/asset['url'],force='mesh',process=False)
            mesh.apply_transform(np.asarray(asset['transform']).reshape(4,4))
            self.products[pid]=mesh
    def _verify_hashes(self):
        for name,expected in self.frozen.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Frozen source changed: {name}')
    def verify_unchanged(self):
        self._verify_hashes()
        return True
    def check(self,*,robot_id,other_robot_id,q,other_q,driver_mm,weld,phase,margin_mm=0.):
        q,other_q=_validate_state_inputs(self.world,robot_id,other_robot_id,q,other_q,driver_mm,phase,margin_mm)
        source_weld=next((w for w in self.scene['welds'] if weld and w['id']==weld['id']),None)
        if weld is not None and source_weld!=weld:raise RuntimeError('Weld differs from frozen source')
        # Source driver limits are enforced by GunKinematics.transforms.
        result=_check_simultaneous_source_faces(self.world,robot_id=robot_id,other_robot_id=other_robot_id,
            q=q,other_q=other_q,driver_mm=driver_mm,weld=source_weld,phase=phase,
            source_parts=self.parts[robot_id],product_mesh=self.products[source_weld['product_id']] if source_weld else None,
            margin_mm=margin_mm)
        return result
