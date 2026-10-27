#!/usr/bin/env python3
"""Independent source-XML FK and exact-electrode release contact audit.

No planner/production FK/collision module is imported. The fixed electrode is
attached to the source gun root, so its pose is independent of gun opening.
This focused check can refute release safety; it is not a full-system pass.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import fcl
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]


def joints_from_raw(raw):
    text=raw['kinematicsSource'];text=text[text.index('<devices>'):text.index('</devices>')+10]
    return list(ET.fromstring(text).find('device/joints'))


def source_rotation(joint,angle):
    p=np.array([float(joint.findtext('axis1/'+k))for k in ('x','y','z')])
    tip=np.array([float(joint.findtext('axis2/'+k))for k in ('x','y','z')]);axis=(tip-p)/np.linalg.norm(tip-p)
    x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    r=np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=p-r@p
    return t


def flange_source(q,joints):
    index={j.get('key'):i for i,j in enumerate(joints)};t=np.eye(4)
    for i,j in enumerate(joints):
        lead=j.find('leadingJoints');factors=j.find('leadingFactors')
        coupled=sum(q[index[k.text]]*float(f.text)for k,f in zip(list(lead)if lead is not None else [],list(factors)if factors is not None else []))
        t=t@source_rotation(j,q[i]+coupled)
    return t


def source_tcp_home(raw,robot):
    blob=base64.b64decode(raw['rawPmi'][1]['bytes']);v=struct.unpack_from('<9f',blob,112)
    tool=np.eye(4);tool[:3,3]=v[:3];tool[:3,0]=v[3:6];tool[:3,1]=v[6:9];tool[:3,2]=np.cross(tool[:3,0],tool[:3,1])
    return tool@np.linalg.inv(np.array(robot['robot']['tcp_relative_frame']).reshape(4,4))


def collision_object(mesh,pose):
    model=fcl.BVHModel();v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int32)
    model.beginModel(len(v),len(f));model.addSubModel(v,f);model.endModel()
    return fcl.CollisionObject(model,fcl.Transform(pose[:3,:3],pose[:3,3]))


def audit(plan_path,scene_path,baseline_path=None):
    scene=json.loads(scene_path.read_text());plan=json.loads(plan_path.read_text());raw_path=ROOT/'build/robot.json';raw=json.loads(raw_path.read_text());joints=joints_from_raw(raw)[:6]
    robot=next(a for a in scene['assets']if a['kind']=='robot' and 'R1'in a['name']);rid=robot['id'];base=np.array(robot['transform']).reshape(4,4)
    gun=next(a for a in scene['assets']if a['kind']=='gun'and 'R1'in a['name']);gun_raw_path=ROOT/gun['gun']['source_detail'];gun_joints=joints_from_raw(json.loads(gun_raw_path.read_text()));root_link=gun_joints[0].findtext('parentLink')
    electrode_id,part=next((k,p)for k,p in gun['gun']['link_models'].items()if p['kind']=='electrode'and p['link_key']==root_link)
    electrode_path=ROOT/'build'/part['url'];electrode=collision_object(trimesh.load(electrode_path,force='mesh',process=False),np.eye(4))
    gun_home=np.linalg.inv(base@flange_source(robot['robot']['q_initial'],joints))@np.array(gun['transform']).reshape(4,4);tcp_home=source_tcp_home(raw,robot)
    matrix_errors=[]
    for frame in plan['frames']:
        independent=base@flange_source(frame['robots'][rid]['q_rad'],joints)
        matrix_errors.append(float(np.max(abs(independent-np.array(frame['robots'][rid]['links']['245']).reshape(4,4)))))
    products={};rows=[];hashes={}
    for segment in plan['segments']:
        if segment['phase']not in ('retreat','approach'):continue
        weld=next(w for w in scene['welds']if w['id']==segment['weld_id']);target=np.array(weld['world_xyz'])
        asset=next(a for a in scene['assets']if a['id']==weld['product_id']);product_path=ROOT/'build'/asset['url']
        if asset['id']not in products:products[asset['id']]=collision_object(trimesh.load(product_path,force='mesh',process=False),np.array(asset['transform']).reshape(4,4))
        hashes[str(product_path.relative_to(ROOT))]=hashlib.sha256(product_path.read_bytes()).hexdigest()
        qa=np.array(segment['samples'][0]['q_rad']);qb=np.array(segment['samples'][-1]['q_rad'])
        if segment['phase']=='approach':qa,qb=qb,qa
        trials=[]
        for u in (0.,.0001,.0005,.001,.002,.005,.01,.02,.05):
            q=qa+u*(qb-qa);flange=base@flange_source(q,joints);pose=flange@gun_home
            electrode.setTransform(fcl.Transform(pose[:3,:3],pose[:3,3]));tcp=(flange@tcp_home)[:3,3];error=float(np.linalg.norm(tcp-target))
            result=fcl.CollisionResult();fcl.collide(electrode,products[asset['id']],fcl.CollisionRequest(num_max_contacts=10000,enable_contact=True),result)
            contacts=result.contacts;allow=bool(result.is_collision and error<.1 and len(contacts)<10000 and all(np.linalg.norm(c.pos-target)<=25 and c.penetration_depth<=2 for c in contacts))
            trials.append({'fraction':u,'tcp_error_mm':error,'raw_collision':bool(result.is_collision),'contact_count':len(contacts),'contacts_truncated':len(contacts)>=10000,'intentional_contact_allowance_satisfied':allow,'release_pair_valid':not result.is_collision or allow,'maximum_contact_depth_mm':max((float(c.penetration_depth)for c in contacts),default=0.),'maximum_contact_distance_to_weld_mm':max((float(np.linalg.norm(c.pos-target))for c in contacts),default=0.),'first_contact':{'xyz_mm':contacts[0].pos.tolist(),'depth_mm':float(contacts[0].penetration_depth)}if contacts else None})
        rows.append({'phase':segment['phase'],'weld_id':weld['id'],'electrode_part':electrode_id,'product_id':weld['product_id'],'trials':trials})
    comparison=None
    if baseline_path:
        baseline=json.loads(baseline_path.read_text());diff=[]
        for independent,old in zip(rows,baseline):
            assert (independent['phase'],independent['weld_id'])==(old['phase'],old['weld_id'])
            for a,b in zip(independent['trials'],old['trials']):diff.append({'tcp_error_difference_mm':abs(a['tcp_error_mm']-b['tcp_error_mm']),'validity_agrees':a['release_pair_valid']==b['collision_verified']})
        comparison={'maximum_tcp_error_difference_mm':max(x['tcp_error_difference_mm']for x in diff),'validity_agrees_all_samples':all(x['validity_agrees']for x in diff),'compared_samples':len(diff)}
    hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in (scene_path,plan_path,raw_path,gun_raw_path,electrode_path)})
    return {'scope':'fixed source electrode to designated full original workpiece; source XML Rodrigues FK, raw PMI TCP; no production planner/FK/collision imports','source_hashes':hashes,'max_flange_matrix_error_vs_archived':max(matrix_errors),'rows':rows,'production_baseline_crosscheck':comparison,'invalid_release_samples':sum(not t['release_pair_valid']for r in rows for t in r['trials']),'system_safe':False,'limitations':['Focused refutation check, not all-pair or continuous safety certificate.','Source TCP and contact tolerances reproduced unchanged: <0.1 mm TCP, <=25 mm contact radius, <=2 mm penetration, untruncated contacts.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--scene',type=Path,default=ROOT/'build/scene.json');p.add_argument('--baseline',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.plan.resolve(),a.scene.resolve(),a.baseline.resolve()if a.baseline else None);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k]for k in ('max_flange_matrix_error_vs_archived','production_baseline_crosscheck','invalid_release_samples')},indent=2))


if __name__=='__main__':main()
