#!/usr/bin/env python3
"""Independent direct-FCL audit of collision pairs omitted by production policy.

This script does not call CollisionWorld, planner FK, or display LOD meshes. It
uses archived world matrices and original source GLBs, and reports contacts;
it never turns an omitted pair into a safety pass.
"""
import argparse, hashlib, json
import xml.etree.ElementTree as ET
from pathlib import Path
import fcl
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]

def obj(mesh):
    m=fcl.BVHModel();v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int32)
    m.beginModel(len(v),len(f));m.addSubModel(v,f);m.endModel();return fcl.CollisionObject(m)
def set_pose(x,flat):
    t=np.asarray(flat,float).reshape(4,4);x.setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
def collide(a,b):
    r=fcl.CollisionResult();fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=8,enable_contact=True),r);return r

def omitted_self_pairs(keys):
    """Exact complement of CollisionWorld's current checked self pairs."""
    keys=list(keys);main=[str(i)for i in range(240,246)]
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            ag=a.startswith('gun:');bg=b.startswith('gun:')
            if ag and bg:policy='internal_gun_pair_exemption'
            elif ag or bg:
                robot=b if ag else a
                if robot in main[:-1]:continue
                policy='flange245_gun_exemption'if robot=='245'else 'base_or_counterweight_gun_exemption'
            elif a in main and b in main:
                if abs(int(a)-int(b))>1:continue
                policy='adjacent_main_link_exemption'
            else:policy='base_or_counterweight_self_exemption'
            yield a,b,policy

def audit(scene_path,plan_path,stride=1):
    scene=json.loads(scene_path.read_text());plan=json.loads(plan_path.read_text())
    robots=[a for a in scene['assets'] if a['kind']=='robot'];static=[];rparts={};gun_by_robot={};gun_sources={}
    for a in scene['assets']:
        if a['kind'] in ('robot','gun'):continue
        x=obj(trimesh.load(ROOT/'build'/a['url'],force='mesh',process=False));set_pose(x,a['transform']);static.append((a['id'],a['name'],x))
    for robot in robots:
        gun=next(a for a in scene['assets'] if a['kind']=='gun' and ('R1' if 'R1' in robot['name'] else 'R2') in a['name']);gun_by_robot[robot['id']]=gun['id'];parts={}
        gun_joint_graph={frozenset((j.findtext('parentLink'),j.findtext('childLink')))for j in ET.fromstring(gun['gun']['source_xml']).find('device/joints')}
        gun_sources[robot['id']]=(gun['gun']['link_models'],gun_joint_graph)
        for k,d in robot['robot']['link_models'].items():parts[k]=obj(trimesh.load(ROOT/'build'/d['url'],force='mesh',process=False))
        for k,d in gun['gun']['link_models'].items():parts['gun:'+k]=obj(trimesh.load(ROOT/'build'/d['url'],force='mesh',process=False))
        rparts[robot['id']]=parts
    groups={}
    checks=0
    graph={frozenset((str(j['parentLink']),str(j['childLink'])))for j in scene['joints']}
    def record(rid,a,b,policy,frame,other_name,result):
        if not result.is_collision:return
        key=(rid,a,b,policy)
        source='source_joint_attachment'if frozenset((a,b))in graph else 'requires_component_specific_source_evidence'
        if a.startswith('gun:') and b.startswith('gun:'):
            models,edges=gun_sources[rid];ka=models[a[4:]]['link_key'];kb=models[b[4:]]['link_key']
            if ka==kb:source='same_source_rigid_link'
            elif frozenset((ka,kb))in edges:source='source_gun_joint_attachment'
        row=groups.setdefault(key,{'robot_id':rid,'a':a,'b':b,'production_policy':policy,'source_interface_evidence':source,'other_name':other_name,'frames':[],'max_penetration_mm':0.,'contacts':[]})
        row['frames'].append(frame['index'])
        for c in result.contacts:
            row['max_penetration_mm']=max(row['max_penetration_mm'],float(c.penetration_depth))
            if len(row['contacts'])<6:row['contacts'].append({'frame':frame['index'],'time_s':frame['time_s'],'phase':frame.get('phase'),'xyz_mm':c.pos.tolist(),'depth_mm':float(c.penetration_depth)})
    for fi,rawframe in enumerate(plan['frames']):
        if fi%stride:continue
        frame={'index':fi,**rawframe}
        # Update every robot first. No robot is left at an earlier frame.
        for robot in robots:
            rid=robot['id'];parts=rparts[rid];links=rawframe['robots'][rid]['links'];gun=rawframe['guns'][gun_by_robot[rid]]
            for k,x in parts.items():set_pose(x,gun['parts'][k[4:]] if k.startswith('gun:') else links[k])
        for robot in robots:
            rid=robot['id'];parts=rparts[rid];main=[str(i) for i in range(240,246)];guns=[k for k in parts if k.startswith('gun:')]
            for a,b,policy in omitted_self_pairs(parts):
                checks+=1;record(rid,a,b,policy,frame,'self',collide(parts[a],parts[b]))
            # Base is skipped before every environment/other robot check.
            for oid,name,x in static:
                checks+=1;record(rid,'239',oid,'base239_environment_exemption',frame,name,collide(parts['239'],x))
            for other,oparts in rparts.items():
                if other==rid:continue
                for b,x in oparts.items():
                    checks+=1;record(rid,'239',other+':'+b,'base239_other_robot_exemption',frame,'other_robot',collide(parts['239'],x))
    rows=list(groups.values())
    return {'scope':'direct original source GLB FCL; all robots updated before each archived frame; omitted-pair contacts only', 'scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'frames':len(plan['frames']),'stride':stride,'pair_checks':checks,'colliding_pair_groups':len(rows),'pairs':rows,'contact_limit_per_pair':8,'unknown_dress_hose_policy':'no exemption; no dress/hose component is silently removed','limitations':['FCL triangle contact is not a continuous solid-volume proof.','Production policy and source joint adjacency are recorded as evidence, not accepted as safety.','Static source meshes are full GLBs; no layout crop or display LOD is used.','Penetration maxima are over returned contacts (up to 8 per check), not all mesh contacts; collision detection itself is not truncated.','This audits archived poses. The original 105/210 plans contain a known invalid counterbalance FK; their 246/247 contacts must not be treated as corrected physical-pose evidence. Use the independent corrected counterbalance component report.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--scene',type=Path,default=ROOT/'build/scene.json');p.add_argument('--stride',type=int,default=1);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.scene.resolve(),a.plan.resolve(),a.stride);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps({'frames':r['frames'],'stride':r['stride'],'colliding_pair_groups':r['colliding_pair_groups'],'pairs':len(r['pairs'])},indent=2))
if __name__=='__main__':main()
