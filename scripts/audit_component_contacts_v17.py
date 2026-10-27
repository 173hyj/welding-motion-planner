#!/usr/bin/env python3
"""Trace FCL triangle contacts to exact raw JT named source components.

Names identify source nodes only; neither attachment nor component name grants
a safety exemption. Exact face ordering and float32 GLB vertices are verified
before contact triangle ids are attributed.
"""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import fcl
import numpy as np
import trimesh
from audit_counterbalance_fk_v17 import counterbalance

ROOT=Path(__file__).resolve().parents[1]


def trace_mesh(raw,root_id,skip_root=False,excluded=()):
    nodes={n['id']:n for n in raw['nodes']};vertices=[];faces=[];face_owner=[];components={};offset=0
    def visit(nid,t,path,first=False):
        nonlocal offset
        if nid in excluded:return
        node=nodes[nid]
        if node['transform']and not(first and skip_root):t=t@np.array(node['transform']).reshape(4,4).T
        name=node['properties'].get('JT_PROP_NAME')
        if name:path=[*path,{'node_id':nid,'name':name,'type':node['type']}]
        for mesh_id in node['meshes']:
            source=raw['meshes'][mesh_id];v=np.array(source['vertices']).reshape(-1,3)@t[:3,:3].T+t[:3,3];f=np.array(source['indices']).reshape(-1,3);f=f[(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])]
            owner=str(path[-1]['node_id'])if path else str(nid)
            c=components.setdefault(owner,{'source_node_id':int(owner),'named_path':path,'mesh_node_ids':[],'mesh_ids':[],'faces':0,'bounds_local_mm':[v.min(axis=0).tolist(),v.max(axis=0).tolist()]})
            c['mesh_node_ids'].append(nid);c['mesh_ids'].append(mesh_id);c['faces']+=len(f);c['bounds_local_mm']=np.stack((np.minimum(c['bounds_local_mm'][0],v.min(axis=0)),np.maximum(c['bounds_local_mm'][1],v.max(axis=0)))).tolist()
            vertices.append(v);faces.append(f+offset);face_owner.extend([owner]*len(f));offset+=len(v)
        children=node['children'][:1]if node['type']=='RangeLODNodeData'else node['children']
        for child in children:visit(child,t,path)
    visit(root_id,np.eye(4),[],True)
    return np.concatenate(vertices),np.concatenate(faces),face_owner,components


def object_for(mesh):
    model=fcl.BVHModel();v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int32);model.beginModel(len(v),len(f));model.addSubModel(v,f);model.endModel();return fcl.CollisionObject(model)


def audit(plan_path,scene_path,stride=10,correct_counterbalance=False,robot_only=False):
    scene=json.loads(scene_path.read_text());plan=json.loads(plan_path.read_text());source_hashes={};parts={};trace_checks={};robots=[a for a in scene['assets']if a['kind']=='robot'];joint_axis_by_pair={}
    def register(rid,key,raw_path,root_id,skip_root,excluded,url):
        raw=json.loads(raw_path.read_text());v,f,owners,components=trace_mesh(raw,root_id,skip_root,excluded);mesh_path=ROOT/'build'/url;mesh=trimesh.load(mesh_path,force='mesh',process=False)
        assert np.array_equal(mesh.faces,f),(key,'face ordering mismatch')
        assert np.array_equal(np.asarray(mesh.vertices,dtype=np.float32),np.asarray(v,dtype=np.float32)),(key,'vertices differ beyond source GLB quantization')
        trace_checks[rid+':'+key]={'exact_face_order_matches':True,'exact_float32_vertices_match':True,'source_to_glb_max_vertex_roundoff_mm':float(np.max(abs(mesh.vertices-v))),'source_components':components}
        parts[(rid,key)]={'object':object_for(mesh),'owners':owners,'components':components,'triangles':np.asarray(mesh.triangles)};source_hashes[str(raw_path.relative_to(ROOT))]=hashlib.sha256(raw_path.read_bytes()).hexdigest();source_hashes[str(mesh_path.relative_to(ROOT))]=hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    robot_raw_path=ROOT/'build/robot.json';robot_raw=json.loads(robot_raw_path.read_text());nodes={n['id']:n for n in robot_raw['nodes']}
    gunids={};pair_specs=[]
    for robot in robots:
        rid=robot['id']
        for key in ('240','241','246','247'):
            link=next(n for n in robot_raw['nodes']if n['properties'].get('TxObjType')=='TxLink'and n['properties'].get('TxGmProtoKey')==key);root=link if link['parent']is None else nodes[link['parent']]
            register(rid,key,robot_raw_path,root['id'],True,(),robot['robot']['link_models'][key]['url'])
        pair_specs.extend((rid,a,b)for a,b in [('240','246'),('241','246'),('240','247')])
        joint_axis_by_pair[(rid,'241','246')]=('473',np.array(scene['joints'][6]['axis1']),np.array(scene['joints'][6]['axis2']))
        joint_axis_by_pair[(rid,'240','247')]=('475',np.array(scene['joints'][8]['axis1']),np.array(scene['joints'][8]['axis2']))
        if robot_only:continue
        gun=next(a for a in scene['assets']if a['kind']=='gun'and ('R1'if 'R1'in robot['name']else 'R2')in a['name']);gunids[rid]=gun['id'];raw_path=ROOT/gun['gun']['source_detail'];raw=json.loads(raw_path.read_text());gnodes={n['id']:n for n in raw['nodes']};bodies={}
        for part_id,data in gun['gun']['link_models'].items():
            if data['kind']!='body':continue
            link=next(n for n in raw['nodes']if n['properties'].get('TxObjType')=='TxLink'and n['properties'].get('TxGmProtoKey')==data['link_key']);root=link if link['parent']in(None,1)else gnodes[link['parent']];caps={n['id']for n in raw['nodes']if 'K-07-0001.asm'in n['properties'].get('JT_PROP_NAME','')}
            register(rid,'gun:'+part_id,raw_path,root['id'],False,caps,data['url']);bodies[data['link_key']]='gun:'+part_id
        body_keys=list(bodies.values());fixed=body_keys[0];moving=body_keys[1];cylinder=body_keys[2];rod=body_keys[3]
        pair_specs.extend((rid,a,b)for a,b in [(fixed,moving),(fixed,cylinder),(moving,rod)])
        text=raw['kinematicsSource'];source_joints=list(ET.fromstring(text[text.index('<devices>'):text.index('</devices>')+10]).find('device/joints'))
        for j in source_joints:
            parent=bodies.get(j.findtext('parentLink'));child=bodies.get(j.findtext('childLink'))
            if parent is None or child is None:continue
            point=np.array([float(j.findtext('axis1/'+k))for k in ('x','y','z')]);tip=np.array([float(j.findtext('axis2/'+k))for k in ('x','y','z')]);joint_axis_by_pair[(rid,parent,child)]=(j.get('key'),point,tip)
    groups={};frames=list(range(0,len(plan['frames']),stride));saturated=[]
    for fi in frames:
        frame=plan['frames'][fi]
        truth_by_robot={}
        if correct_counterbalance:
            for robot in robots:
                rid=robot['id'];main,passive,_,_=counterbalance(frame['robots'][rid]['q_rad'],scene);base=np.array(robot['transform']).reshape(4,4);truth_by_robot[rid]={key:base@pose for key,pose in {**main,**passive}.items()}
        for (rid,key),data in parts.items():
            flat=frame['guns'][gunids[rid]]['parts'][key[4:]]if key.startswith('gun:')else frame['robots'][rid]['links'][key];pose=np.array(flat).reshape(4,4);data['pose']=pose;data['object'].setTransform(fcl.Transform(pose[:3,:3],pose[:3,3]))
            if correct_counterbalance and not key.startswith('gun:'):
                pose=truth_by_robot[rid][key];data['pose']=pose;data['object'].setTransform(fcl.Transform(pose[:3,:3],pose[:3,3]))
        for rid,a,b in pair_specs:
            max_contacts=200000
            pa,pb=parts[(rid,a)],parts[(rid,b)];result=fcl.CollisionResult();fcl.collide(pa['object'],pb['object'],fcl.CollisionRequest(num_max_contacts=max_contacts,enable_contact=True),result)
            if len(result.contacts)>=max_contacts:saturated.append((fi,rid,a,b))
            for c in result.contacts:
                if c.b1<0 or c.b2<0:raise ValueError('Expected BVH triangle contact identifiers')
                ca,cb=pa['owners'][c.b1],pb['owners'][c.b2];key=(rid,a,b,ca,cb);local_a=np.linalg.inv(pa['pose'])@np.r_[c.pos,1.];local_b=np.linalg.inv(pb['pose'])@np.r_[c.pos,1.]
                row=groups.setdefault(key,{'robot_id':rid,'a':a,'b':b,'source_component_a':pa['components'][ca],'source_component_b':pb['components'][cb],'contact_count':0,'frames':set(),'max_reported_triangle_penetration_mm':0.,'triangle_ids_a':set(),'triangle_ids_b':set(),'contact_examples':[]})
                row['frames'].add(fi);row['contact_count']+=1;row['max_reported_triangle_penetration_mm']=max(row['max_reported_triangle_penetration_mm'],float(c.penetration_depth));row['triangle_ids_a'].add(c.b1);row['triangle_ids_b'].add(c.b2)
                for label,point in (('a_local_contact_bounds_mm',local_a[:3]),('b_local_contact_bounds_mm',local_b[:3])):
                    old=row.get(label,[point.tolist(),point.tolist()]);row[label]=np.stack((np.minimum(old[0],point),np.maximum(old[1],point))).tolist()
                for label,triangle in (('a_local_participating_triangle_bounds_mm',pa['triangles'][c.b1]),('b_local_participating_triangle_bounds_mm',pb['triangles'][c.b2])):
                    low=triangle.min(axis=0);high=triangle.max(axis=0);old=row.get(label,[low.tolist(),high.tolist()]);row[label]=np.stack((np.minimum(old[0],low),np.maximum(old[1],high))).tolist()
                if (rid,a,b)in joint_axis_by_pair:
                    source_key,point,tip=joint_axis_by_pair[(rid,a,b)];axis=(tip-point)/np.linalg.norm(tip-point);radial=float(np.linalg.norm(np.cross(local_a[:3]-point,axis)));metrics=row.setdefault('source_joint_axis_contact_radius_mm',{'joint_key':source_key,'min':radial,'max':radial});metrics['min']=min(metrics['min'],radial);metrics['max']=max(metrics['max'],radial)
                if len(row['contact_examples'])<6:row['contact_examples'].append({'frame':fi,'phase':frame['phase'],'world_xyz_mm':c.pos.tolist(),'a_local_xyz_mm':local_a[:3].tolist(),'b_local_xyz_mm':local_b[:3].tolist(),'triangle_id_a':c.b1,'triangle_id_b':c.b2,'penetration_mm':float(c.penetration_depth)})
    rows=[]
    for row in groups.values():
        for key in ('frames','triangle_ids_a','triangle_ids_b'):row[key]=sorted(row[key])
        row['interface_safety_certified']=False;rows.append(row)
    source_hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in (plan_path,scene_path)})
    return {'scope':'exact GLB triangle ids mapped to raw JT named component paths; source mesh order verified independently','robot_pose_source':'independent source joint + orthonormal closure bases'if correct_counterbalance else 'archived matrices; old counterbalance FK is known invalid','robot_only':robot_only,'stride':stride,'contact_limit_per_check':200000,'sampled_frame_ids':frames,'source_hashes':source_hashes,'trace_verification':trace_checks,'component_contact_groups':rows,'saturated_checks':saturated,'unknown_hose_dress_exemption':False,'system_safe':False,'limitations':['Source code-style component names do not by themselves identify permitted contact surfaces.','Triangle contact depth is not solid penetration depth.','FCL contact positions are representative triangle-contact points and may not lie on both surfaces; use source triangle ids and participating triangle bounds to localize geometry.','Sampled component contacts localize evidence; they do not justify blanket link exemptions.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--scene',type=Path,default=ROOT/'build/scene.json');p.add_argument('--stride',type=int,default=10);p.add_argument('--correct-counterbalance',action='store_true');p.add_argument('--robot-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.plan.resolve(),a.scene.resolve(),a.stride,a.correct_counterbalance,a.robot_only);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps({'component_contact_groups':len(r['component_contact_groups']),'sampled_frames':len(r['sampled_frame_ids']),'saturated_checks':len(r['saturated_checks'])},indent=2))


if __name__=='__main__':main()
