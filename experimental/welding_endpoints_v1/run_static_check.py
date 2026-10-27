"""Static candidate audit for one collision-free welding posture per source weld."""
import json, hashlib, sys, itertools
from pathlib import Path
import numpy as np
import fcl
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from collision_audit import CollisionWorld
from r2_safe_park import additional_self_checks
from acceptance import posture_passed
FACE_ROOT=ROOT/'build/eight_hour_v17/experimental_release/multi_point_map'
sys.path.insert(0,str(FACE_ROOT))
from source_face_world_adapter_v1 import _check_simultaneous_source_faces
from gun_kinematics import gun_mesh_parts

def main():
    scene=json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text())
    cmap=json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
    world=CollisionWorld(scene)
    robots={a['id']:a for a in scene['assets'] if a['kind']=='robot'}
    rids=list(robots)
    # Find static parks independently in corrected scene. Existing R2 park is tested first.
    archived=np.asarray(json.loads((ROOT/'build/r2_safe_park_search.json').read_text())['selected']['q_rad'])
    parks={}
    for rid in rids:
        model=world.models[rid]
        seeds=[model.q_initial.copy(), np.zeros(6), archived.copy()]
        seeds += [np.asarray([j1,j2,j3,0.,j5,1.5707963268]) for j1,j2,j3,j5 in
                  itertools.product((0.,-1.5,1.5,3.),(-.8,.2,.8),(-.2,.5),(0.,-.6,.6))]
        for q in seeds:
            # A park is tested against its environment and self here; the
            # simultaneous other-arm state is checked separately per endpoint.
            pairs=world.check(rid,q,max_pairs=sys.maxsize)+additional_self_checks(world,rid) if model.limits_valid(q) else [dict(asset_name='limits')]
            pairs=[p for p in pairs if p['asset_name']!='other_robot']
            if not pairs:
                parks[rid]=q;break
        print('PARK',robots[rid]['name'],'found',rid in parks,flush=True)
    if len(parks)!=2: raise RuntimeError('no corrected-scene collision-free static park for both robots')
    parts={};products={}
    for rid,model in world.models.items():
        asset=next(a for a in scene['assets'] if a['id']==model.gun_id)
        _,parts[rid]=gun_mesh_parts(json.loads((ROOT/asset['gun']['source_detail']).read_text()))
    for asset in scene['assets']:
        if asset['id'] in {w['product_id'] for w in scene['welds']}:
            mesh=trimesh.load(ROOT/'build'/asset['url'],force='mesh',process=False)
            mesh.apply_transform(np.asarray(asset['transform']).reshape(4,4));products[asset['id']]=mesh
    rows=[]; total=0
    for weld,row in zip(scene['welds'],cmap['rows']):
        candidates=[]
        for c in row['candidates']:
            if not c['fk_candidate_passed']: continue
            rid=c['robot_id']; q=np.asarray(c['q_rad'],float)
            if rid not in world.models or not world.models[rid].limits_valid(q): continue
            other=next(x for x in rids if x!=rid)
            state=_check_simultaneous_source_faces(world,robot_id=rid,other_robot_id=other,q=q,other_q=parks[other],
                driver_mm=0.,weld=weld,phase='closed',source_parts=parts[rid],product_mesh=products[weld['product_id']])
            tcp_err=float(np.linalg.norm(world.models[rid].tcp(q)[:3,3]-np.asarray(weld['world_xyz'])))
            electrodes=['gun:'+pid for pid,data in world.models[rid].gun_parts.items() if data['kind']=='electrode']
            product_obj=next(obj for oid,name,obj in world.static if oid==weld['product_id'])
            gaps={key:float(fcl.distance(world.robots[rid][key],product_obj,fcl.DistanceRequest(),fcl.DistanceResult()))
                  for key in electrodes if key not in state['certificates']}
            passed=posture_passed(state,tcp_err,electrodes,gaps)
            candidates.append({'robot_id':rid,'candidate_index':c['candidate_index'],'q_rad':q.tolist(),'tcp_error_mm':tcp_err,
                'pair_count':len(state['remaining_pairs']),'pairs':state['remaining_pairs'],'passed':passed,
                'electrode_gap_mm':gaps,'certificates':state['certificates'],'raw_contact_pairs':state['original_pairs']})
            total+=1
        selected=next((x for x in candidates if x['passed']),None)
        rows.append({'weld_id':weld['id'],'world_xyz':weld['world_xyz'],'candidate_count_checked':len(candidates),'selected':selected,'any_passed':selected is not None,'best_pair_count':min((x['pair_count'] for x in candidates),default=None),'best_candidates':sorted(candidates,key=lambda x:(x['pair_count'],x['tcp_error_mm']))[:5]})
        print(weld['id'],'checked',len(candidates),'selected',None if selected is None else (selected['robot_id'],selected['candidate_index']),'best_pairs',rows[-1]['best_pair_count'],flush=True)
    out={'schema':'corrected_scene_static_welding_posture_audit_v1','scene':'build/scene_placement_corrected_v1.json',
         'scene_sha256':hashlib.sha256((ROOT/'build/scene_placement_corrected_v1.json').read_bytes()).hexdigest(),
         'candidate_source':'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json',
         'contact_rule':'Only exact original electrode/designated product contacts certified within 25mm source-face bound and 2mm discretized contact depth; otherwise <=0.1mm separated gap. No gun-body/fixture tolerance.',
         'static_parks':{rid:q.tolist() for rid,q in parks.items()},'welds':rows,'summary':{'source_weld_count':len(rows),'welds_with_collision_free_posture':sum(x['any_passed'] for x in rows),'candidate_states_checked':total,'all_29_have_collision_free_posture':all(x['any_passed'] for x in rows),'scope':'static weld posture only; no ingress/retreat/transfer'}}
    outp=ROOT/'build/welding_endpoints_v1_corrected_static.json';outp.write_text(json.dumps(out,ensure_ascii=False,indent=2));
    md=['# 29个焊点静态焊接姿态核验','','场景：`build/scene_placement_corrected_v1.json`。仅核验焊接瞬间构型，不包含进枪、退枪或点间转移。','',f"候选构型检查 {total} 个；无非预期碰撞姿态 {out['summary']['welds_with_collision_free_posture']}/29。",'']
    md += ['|焊点|检查候选|无碰撞姿态|','|---|---:|---|']
    for x in rows: md.append(f"|{x['weld_id']}|{x['candidate_count_checked']}|{'有' if x['any_passed'] else '无'}|")
    (ROOT/'build/welding_endpoints_v1_corrected_static.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(out['summary'],ensure_ascii=False))
if __name__=='__main__':main()
