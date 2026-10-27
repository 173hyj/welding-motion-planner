"""Fixed-position/fixed-normal roll search; no motion planning."""
import hashlib, json, sys, time, itertools, argparse
from pathlib import Path
import numpy as np
import fcl, trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experimental/welding_endpoints_v1'),str(ROOT),str(ROOT/'build/eight_hour_v17/experimental_release/multi_point_map')]
from collision_audit import CollisionWorld
from gun_kinematics import gun_mesh_parts
from r2_safe_park import additional_self_checks
from refinement import solve_seed_bank, roll_target, refinement_angles
from source_face_world_adapter_v1 import _check_simultaneous_source_faces
from acceptance import posture_passed
OUT=ROOT/'build/welding_endpoints_fixed_normal_v3'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists() and not args.resume:raise FileExistsError('preserve prior refinement run')
    checkpoint=json.loads((OUT/'report.json').read_text()) if args.resume and (OUT/'report.json').exists() else None
    scene_path=ROOT/'build/scene_placement_corrected_v1.json'
    scene=json.loads(scene_path.read_text()); baseline=json.loads((ROOT/'build/welding_endpoints_fixed_normal_v2/report.json').read_text())
    park_baseline=json.loads((ROOT/'build/welding_endpoints_v1_corrected_static.json').read_text())
    cmap=json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
    world=CollisionWorld(scene);rids=list(world.models); parts={};products={};parks={}
    started=time.time()
    for rid,model in world.models.items():
        asset=next(a for a in scene['assets'] if a['id']==model.gun_id)
        _,parts[rid]=gun_mesh_parts(json.loads((ROOT/asset['gun']['source_detail']).read_text()))
        parks[rid]=[np.asarray(park_baseline['static_parks'][rid])]
        # Several independently clear stationary parks, to avoid rejecting an
        # endpoint only because the unused arm was parked in its workspace.
        grid=itertools.product((3.,-3.,1.5,-1.5,0.),(-.8,.2,.8),(-.2,.5),(0.,-.6,.6))
        for j1,j2,j3,j5 in grid:
            q=np.array([j1,j2,j3,0.,j5,np.pi/2])
            if not model.limits_valid(q):continue
            pairs=world.check(rid,q,max_pairs=sys.maxsize)+additional_self_checks(world,rid)
            if not [p for p in pairs if p['asset_name']!='other_robot']:
                if not any(np.linalg.norm(q-prev)<.01 for prev in parks[rid]):parks[rid].append(q)
                if len(parks[rid])>=8:break
        print('PARKS',model.asset['name'],len(parks[rid]),flush=True)
    for weld in scene['welds']:
        pid=weld['product_id']
        if pid not in products:
            asset=next(a for a in scene['assets'] if a['id']==pid)
            mesh=trimesh.load(ROOT/'build'/asset['url'],force='mesh',process=False)
            mesh.apply_transform(np.asarray(asset['transform']).reshape(4,4));products[pid]=mesh
    rows=checkpoint['rows'] if checkpoint else []
    if checkpoint:
        assert checkpoint['scene_sha256']==sha(scene_path)
        assert checkpoint['previous_report_sha256']==sha(ROOT/'build/welding_endpoints_fixed_normal_v2/report.json')
        started-=checkpoint['summary']['elapsed_s']
    for index,(weld,maprow,old) in enumerate(zip(scene['welds'],cmap['rows'],baseline['rows'])):
        if index<len(rows):continue
        asset=next(a for a in scene['assets'] if a['id']==weld['product_id'])
        base=np.eye(4);base[:3,3]=weld['world_xyz']
        base[:3,:3]=np.asarray(asset['transform']).reshape(4,4)[:3,:3]@Rotation.from_euler('xyz',weld['projected_local_pose'][3:]).as_matrix()
        mesh_normal=np.asarray(maprow['nearest_product_surface']['normal_world'])
        agreement=float(np.rad2deg(np.arccos(np.clip(abs(mesh_normal@base[:3,2]),0,1))))
        selected=None; attempts=[]; gun_rows=[]; geochecks=0
        def full_check(rid,q,deg,sign):
            nonlocal selected
            model=world.models[rid];other=next(r for r in rids if r!=rid)
            for park in parks[other]:
                state=_check_simultaneous_source_faces(world,robot_id=rid,other_robot_id=other,q=q,other_q=park,
                    driver_mm=0.,weld=weld,phase='closed',source_parts=parts[rid],product_mesh=products[weld['product_id']])
                electrodes=['gun:'+pid for pid,info in model.gun_parts.items() if info['kind']=='electrode']
                product_obj=next(obj for oid,name,obj in world.static if oid==weld['product_id'])
                gaps={key:float(fcl.distance(world.robots[rid][key],product_obj,fcl.DistanceRequest(),fcl.DistanceResult()))
                      for key in electrodes if key not in state['certificates']}
                pose=model.tcp(q);err=float(np.linalg.norm(pose[:3,3]-base[:3,3]))
                normal_err=float(np.rad2deg(np.arccos(np.clip(abs(pose[:3,2]@base[:3,2]),0,1))))
                passed=posture_passed(state,err,electrodes,gaps) and normal_err<.01
                attempt=dict(robot_id=rid,q_rad=q.tolist(),other_robot_id=other,other_q_rad=park.tolist(),roll_deg=deg,axis_sign=sign,
                    passed=passed,tcp_error_mm=err,normal_axis_error_deg=normal_err,remaining_pairs=state['remaining_pairs'],
                    electrode_gap_mm=gaps,certificates=state['certificates'],raw_contacts=state['raw_contacts'],
                    original_pairs=state['original_pairs'],actual_tcp_matrix=pose.ravel().tolist())
                attempts.append(attempt)
                if passed:selected=attempt;return True
                # Changing other robot's park cannot fix own environment/self/contact errors.
                if any(p['asset_name']!='other_robot' for p in state['remaining_pairs']):break
            return False
        if old['selected']:
            c=old['selected'];q=np.asarray(c['q_rad']);pose=world.models[c['robot_id']].tcp(q)
            if abs(pose[:3,2]@base[:3,2])>1-1e-8:full_check(c['robot_id'],q,c['roll_deg'],c['axis_sign'])
        # The gun pose depends only on the target TCP, not on the IK branch.
        # Screen all source gun parts against fixtures and unintended products
        # before solving the much more expensive full arm inverse kinematics.
        for step in (1.,.25):
            if selected:break
            for deg in np.arange(0.,360.,step):
                if step==.25 and abs(deg-round(deg))<1e-9:continue
                for sign in (1,-1):
                    target=base.copy()
                    if sign<0:target[:3,:3]=target[:3,:3]@Rotation.from_euler('x',np.pi).as_matrix()
                    target[:3,:3]=target[:3,:3]@Rotation.from_euler('z',deg,degrees=True).as_matrix()
                    for rid in rids:
                        model=world.models[rid];gun_T=target@np.linalg.inv(model.tcp_home)@model.gun_home
                        pairs=[]
                        for pid,(link,kind,mesh) in parts[rid].items():
                            obj=world.robots[rid]['gun:'+pid]
                            obj.setTransform(fcl.Transform(gun_T[:3,:3],gun_T[:3,3]))
                            for oid,name,env in world.static:
                                if kind=='electrode' and oid==weld['product_id']:continue
                                res=fcl.CollisionResult()
                                if fcl.collide(obj,env,fcl.CollisionRequest(),res):pairs.append(dict(part=pid,asset_id=oid,asset_name=name))
                        geochecks+=1
                        gun_rows.append(dict(robot_id=rid,roll_deg=float(deg),axis_sign=sign,blocking_pairs=pairs))
                        if pairs:continue
                        sources=[np.asarray(c['q_rad']) for c in maprow['candidates'] if c['robot_id']==rid and c['fk_candidate_passed']]
                        seed=sources[0] if sources else model.q_initial
                        # More than one wrist/elbow branch; exact dependent source limits enforced by solve().
                        near=[np.asarray(a['q_rad']) for a in old['full_checks'] if a['robot_id']==rid]
                        solutions,ik_stats=solve_seed_bank(model,target,[seed]+sources+near[:8],random_count=64)
                        gun_rows[-1]['ik_stats']=ik_stats
                        for solution in solutions:
                            if full_check(rid,np.asarray(solution['q_rad']),float(deg),sign):break
                        print('GUN_CLEAR',weld['id'],model.asset['name'],'roll',deg,'sign',sign,'IK',len(solutions),'passed',selected is not None,flush=True)
                        if selected:break
                    if selected:break
                if selected:break
        row=dict(weld_id=weld['id'],world_xyz=weld['world_xyz'],normal_world=base[:3,2].tolist(),
            projected_normal_vs_source_mesh_deg=agreement,selected=selected,passed=selected is not None,
            gun_placements_checked=geochecks,gun_clear_placements=sum(not r['blocking_pairs'] for r in gun_rows),
            ik_states_checked=len(attempts),full_checks=attempts,gun_screen=gun_rows,
            status='static_posture_verified' if selected else 'not_found_in_fixed_normal_roll_search')
        rows.append(row)
        report=dict(scene_sha256=sha(scene_path),scope='Only static closed-gun welding postures. No motions planned.',
            previous_report_sha256=sha(ROOT/'build/welding_endpoints_fixed_normal_v2/report.json'),
            roll_sampling_step_deg=.25,ik_seed_method='analytic source screw-axis Jacobian, source/inventory/neighbor seeds and 64 legal scrambled Sobol starts',
            contact_rule='Exact original electrode/designated product faces only; 25mm patch, <=2mm discretized intended contact depth. Other contacts rejected.',
            normal_source='source projected normal independently checked against source workpiece mesh; axis fixed, full-circle roll searched',
            rows=rows,summary=dict(completed_welds=len(rows),passed_welds=sum(r['passed'] for r in rows),
                source_welds=29,all_29_passed=len(rows)==29 and all(r['passed'] for r in rows),elapsed_s=time.time()-started))
        (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('POINT',index+1,'/29',weld['id'],'PASS',row['passed'],'gun clear',row['gun_clear_placements'],'/',geochecks,'IK checked',len(attempts),flush=True)
    lines=['# 29焊点：固定法向、自由滚转静态姿态检查','',f"通过 {report['summary']['passed_welds']}/29。只检查焊接姿态，不规划运动。",'',
        '采用修正夹具位置的原布局，两台机器人均参与候选。保留完整原网格、原关节限制和双方机器人同时状态。',
        'TCP固定于原焊点；夹紧轴固定于源投影法向（与实际零件面法向交叉核验）；绕轴搜索滚转与不同逆解分支。',
        '电极与指定工件的预期接触独立按原电极源面核验；枪体、夹具、自碰撞及其他工件均不允许碰撞。',
        '未通过代表本轮搜索未找到，不能证明任何姿态都不可行。','',
        '|焊点|结果|焊枪无干涉角度样本|完整逆解检查|','|---|---|---:|---:|']
    for r in rows:lines.append(f"|{r['weld_id']}|{'通过' if r['passed'] else '未找到'}|{r['gun_clear_placements']}|{r['ik_states_checked']}|")
    (OUT/'结果报告.md').write_text('\n'.join(lines)+'\n')
    print('SUMMARY',json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
