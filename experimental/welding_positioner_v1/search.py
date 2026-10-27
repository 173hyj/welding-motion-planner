"""5-degree static posture screening, then real source external-axis search."""
import argparse,copy,hashlib,json,sys,time,itertools
from pathlib import Path
import numpy as np,fcl,trimesh
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'experimental/welding_endpoints_v1'),str(ROOT),str(ROOT/'build/eight_hour_v17/experimental_release/multi_point_map')]
from positioner import SourcePositioner,transform_weld,angle_grid
from refinement import roll_target,solve_seed_bank
from acceptance import posture_passed
from collision_audit import CollisionWorld,fcl_object
from build_scene import flatten
from gun_kinematics import gun_mesh_parts
from r2_safe_park import additional_self_checks
from source_face_world_adapter_v1 import _check_simultaneous_source_faces
OUT=ROOT/'build/welding_positioner_static_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);report_path=OUT/'report.json'
    if report_path.exists() and not args.resume:raise FileExistsError('Use --resume to preserve existing search')
    source_path=ROOT/'build/scene_placement_corrected_v1.json';scene=json.loads(source_path.read_text());source_hash=sha(source_path)
    inventory=json.loads((ROOT/'build/eight_hour_v17/experimental_release/multi_point_map/candidate_map_v1.json').read_text())
    previous=json.loads((ROOT/'build/welding_endpoints_fixed_normal_v2/report.json').read_text());prev_by_id={r['weld_id']:r for r in previous['rows']}
    fixture=SourcePositioner(scene);world=CollisionWorld(scene);rids=list(world.models)
    raw=json.loads((ROOT/'build/fixtureL.json').read_text());moving=flatten(raw,[2]);fixed=flatten(raw,[6196]);
    full_triangles=next(a.get('triangles') for a in scene['assets']if a['id']==fixture.asset['id'])
    assert len(moving.faces)+len(fixed.faces)==full_triangles
    for name,mesh in [('moving',moving),('fixed',fixed)]:
        p=OUT/('positioner_'+name+'.glb')
        if not p.exists():mesh.export(p)
    moving_obj=fcl_object(moving,fixture.base);fixed_obj=fcl_object(fixed,fixture.base)
    world.static=[x for x in world.static if x[0]!=fixture.asset['id']]
    moving_id=fixture.asset['id']+':DONG';fixed_id=fixture.asset['id']+':JING'
    world.static+=[(moving_id,'ST1-LZ-OP10L:DONG',moving_obj),(fixed_id,'ST1-LZ-OP10L:JING',fixed_obj)]
    product_objs={oid:obj for oid,name,obj in world.static if oid in fixture.attached_product_ids}
    parts={};products={};parks={}
    static_park=json.loads((ROOT/'build/welding_endpoints_v1_corrected_static.json').read_text())['static_parks']
    for rid,model in world.models.items():
        gun=next(a for a in scene['assets']if a['id']==model.gun_id);_,parts[rid]=gun_mesh_parts(json.loads((ROOT/gun['gun']['source_detail']).read_text()))
        parks[rid]=[np.asarray(static_park[rid]),model.q_initial,np.zeros(6)]
        parks[rid]+=[np.array([j1,j2,j3,0.,j5,np.pi/2])for j1,j2,j3,j5 in itertools.product((3.,-3.,1.5,-1.5,0.),(-.8,.2,.8),(-.2,.5),(0.,-.6,.6))]
        parks[rid]=[q for q in parks[rid]if model.limits_valid(q)]
    for pid in fixture.attached_product_ids:
        asset=next(a for a in scene['assets']if a['id']==pid);mesh=trimesh.load(ROOT/'build'/asset['url'],force='mesh',process=False)
        mesh.apply_transform(np.asarray(asset['transform']).reshape(4,4));products[pid]=mesh
    relative_ids={moving_id,*fixture.attached_product_ids}
    relative_env=[x for x in world.static if x[0]in relative_ids]
    fixed_env=[x for x in world.static if x[0]not in relative_ids]
    checkpoint=json.loads(report_path.read_text())if args.resume and report_path.exists()else None
    if checkpoint:assert checkpoint['scene_sha256']==source_hash
    rows=checkpoint['rows']if checkpoint else [];started=time.time()-(checkpoint['summary']['elapsed_s']if checkpoint else 0.)
    traces=[];park_cache={};current_angle=None;fixture_angle_reject={}
    def set_angle(deg):
        nonlocal current_angle
        d=fixture.delta(deg);T=d@fixture.base;moving_obj.setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
        for obj in product_objs.values():obj.setTransform(fcl.Transform(d[:3,:3],d[:3,3]))
        current_angle=deg
        return d
    def clear_parks(rid,deg):
        key=(rid,deg)
        if key in park_cache:return park_cache[key]
        clear=[]
        for q in parks[rid]:
            pairs=world.check(rid,q,max_pairs=sys.maxsize)+additional_self_checks(world,rid)
            if not [p for p in pairs if p['asset_name']!='other_robot']:
                clear.append(q)
                if len(clear)>=6:break
        park_cache[key]=clear;return clear
    def gun_pairs(rid,target,weld,environment):
        model=world.models[rid];T=target@np.linalg.inv(model.tcp_home)@model.gun_home;pairs=[]
        for pid,(link,kind,mesh)in parts[rid].items():
            obj=world.robots[rid]['gun:'+pid];obj.setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
            for oid,name,env in environment:
                if kind=='electrode' and oid==weld['product_id']:continue
                res=fcl.CollisionResult()
                if fcl.collide(obj,env,fcl.CollisionRequest(),res):pairs.append(dict(part=pid,asset_id=oid,asset_name=name))
        return pairs
    def stationary_layout_collisions(deg):
        if deg in fixture_angle_reject:return fixture_angle_reject[deg]
        blocking=[]
        # The source rotary parent/child bearing is an assembled joint interface;
        # no exemption is applied to the floor/layout or opposite fixture.
        for obj,label in [(moving_obj,'DONG')]+[(o,pid)for pid,o in product_objs.items()]:
            for oid,name,env in fixed_env:
                if oid==fixed_id:continue
                res=fcl.CollisionResult()
                if fcl.collide(obj,env,fcl.CollisionRequest(),res):blocking.append(dict(part=label,asset_id=oid,asset_name=name))
        fixture_angle_reject[deg]=blocking;return blocking
    for index,(weld,maprow)in enumerate(zip(scene['welds'],inventory['rows'])):
        if index<len(rows):continue
        set_angle(0);_,base=transform_weld(scene,weld,np.eye(4));allowed=[];relative_rejections=[]
        # Invariance: co-rotation of gun target, workpieces and DONG preserves
        # these pair intersections for every external angle.
        for roll in range(0,360,5):
            for sign in (1,-1):
                target=roll_target(base,roll,sign)
                for rid in rids:
                    pairs=gun_pairs(rid,target,weld,relative_env)
                    if pairs:relative_rejections.append(dict(roll_deg=roll,axis_sign=sign,robot_id=rid,pairs=pairs))
                    else:allowed.append(dict(roll_deg=roll,axis_sign=sign,robot_id=rid))
        full_checks=[];angle_trials=[];selected=None;stationary_passed=False;ik_seeds_total=0;pose_clear_count=0
        def full_check(rid,q,deg,roll,sign,moved,target,d):
            nonlocal selected
            model=world.models[rid];other=next(r for r in rids if r!=rid)
            for park in clear_parks(other,deg):
                mesh=products[moved['product_id']].copy();mesh.apply_transform(d)
                state=_check_simultaneous_source_faces(world,robot_id=rid,other_robot_id=other,q=q,other_q=park,
                    driver_mm=0.,weld=moved,phase='closed',source_parts=parts[rid],product_mesh=mesh)
                keys=['gun:'+p for p,info in model.gun_parts.items()if info['kind']=='electrode'];product=product_objs[moved['product_id']]
                gaps={k:float(fcl.distance(world.robots[rid][k],product,fcl.DistanceRequest(),fcl.DistanceResult()))for k in keys if k not in state['certificates']}
                actual=model.tcp(q);error=float(np.linalg.norm(actual[:3,3]-moved['world_xyz']));axis_error=float(np.degrees(np.arccos(np.clip(abs(actual[:3,2]@target[:3,2]),0,1))))
                passed=posture_passed(state,error,keys,gaps)and axis_error<.01
                record=dict(weld_id=weld['id'],robot_id=rid,q_rad=q.tolist(),other_robot_id=other,other_q_rad=park.tolist(),
                    positioner_deg=deg,roll_deg=roll,axis_sign=sign,transformed_weld=moved,target_tcp_matrix=target.ravel().tolist(),
                    actual_tcp_matrix=actual.ravel().tolist(),positioner_delta=d.ravel().tolist(),tcp_error_mm=error,normal_error_deg=axis_error,
                    remaining_pairs=state['remaining_pairs'],certificates=state['certificates'],electrode_gap_mm=gaps,
                    raw_contacts=state['raw_contacts'],passed=passed)
                full_checks.append(record)
                if passed:selected=record;return True
                if any(p['asset_name']!='other_robot'for p in state['remaining_pairs']):break
            return False
        # Phase 1: positioner fixed at source angle, roll strictly on 5-degree grid.
        for deg in angle_grid():
            if selected:break
            d=set_angle(deg);moved,moved_base=transform_weld(scene,weld,d)
            environment_block=stationary_layout_collisions(deg)
            angle_record=dict(positioner_deg=deg,scene_blocking_pairs=environment_block,gun_clear_candidates=0,ik_states=0,ik_seed_starts=0)
            angle_trials.append(angle_record)
            if environment_block:continue
            if not allowed:break
            if deg==0:
                old=prev_by_id[weld['id']].get('selected')
                if old:
                    # Preserve its actual 5-degree roll rather than trusting a stale label.
                    actual=world.models[old['robot_id']].tcp(np.asarray(old['q_rad']));sign=1 if actual[:3,2]@base[:3,2]>0 else -1
                    start=roll_target(base,0,sign);roll=float(np.degrees(np.arctan2((start[:3,:3].T@actual[:3,:3])[1,0],(start[:3,:3].T@actual[:3,:3])[0,0]))%360)
                    if abs(roll/5-round(roll/5))<.001:
                        full_check(old['robot_id'],np.asarray(old['q_rad']),0,round(roll/5)*5%360,sign,moved,actual,d)
                        if selected:stationary_passed=True;break
            for candidate in allowed:
                rid=candidate['robot_id'];roll=candidate['roll_deg'];sign=candidate['axis_sign'];target=roll_target(moved_base,roll,sign)
                fixed_pairs=gun_pairs(rid,target,moved,fixed_env)
                if fixed_pairs:continue
                pose_clear_count+=1;angle_record['gun_clear_candidates']+=1
                model=world.models[rid];sources=[np.asarray(c['q_rad'])for c in maprow['candidates']if c['robot_id']==rid and c['fk_candidate_passed']]
                near=[np.asarray(r['q_rad'])for r in full_checks[-6:]if r['robot_id']==rid]
                solutions,stats=solve_seed_bank(model,target,sources+near,random_count=16)
                angle_record['ik_seed_starts']+=stats['starts'];ik_seeds_total+=stats['starts']
                angle_record['ik_states']+=len(solutions)
                for sol in solutions:
                    if full_check(rid,np.asarray(sol['q_rad']),deg,roll,sign,moved,target,d):break
                if selected:
                    if deg==0:stationary_passed=True
                    break
            print('ANGLE',index+1,weld['id'],deg,'relative rolls',len(allowed),'gun clear',angle_record['gun_clear_candidates'],'IK',angle_record['ik_states'],'PASS',selected is not None,flush=True)
            if deg==0:
                print('STATIONARY',weld['id'],'PASS',stationary_passed,'remaining search',not stationary_passed,flush=True)
        row=dict(weld_id=weld['id'],stationary_passed=stationary_passed,passed=selected is not None,selected=selected,
            relative_gun_candidates=len(allowed),relative_rejections=relative_rejections,angle_trials=angle_trials,
            gun_clear_candidates=pose_clear_count,ik_seed_starts=ik_seeds_total,full_checks=full_checks,
            status='stationary_verified'if stationary_passed else 'positioner_verified'if selected else
                'all_5deg_rolls_blocked_by_co_rotating_geometry'if not allowed else 'not_found_in_5deg_external_axis_and_roll_search')
        rows.append(row)
        result=dict(schema='source_positioner_5deg_static_search_v1',scene_sha256=source_hash,positioner=fixture.metadata,
            mesh_partition=dict(original_triangles=full_triangles,moving_triangles=len(moving.faces),fixed_triangles=len(fixed.faces),
                source_raw='build/fixtureL.json',source_moving_node=2,source_fixed_node=6196),
            angle_step_deg=5,roll_step_deg=5,positioner_angles_deg=angle_grid(),rows=rows,
            summary=dict(completed_welds=len(rows),source_welds=29,stationary_passed=sum(r['stationary_passed']for r in rows),
                additional_positioner_passed=sum(r['passed']and not r['stationary_passed']for r in rows),
                total_passed=sum(r['passed']for r in rows),all_29_passed=len(rows)==29 and all(r['passed']for r in rows),elapsed_s=time.time()-started),
            scope='Static closed-gun configurations only; no external-axis rotation path or robot transit planned; original adjacent positioner bearing interface excluded as assembly joint, same as adjacent arm links.')
        tmp=OUT/'report.inprogress.json';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(report_path)
        print('POINT',index+1,'/29',weld['id'],'PASS',row['passed'],'positioner',selected['positioner_deg']if selected else None,flush=True)
    assert sha(source_path)==source_hash
    print('SUMMARY',json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()
