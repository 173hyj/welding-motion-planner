"""Fresh complete-pair audit of selected static states, plus raw-source FK."""
import hashlib,json,sys,importlib.util,argparse
from pathlib import Path
import numpy as np,trimesh,fcl
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'build/eight_hour_v17/experimental_release/multi_point_map')]
from collision_audit import CollisionWorld
from gun_kinematics import gun_mesh_parts
from source_face_world_adapter_v1 import _check_simultaneous_source_faces
from acceptance import posture_passed
spec=importlib.util.spec_from_file_location('source_fk_audit',ROOT/'scripts/audit_contact_release_v17.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
OUT=ROOT/'build/welding_endpoints_fixed_normal_v2'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,default=OUT)
    args=parser.parse_args();out=args.folder.resolve()
    scene_path=ROOT/'build/scene_placement_corrected_v1.json'
    scene=json.loads(scene_path.read_text());report_path=out/'report.json'
    report=json.loads(report_path.read_text());assert len(report['rows'])==29
    assert hashlib.sha256(scene_path.read_bytes()).hexdigest()==report['scene_sha256']
    world=CollisionWorld(scene);parts={};products={};rows=[]
    joints=audit.joints_from_raw(json.loads((ROOT/'build/robot.json').read_text()))[:6]
    for rid,model in world.models.items():
        asset=next(a for a in scene['assets'] if a['id']==model.gun_id)
        _,parts[rid]=gun_mesh_parts(json.loads((ROOT/asset['gun']['source_detail']).read_text()))
    for row in report['rows']:
        if not row['passed']:continue
        a=row['selected'];rid=a['robot_id'];model=world.models[rid];q=np.asarray(a['q_rad'])
        weld=next(w for w in scene['welds'] if w['id']==row['weld_id']);pid=weld['product_id']
        if pid not in products:
            asset=next(a for a in scene['assets'] if a['id']==pid)
            mesh=trimesh.load(ROOT/'build'/asset['url'],force='mesh',process=False)
            mesh.apply_transform(np.asarray(asset['transform']).reshape(4,4));products[pid]=mesh
        # Rodrigues screw-axis FK parsed independently from raw original robot.
        independent_pose=model.base@audit.flange_source(q,joints)@model.tcp_home
        fk_difference=float(np.max(np.abs(independent_pose-model.tcp(q))))
        assert fk_difference<1e-7
        error=float(np.linalg.norm(independent_pose[:3,3]-weld['world_xyz']))
        angle=float(np.degrees(np.arccos(np.clip(abs(independent_pose[:3,2]@np.asarray(row['normal_world'])),0,1))))
        state=_check_simultaneous_source_faces(world,robot_id=rid,other_robot_id=a['other_robot_id'],q=q,
            other_q=np.asarray(a['other_q_rad']),driver_mm=0.,weld=weld,phase='closed',source_parts=parts[rid],product_mesh=products[pid])
        keys=['gun:'+key for key,value in model.gun_parts.items() if value['kind']=='electrode']
        product=next(obj for oid,name,obj in world.static if oid==pid)
        gaps={key:float(fcl.distance(world.robots[rid][key],product,fcl.DistanceRequest(),fcl.DistanceResult())) for key in keys if key not in state['certificates']}
        passed=posture_passed(state,error,keys,gaps) and angle<.01
        assert passed,(row['weld_id'],state['remaining_pairs'])
        rows.append(dict(weld_id=row['weld_id'],passed=passed,tcp_error_mm=error,normal_error_deg=angle,
            independent_fk_difference_mm=fk_difference,remaining_pairs=state['remaining_pairs'],certificates=state['certificates']))
        print('VERIFIED',row['weld_id'],'TCP',error,'normal',angle,flush=True)
    output=dict(selected_count=len(rows),all_selected_static_states_passed=bool(rows) and all(r['passed'] for r in rows),all_29_passed=len(rows)==29 and all(r['passed'] for r in rows),
        scene_sha256=report['scene_sha256'],report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),rows=rows,
        scope='Independent raw-source arm FK and fresh full original-pair static FCL recheck; no motion certification')
    (out/'selected_independent_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print('VERIFICATION',len(rows),'passed',flush=True)
if __name__=='__main__':main()
