#!/usr/bin/env python3
"""Independent closure-pin reconstruction for source robot counterbalance links.

The archived plan stores 246/247 matrices, but this audit rebuilds them from
source joint 473/475 pin positions and the source 240 local frame. It does not
call production RobotKinematics. The corrected local-frame angle prevents the
base joint rotation from being counted twice.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]

def axis_matrix(point,axis,value):
    r=Rotation.from_rotvec(np.asarray(axis)/np.linalg.norm(axis)*value).as_matrix();t=np.eye(4);t[:3,:3]=r;t[:3,3]=point-r@point;return t

def links_240_241(q,scene):
    t=np.eye(4);out={'239':t.copy()}
    joints=scene['joints'][:6]
    idx={j['key']:i for i,j in enumerate(joints)}
    for i,j in enumerate(joints):
        p=np.array(j['axis1']);axis=np.array(j['axis2'])-p;v=q[i]
        v+=sum(q[idx[k]]*float(f)for k,f in zip(j['leadingJoints'],j['leadingFactors']))
        t=t@axis_matrix(p,axis,v);out[str(j['childLink'])]=t.copy()
    return out

def counterbalance(q,scene):
    r=links_240_241(np.asarray(q),scene);a=np.array(scene['joints'][6]['axis1']);b=np.array(scene['joints'][8]['axis1']);
    pa=r['241'][:3,:3]@a+r['241'][:3,3];pb=r['240'][:3,:3]@b+r['240'][:3,3]
    dlocal=r['240'][:3,:3].T@(pa-pb);home=a-b
    # Vector-frame alignment is independent of the production angle formula.
    ey=np.array([0.,1.,0.]);hu=home/np.linalg.norm(home);du=dlocal/np.linalg.norm(dlocal)
    hframe=np.column_stack((hu,ey,np.cross(hu,ey)));dframe=np.column_stack((du,ey,np.cross(du,ey)))
    ori=r['240'][:3,:3]@dframe@hframe.T;out={}
    for key,p,now in (('246',a,pa),('247',b,pb)):
        t=np.eye(4);t[:3,:3]=ori;t[:3,3]=now-ori@p;out[key]=t
    return r,out,pa,pb

def passive_joint_values(q,scene):
    """Signed source joint values, derived from independently closed poses.

    The neutral reference is source joint q2=0 (not q_initial). Joint474's
    axis points from the cylinder toward its rod, opposite the home pin span.
    Thus extension is positive ||current span|| - ||home span||.
    """
    main,passive,pa,pb=counterbalance(q,scene);values=[];residuals={}
    for joint,parent,child in ((scene['joints'][6],main['241'],passive['246']),(scene['joints'][7],passive['246'],passive['247']),(scene['joints'][8],main['240'],passive['247'])):
        relative=np.linalg.inv(parent)@child;axis=np.array(joint['axis2'])-np.array(joint['axis1']);axis/=np.linalg.norm(axis)
        if joint['type']==0:values.append(float(Rotation.from_matrix(relative[:3,:3]).as_rotvec()@axis))
        else:
            value=float(relative[:3,3]@axis);values.append(value);residuals['prismatic_translation_perpendicular_mm']=float(np.linalg.norm(relative[:3,3]-axis*value))
    return {'joint_keys':['473','474','475'],'values_rad_mm_rad':values,'source_neutral_reference':'all source driver values zero; not q_initial','residuals':residuals}

def audit(plan_path,scene_path):
    scene=json.loads(scene_path.read_text());plan=json.loads(plan_path.read_text());robots=[a for a in scene['assets']if a['kind']=='robot'];rows=[]
    for robot in robots:
        rid=robot['id'];errors=[];pin_errors=[];base_errors=[];base=np.array(robot['transform']).reshape(4,4);corrected_pin_errors=[];axis_errors=[]
        for i,frame in enumerate(plan['frames']):
            q=frame['robots'][rid]['q_rad'];_,truth,pa,pb=counterbalance(q,scene);arch=frame['robots'][rid]['links'];a246=np.linalg.inv(base)@np.array(arch['246']).reshape(4,4);a247=np.linalg.inv(base)@np.array(arch['247']).reshape(4,4);errors.append({'frame':i,'max_matrix_error_246':float(np.max(abs(a246-truth['246']))),'max_matrix_error_247':float(np.max(abs(a247-truth['247'])))});pin_errors.append({'frame':i,'pin473_mm':float(np.linalg.norm(a246[:3,:3]@np.array(scene['joints'][6]['axis1'])+a246[:3,3]-pa)),'pin475_mm':float(np.linalg.norm(a247[:3,:3]@np.array(scene['joints'][8]['axis1'])+a247[:3,3]-pb))});base_errors.append({'frame':i,'local_closure_mm':float(np.linalg.norm((truth['246'][:3,:3].T@(pa-pb))-(truth['247'][:3,:3].T@(pa-pb))))})
            corrected_pin_errors.append([float(np.linalg.norm(truth['246'][:3,:3]@np.array(scene['joints'][6]['axis1'])+truth['246'][:3,3]-pa)),float(np.linalg.norm(truth['247'][:3,:3]@np.array(scene['joints'][8]['axis1'])+truth['247'][:3,3]-pb))])
            source_axis=np.array(scene['joints'][7]['axis2'])-np.array(scene['joints'][7]['axis1']);transformed_axis=truth['246'][:3,:3]@source_axis;axis_errors.append(float(np.linalg.norm(np.cross(transformed_axis/np.linalg.norm(transformed_axis),(pa-pb)/np.linalg.norm(pa-pb)))))
        rows.append({'robot_id':rid,'archived_frames':len(errors),'maximum_archived_vs_corrected_matrix_error_246':max(x['max_matrix_error_246']for x in errors),'maximum_archived_vs_corrected_matrix_error_247':max(x['max_matrix_error_247']for x in errors),'maximum_corrected_pin473_residual_mm':max(x[0]for x in corrected_pin_errors),'maximum_corrected_pin475_residual_mm':max(x[1]for x in corrected_pin_errors),'maximum_prismatic_axis_cross_sine':max(axis_errors),'maximum_corrected_local_closure_residual_mm':max(x['local_closure_mm']for x in base_errors),'archived_pin473_max_residual_mm':max(x['pin473_mm']for x in pin_errors),'archived_vs_corrected_samples':errors[::max(1,len(errors)//5)]})
    return {'scope':'source joints 473/475 closure independently rebuilt in parent-240 local frame','scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'robots':rows,'system_safe':False,'limitations':['This is a kinematic truth audit only; it does not certify collisions or dynamics.','Archived 246/247 poses are diagnosed, not silently replaced in any production artifact.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--scene',type=Path,default=ROOT/'build/scene.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.plan.resolve(),a.scene.resolve());a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r['robots'],indent=2))
if __name__=='__main__':main()
