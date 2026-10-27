#!/usr/bin/env python3
"""Source XML limits and physical-time audit, independent of planner classes."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from audit_counterbalance_fk_v17 import passive_joint_values

ROOT=Path(__file__).resolve().parents[1]


def source_joints(raw):
    text=raw['kinematicsSource'];xml=text[text.index('<devices>'):text.index('</devices>')+10]
    return list(ET.fromstring(xml).find('device/joints'))


def polygon_contains(point,points,tolerance=1e-7):
    p=np.asarray(point);inside=False
    for a,b in zip(points,points[1:]+points[:1]):
        a,b=np.asarray(a),np.asarray(b);d=b-a
        if d@d and np.linalg.norm(p-(a+np.clip((p-a)@d/(d@d),0,1)*d))<=tolerance:return True
        if (a[1]>p[1])!=(b[1]>p[1]) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def limits(q,joints):
    index={j.get('key'):i for i,j in enumerate(joints)};fail=[]
    for i,j in enumerate(joints):
        r=j.find('jointRange');kind=r.findtext('limitType')
        if kind=='1' and not float(r.findtext('lowLimit'))-1e-7<=q[i]<=float(r.findtext('highLimit'))+1e-7:fail.append(j.get('key'))
        if kind=='2':
            points=list(zip([float(x.text)for x in r.find('leadPoints')],[float(x.text)for x in r.find('dependPoints')]))
            if not polygon_contains((q[index[r.findtext('rangeJoint')]],q[i]),points):fail.append(j.get('key'))
    return fail


def rates(values,times,speed_limits,acc_limits,labels):
    values=np.asarray(values,float);dt=np.diff(times);vel=np.diff(values,axis=0)/dt[:,None]
    vtimes=(times[:-1]+times[1:])/2;acc=np.diff(vel,axis=0)/np.diff(vtimes)[:,None]
    maxv=np.max(abs(vel),axis=0);maxa=np.max(abs(acc),axis=0)
    discontinuities=np.argwhere(abs(np.diff(vel,axis=0))>1e-7)
    return {'joint_labels':labels,'maximum_interval_speed':maxv.tolist(),'source_speed_limit':speed_limits,
            'maximum_finite_difference_acceleration':maxa.tolist(),'source_acceleration_limit':acc_limits,
            'speed_exceeded':[labels[i]for i in range(len(labels))if maxv[i]>speed_limits[i]+1e-6],
            'sampled_acceleration_exceeded':[labels[i]for i in range(len(labels))if maxa[i]>acc_limits[i]+1e-6],
            'velocity_discontinuity_count':len(discontinuities),
            'piecewise_linear_has_unbounded_acceleration':bool(len(discontinuities)),
            'acceleration_semantics':'finite differences are sampled diagnostics; nonzero velocity changes at linear interpolation knots imply instantaneous/unbounded acceleration'}


def audit(plan_path,scene_path):
    plan=json.loads(plan_path.read_text());scene=json.loads(scene_path.read_text());frames=plan['frames'];t=np.array([f['time_s']for f in frames]);issues=[]
    if np.any(np.diff(t)<=0):return {'valid_time_axis':False,'issues':['time_nonmonotonic']}
    all_robot_joints=source_joints(json.loads((ROOT/'build/robot.json').read_text()));robot_joints=all_robot_joints[:6]
    robot_index={j.get('key'):i for i,j in enumerate(robot_joints)};robots=[];guns=[]
    for asset in scene['assets']:
        if asset['kind']=='robot':
            q=np.array([f['robots'][asset['id']]['q_rad']for f in frames]);errors=[{'frame':i,'joint_keys':limits(row,robot_joints)}for i,row in enumerate(q)if limits(row,robot_joints)]
            physical=q.copy()
            for i,j in enumerate(robot_joints):
                # Source XML spells the child <factors>, not <factor>. Iterate
                # the container children so source coupling cannot vanish.
                leading=j.find('leadingJoints');factors=j.find('leadingFactors')
                for k,factor in zip(list(leading) if leading is not None else [],list(factors) if factors is not None else []):physical[:,i]+=q[:,robot_index[k.text]]*float(factor.text)
            record={'id':asset['id'],'name':asset['name'],'joint_limit_violations':errors,
                    'rates':rates(physical,t,[float(j.findtext('maxSpeed'))for j in robot_joints],[float(j.findtext('maxAcc'))for j in robot_joints],[j.get('key')for j in robot_joints])}
            passive=np.array([passive_joint_values(row,scene)['values_rad_mm_rad']for row in q])
            record['passive_rates_corrected_closure']=rates(passive,t,[float(j.findtext('maxSpeed'))for j in all_robot_joints[6:9]],[float(j.findtext('maxAcc'))for j in all_robot_joints[6:9]],[j.get('key')for j in all_robot_joints[6:9]])
            record['passive_pose_provenance']='independent corrected orthonormal closure truth; known-invalid archived passive matrices are not used'
            robots.append(record)
        if asset['kind']=='gun':
            raw=json.loads((ROOT/asset['gun']['source_detail']).read_text());joints=source_joints(raw);d=np.array([f['guns'][asset['id']]['driver_mm']for f in frames]);theta=np.arcsin(d/485.)
            length=np.sqrt(123190.999924-121440.952596*np.cos(.686251+theta));rod=length-171.;cylinder=.995203-np.arccos((143882.000258116-121440.952596*np.cos(.686251+theta))/(536.43639*length))
            values=np.column_stack((d,theta,rod,cylinder));lo=float(joints[0].findtext('jointRange/lowLimit'));hi=float(joints[0].findtext('jointRange/highLimit'))
            guns.append({'id':asset['id'],'name':asset['name'],'driver_limit_mm':[lo,hi],
                         'driver_limit_violations':np.flatnonzero((d<lo)|(d>hi)).tolist(),
                         'rates':rates(values,t,[float(j.findtext('maxSpeed'))for j in joints],[float(j.findtext('maxAcc'))for j in joints],[j.get('key')for j in joints])})
    segment=[]
    for i,f in enumerate(frames):
        if not segment or segment[-1]['phase']!=f['phase']:segment.append({'phase':f['phase'],'first_frame':i,'last_frame':i,'weld_id':f.get('weld_id')})
        else:segment[-1]['last_frame']=i
    for s in segment:s['time_start_s']=float(t[s['first_frame']]);s['time_end_s']=float(t[s['last_frame']])
    return {'plan':str(plan_path),'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in ['build/robot.json','build/gunR1_detail.json','build/gunR2_detail.json']},
            'valid_time_axis':True,'frames':len(frames),'duration_s':float(t[-1]-t[0]),'frame_delta_range_s':[float(np.diff(t).min()),float(np.diff(t).max())],
            'robots':robots,'guns':guns,'phases':segment,'motion_measurement_fields':{k:plan[k]for k in plan if any(word in k.lower()for word in ['velocity','acceleration','measurement','time'])},
            'limitations':['No acceleration/jerk controlled time law is present; timestamps alone do not certify dynamic feasibility.',
                           'No welding dwell, electrode force/current, thermal exposure, camera measurement or process quality fields are present.',
                           'Gun dependent equations reproduced numerically from source XML; mesh geometry and collisions are audited separately.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('plans',nargs='+',type=Path);p.add_argument('--scene',type=Path,default=ROOT/'build/scene.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reports=[audit(f.resolve(),a.scene.resolve())for f in a.plans];a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(reports,indent=2));print(json.dumps([{k:r[k]for k in ['plan','frames','duration_s','valid_time_axis']}for r in reports],indent=2))


if __name__=='__main__':main()
