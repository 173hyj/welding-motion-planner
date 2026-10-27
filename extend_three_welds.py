"""Extend frozen two-weld alternative simulation to source SUB1 point 8."""
import copy
import json
from pathlib import Path

import numpy as np

from collision_audit import CollisionWorld
from r2_safe_park import additional_self_checks
from solve_welds import solve

ROOT = Path(__file__).resolve().parent


def main():
    scene = json.loads((ROOT / 'build/scene.json').read_text())
    plan = json.loads((ROOT / 'build/welding_plan_both_alternative.json').read_text())
    world = CollisionWorld(scene)
    rid1 = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R1' in a['name'])
    rid2 = next(a['id'] for a in scene['assets'] if a['kind'] == 'robot' and 'R2' in a['name'])
    k1,k2 = world.models[rid1],world.models[rid2]
    park = np.array(plan['frames'][0]['robots'][rid2]['q_rad'])
    world.update(rid2,park)
    q6 = np.array(plan['frames'][-1]['robots'][rid1]['q_rad'])
    candidates = json.loads((ROOT / 'build/collision_plan_source_normal_roll_optimized.json').read_text())['candidates']
    endpoint8 = [c for c in candidates if c['weld_id'] == 'OP10L-SUB1-POINT-GW-8' and c['robot_id'] == rid1 and c['collision_free_environment']]
    q8 = min((np.array(c['q_rad']) for c in endpoint8), key=lambda q:np.linalg.norm(q-q6))
    weld6 = next(w for w in scene['welds'] if w['id'] == 'OP10L-SUB1-POINT-GW-6')
    weld8 = next(w for w in scene['welds'] if w['id'] == 'OP10L-SUB1-POINT-GW-8')
    retreat6 = np.array(next(s for s in plan['segments'] if s['phase']=='approach')['samples'][0]['q_rad'])
    attempts=[]
    pose8=k1.tcp(q8)
    target=pose8.copy();target[:3,3]-=100*pose8[:3,2]
    solutions=solve(k1,target,q8,6)
    if not solutions:
        raise RuntimeError('No original-limit point8 retreat inverse solution')
    retreat8=np.array(solutions[0]['q_rad'])
    for turn in (-2*np.pi,0.,2*np.pi):
        chart=retreat8.copy();chart[5]+=turn
        if all(k1.limits_valid(q8+t*(chart-q8)) for t in np.linspace(0,1,21)) and np.linalg.norm(chart-q8)<np.linalg.norm(retreat8-q8):
            retreat8=chart

    def audit(phase,qa,qb,da,db,weld):
        samples,failures,frames=[],[],[]
        for i,t in enumerate(np.linspace(0,1,21)):
            q=qa+t*(qb-qa);driver=da+t*(db-da)
            pairs1=world.check(rid1,q,weld=weld,gun_driver_mm=driver)+additional_self_checks(world,rid1)
            pairs2=world.check(rid2,park)+additional_self_checks(world,rid2)
            valid=k1.limits_valid(q) and k2.limits_valid(park) and not pairs1 and not pairs2
            if not valid:failures.append({'sample':i,'R1_pairs':pairs1,'R2_pairs':pairs2,'limits_valid':k1.limits_valid(q)})
            sample={'q_rad':q.tolist(),'driver_mm':float(driver),'joint_limits_valid':bool(k1.limits_valid(q)),'collision_verified':bool(valid),'collision_pairs':pairs1+pairs2};samples.append(sample)
            frame=copy.deepcopy(plan['frames'][-1]);frame.update(time_s=plan['frames'][-1]['time_s']+.1*(len(attempts)*21+i+1),phase=phase,weld_id=weld['id'],collision_verified=bool(valid),collision_pairs=pairs1+pairs2)
            links=k1.links(q,world=True);frame['robots'][rid1]={'q_rad':q.tolist(),'links':{key:m.flatten().tolist() for key,m in links.items()},'tcp_xyz':k1.tcp(q)[:3,3].tolist()}
            pose=links['245']@k1.gun_home;deltas=k1.gun_model.transforms(driver)
            frame['guns'][k1.gun_id]={'transform':pose.flatten().tolist(),'driver_mm':float(driver),'parts':{pid:(pose@deltas[p['link_key']]).flatten().tolist() for pid,p in k1.gun_parts.items()}}
            frames.append(frame)
        result={'phase':phase,'weld_id':weld['id'],'valid':not failures,'sample_count':21,'failure_count':len(failures),'failures':failures,'samples':samples,'gun_driver_start_mm':da,'gun_driver_end_mm':db}
        attempts.append(result);print('extend',phase,weld['id'],'valid',not failures,flush=True)
        return frames

    extension=[]
    extension+=audit('open',q6,q6,0.,-185.,weld6)
    extension+=audit('retreat',q6,retreat6,-185.,-185.,weld6)
    extension+=audit('transfer',retreat6,retreat8,-185.,-185.,weld8)
    extension+=audit('approach',retreat8,q8,-185.,-185.,weld8)
    extension+=audit('close',q8,q8,-185.,0.,weld8)
    valid=all(s['valid'] for s in attempts)
    plan['frames']+=extension;plan['segments']+=attempts
    plan['selected_weld_ids'].append(weld8['id'])
    plan['extension_valid']=valid;plan['extension_attempts']=attempts
    plan['status'].update(local_sequence_collision_verified=valid,both_robots_sampled_collision_verified=valid,transit_collision_verified=valid,collision_verified=False,system_safe=False,diagnostic_only=not valid,reachable_welds=3 if valid else 2,planned_welds=3 if valid else 2)
    plan['status']['note']='3/29 原始焊点（7→6→8）的局部仿真路线，R2采用替代停放姿态；原起点到替代起点、枪内装配与连续扫掠未认证。' if valid else '两点路线已验证，第三点扩展存在失败采样，不能宣称整段三点路线通过。'
    plan['status']['reason']='Original projected-normal alternative source poses and exact TxKin gun parts. Both robot simultaneous sampled checks reported independently of whole-system safety.'
    (ROOT / 'build/welding_plan_three_alternative.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    print('three-weld extension valid',valid,'frames',len(plan['frames']),flush=True)


if __name__=='__main__':
    main()
