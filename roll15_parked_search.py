"""Independent R1 endpoint roll search with R2 at validated static park."""
import json, math, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from collision_audit import CollisionWorld
from solve_welds import solve
from build_scene import matrix
from r2_safe_park import additional_self_checks

ROOT=Path(__file__).resolve().parent

def main():
    scene=json.loads((ROOT/'build/scene.json').read_text()); world=CollisionWorld(scene)
    rid1=next(a['id'] for a in scene['assets'] if a['kind']=='robot' and 'R1' in a['name']);rid2=next(a['id'] for a in scene['assets'] if a['kind']=='robot' and 'R2' in a['name'])
    k1,k2=world.models[rid1],world.models[rid2]
    park=json.loads((ROOT/'build/r2_safe_park_search.json').read_text())['selected']['q_rad']; park=np.array(park);world.update(rid2,park)
    ik=json.loads((ROOT/'build/ik_source_normal_roll_optimized.json').read_text())
    candidates=[]; by_weld={}
    existing=json.loads((ROOT/'build/collision_plan_source_normal_roll_optimized.json').read_text())['candidates']
    for c in existing:
        if c['robot_id']==rid1:by_weld.setdefault(c['weld_id'],[]).append(c)
    for row in ik['welds']:
        ridrow=next(r for r in row['robots'] if r['robot_id']==rid1)
        if any(c['collision_free_environment'] for c in by_weld.get(row['id'],[])):continue
        source=np.array(row['target_transform']).reshape(4,4)
        # Preserve the alternate source projected normal (column 2), as in the
        # original validated endpoints, and spin X/Y in 15-degree increments.
        base=source.copy();normal=base[:3,2].copy();seed=k1.q_initial if not ridrow['solutions'] else np.array(ridrow['solutions'][0]['q_rad'])
        attempts=[]
        for deg in range(-90,91,15):
            target=base.copy();target[:3,:3]=base[:3,:3] @ Rotation.from_rotvec([0,0,math.radians(deg)]).as_matrix()
            sols=solve(k1,target,seed,4)
            for sol in sols[:1]:
                q=np.array(sol['q_rad']);weld=next(w for w in scene['welds'] if w['id']==row['id'])
                pairs=world.check(rid1,q,weld=weld,gun_driver_mm=0)+additional_self_checks(world,rid1)
                item={'roll_deg':deg,'q_rad':q.tolist(),'position_error_mm':sol['position_error_mm'],'rotation_error_rad':sol['rotation_error_rad'],'joint_limits_valid':bool(k1.limits_valid(q)),'collision_pairs':pairs,'collision_free_both_robots':bool(k1.limits_valid(q) and not pairs)}
                attempts.append(item)
                if item['collision_free_both_robots']:break
            if any(a['collision_free_both_robots'] for a in attempts):break
        best=next((a for a in attempts if a['collision_free_both_robots']),None)
        candidates.append({'weld_id':row['id'],'attempt_count':len(attempts),'selected':best,'attempts':attempts,'normal_world':normal.tolist()})
        print(row['id'],'tries',len(attempts),'selected',None if best is None else best['roll_deg'],flush=True)
    out={'orientation_source':'alternate source projected normal fixed; TCP roll increments 15 degrees; source isProjected=0 retained as alternate interpretation','r2_alternative_park':park.tolist(),'source_to_park_path_verified':False,'collision_verified':False,'candidate_limit':360,'candidate_count':sum(c['attempt_count'] for c in candidates),'welds':candidates,'scene_sha256':hashlib.sha256((ROOT/'build/scene.json').read_bytes()).hexdigest()}
    (ROOT/'build/roll15_parked_search.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('selected',sum(x['selected'] is not None for x in candidates),'of',len(candidates),'attempts',out['candidate_count'])

if __name__=='__main__':main()
