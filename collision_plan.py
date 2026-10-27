import json
import numpy as np
import argparse
from pathlib import Path
from collision_audit import CollisionWorld
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--orientation',default='projected');args=parser.parse_args()
scene=json.loads((ROOT/'build/scene.json').read_text());ik=json.loads((ROOT/f'build/ik_{args.orientation}.json').read_text())
world=CollisionWorld(scene);out=[]
for w in ik['welds']:
 source_weld=next(x for x in scene['welds'] if x['id']==w['id'])
 for rr in w['robots']:
  for sol in rr['solutions']:
   pairs=world.check(rr['robot_id'],sol['q_rad'],weld=source_weld)
   legal=[]; env=[]
   for p in pairs:
    if p['link']=='gun' and p.get('asset_id')==source_weld['product_id'] and p.get('contact_xyz') is not None:
     d=float(np.linalg.norm(np.array(p['contact_xyz'])-np.array(source_weld['world_xyz'])))
     if d <= 2.0:
      legal.append(dict(p, weld_contact_error_mm=d))
      continue
    env.append(p)
   out.append({'weld_id':w['id'],'robot_id':rr['robot_id'],'q_rad':sol['q_rad'],'legal_weld_contacts':legal,'collision_pairs':pairs,'environment_collision_pairs':env,'collision_free_environment':not env})
print('candidates',len(out),'envfree',sum(x['collision_free_environment'] for x in out))
report={'tcp_verified':True,'orientation_source':args.orientation,'ik_candidates':len(out),'environment_collision_free_candidates':sum(x['collision_free_environment'] for x in out),'collision_verified':False,'reason':'source robot model reports two self-pairs at current pose; candidate endpoint validation does not establish a safe transit from this pose','candidates':out}
(ROOT/f'build/collision_plan_{args.orientation}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
