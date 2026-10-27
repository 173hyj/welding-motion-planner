"""All 784 fixed-condition rows freshly certified for 300 mm, plus 21 seeds."""
import argparse, concurrent.futures, copy, json, multiprocessing, sys, time
from pathlib import Path
import numpy as np
from experimental.welding_full_candidate_selection_v1.pool import atomic,digest,seal
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/welding_retreat_transfer_v1'),str(ROOT/'experimental/welding_variable_open_v1'),str(ROOT/'experimental/welding_tool_exit_v1')]
ENGINE=OUT=SETTINGS=None
LENGTH_MM=300.

def build_registry(rows):
 result=[]
 for row in rows:
  item=copy.deepcopy(row)
  item['previous_status']=item.pop('status');item['status']='pending';item['attempt_count']=0
  item.pop('action',None);result.append(item)
 return result

def exact_identity(a,b):
 return all(a.get(k)==b.get(k) for k in ('robot_id','roll_deg','axis_sign','q_rad','positioner_deg'))

def ray_spec(direction):
 if not np.isclose(np.linalg.norm(direction),1.,atol=1e-8):raise ValueError('unit ray required')
 return dict(direction_world=list(direction),direction_frame='current_world',length_mm=LENGTH_MM)

def evaluate(job):
 row,reference=job;row=copy.deepcopy(row);row['attempt_count']+=1;started=time.perf_counter()
 from integrated_retreat_v117 import lift_path,check_path_contract
 from run_direct import geometry_prefix
 from .search import try_configuration
 try:
  state=row['catalog_state'];rid=state['robot_id'];source=json.loads(Path(reference['path_file']).read_text())
  if digest(reference['path_file'])!=reference['path_sha256']:raise ValueError('reference changed')
  ENGINE.set_angle(row['angle']);weld=next(w for w in ENGINE.scene['welds'] if w['id']==row['weld_id'])
  candidate={k:state[k] for k in ('robot_id','roll_deg','axis_sign')};q=np.asarray(state['q_rad']);model=ENGINE.world.models[rid]
  moved,target=ENGINE.target(weld,candidate);start=model.tcp(q)
  if not model.limits_valid(q) or np.max(abs(start-target))>1e-6:
   row.update(status='proven_invalid',reason='raw_source_pose_or_limit_invalid')
  else:
   selected=dict(candidate,q_rad=state['q_rad'],other_robot_id=source['other_robot_id'],other_q_rad=source['other_q_rad'],positioner_deg=row['angle'],target_tcp_matrix=target.ravel().tolist(),actual_tcp_matrix=start.ravel().tolist())
   driver=source['driver_mm'];ray=ray_spec(source['direction_world'])
   geometry=geometry_prefix(ENGINE,rid,start,moved,driver,np.asarray(ray['direction_world']),LENGTH_MM)
   path=lift_path(ENGINE,weld,candidate,selected,ray,driver_mm=driver) if geometry['passed'] else dict(passed=False,reason='reference_ray_geometry_failed')
   row['reference_attempt']=dict(passed=path['passed'],reason=path.get('reason'))
   if path['passed']:path['geometry_certificate']=geometry
   else:
    settings=dict(angle_deg=row['angle'],deadline=time.perf_counter()+SETTINGS['row_seconds'],directions=16,amplitudes=list(dict.fromkeys([abs(driver),10.,30.,60.,80.,120.,140.,180.])))
    fallback=try_configuration(ENGINE,weld,candidate,q,{selected['other_robot_id']:[selected['other_q_rad']]},settings,{})
    row['fallback_attempt']=dict(passed=fallback['passed'],reason=fallback.get('reason'),ray_trials=len(fallback.get('attempts',[])))
    if fallback['passed']:path=fallback['path'];selected=fallback['selected']
   if path['passed']:
    if not exact_identity(selected,dict(state,positioner_deg=row['angle'])):raise ValueError('catalog identity changed')
    path['weld_id']=row['weld_id'];check_path_contract(path,selected,LENGTH_MM)
    file=OUT/'paths'/(row['id']+'.json');atomic(file,path)
    action=dict(id=row['id'],source='fresh300',initial=selected,catalog_state=state,catalog_rowid=row.get('catalog_rowid'),path_file=str(file),path_sha256=digest(file),length_mm=LENGTH_MM)
    row.update(status='certified',action=action,reason='fresh_300mm_source_action')
   else:row.update(status='search_unresolved',reason=path.get('reason','finite_search_failed'))
 except Exception as exc:row.update(status='search_unresolved',reason='evaluation_exception',error=repr(exc))
 row.update(elapsed_s=time.perf_counter()-started,search_exhaustive=False,length_mm=LENGTH_MM)
 atomic(OUT/'checkpoints'/(row['id']+'.json'),row)
 print('ROW300',row['id'],row['task'],row['status'],round(row['elapsed_s'],2),row.get('reason'),flush=True)
 return row

def main():
 global ENGINE,OUT,SETTINGS
 p=argparse.ArgumentParser();p.add_argument('--previous-pool',type=Path,required=True);p.add_argument('--seeds',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--row-seconds',type=float,default=20.)
 a=p.parse_args();OUT=a.out.resolve();SETTINGS=dict(row_seconds=a.row_seconds,length_mm=LENGTH_MM,workers=a.workers)
 if a.workers<1 or not np.isfinite(a.row_seconds) or a.row_seconds<=0:raise ValueError('positive finite budget/workers required')
 old=json.loads(a.previous_pool.read_text());seed_report=json.loads(a.seeds.read_text());rows=build_registry(old['registry'])
 if not seed_report.get('complete') or seed_report.get('source_assets_frozen_before_and_after') is not True:raise ValueError('sealed complete seed generation required')
 if len(rows)!=784 or len(old['layers'])!=21:raise ValueError('Expected original fixed-condition 784-row scope')
 bindings=dict(old['manifest']['bindings']);seal(bindings)
 def bind(name,sha=None):
  name=str(Path(name).resolve());sha=sha or digest(name)
  if name in bindings and bindings[name]!=sha:raise ValueError('binding conflict '+name)
  bindings[name]=sha
 bind(a.previous_pool);bind(a.seeds)
 for name,sha in seed_report['manifest']['bindings'].items():bind(name,sha)
 for file in Path(__file__).parent.glob('*.py'):bind(file)
 from run_initial import extended_source_hashes,assert_frozen_geometry_cache
 assert_frozen_geometry_cache();before=extended_source_hashes()
 for name,sha in before.items():bind(ROOT/name,sha)
 seeds={r['task']:r for r in seed_report['rows'] if r['length_mm']==300. and r['passed']}
 if len(seeds)!=21:raise ValueError('21 certified 300 mm incumbent seeds required')
 from integrated_retreat_v117 import check_path_contract
 layers=[]
 for layer in old['layers']:
  seed=seeds[layer['task']];bind(seed['path_file'],seed['path_sha256']);bind(seed['source_path'],seed['source_sha256']);path=json.loads(Path(seed['path_file']).read_text());check_path_contract(path,seed['initial'],LENGTH_MM)
  original=json.loads(Path(seed['source_path']).read_text())
  ray_spec(path['direction_world']);ray_spec(original['direction_world'])
  if not np.allclose(path['direction_world'],original['direction_world'],atol=1e-12,rtol=0) or np.dot(path['direction_world'],original['direction_world'])<=0 or path['driver_mm']!=original['driver_mm']:raise ValueError('Seed ray/opening changed')
  expected=next((x['initial'] for x in layer['actions'] if exact_identity(x['initial'],seed['initial'])),None)
  if expected is None:raise ValueError('Seed absent from prior layer')
  action=dict(id='incumbent300-'+str(layer['task']),source='certified300_seed',initial=seed['initial'],path_file=seed['path_file'],path_sha256=seed['path_sha256'],length_mm=LENGTH_MM)
  layers.append(dict(layer,actions=[action]))
 manifest=dict(bindings=bindings,settings=SETTINGS,seed_normalized_ray_tolerance=1e-12,fixed_conditions=old['manifest']['fixed_conditions'],catalog_path=old['manifest']['catalog_path'],source_sha256=before,scope='all original 784 fixed-angle assigned-arm rows; no candidate truncation; 21 certified300 incumbent seeds')
 OUT.mkdir(parents=True,exist_ok=True);mf=OUT/'manifest.json'
 if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Resume manifest differs')
 atomic(mf,manifest);atomic(OUT/'source_before.json',dict(source_sha256=before));seal(bindings)
 lookup={r['id']:i for i,r in enumerate(rows)}
 for i,row in enumerate(rows):
  file=OUT/'checkpoints'/(row['id']+'.json')
  if file.exists():
   saved=json.loads(file.read_text())
   if any(saved.get(k)!=row.get(k) for k in ('id','catalog_rowid','catalog_key','payload_sha256','task','catalog_state')) or saved['length_mm']!=LENGTH_MM:raise ValueError('Checkpoint identity differs')
   if saved['status']=='certified':
    ac=saved['action'];seal({ac['path_file']:ac['path_sha256']});check_path_contract(json.loads(Path(ac['path_file']).read_text()),ac['initial'],LENGTH_MM)
   rows[i]=saved
 def report(complete=False):
  current=copy.deepcopy(layers)
  for row in rows:
   if row['status']=='certified':current[row['task']-1]['actions'].append(row['action'])
  counts={s:sum(r['status']==s for r in rows) for s in ('pending','certified','search_unresolved','proven_invalid')}
  value=dict(complete=complete,accepted=complete and counts['pending']==0 and all(x['actions'] for x in current),coverage_all_rows=dict(registered=784,attempted=784-counts['pending'],**counts),registry=rows,layers=current,robot_ids=old['robot_ids'],manifest=manifest,length_mm=LENGTH_MM,no_hardware=True,source_assets_frozen_before_and_after=complete,exhaustive_feasibility_search=False)
  atomic(OUT/'report.json',value);return value
 report();queue=[]
 for row in rows:
  if row['status']!='pending':continue
  previous=old['registry'][lookup[row['id']]]
  ref=previous.get('action',old['layers'][row['task']-1]['actions'][0]);queue.append((row,ref))
 if queue:
  from run_geometry import initialize_worker
  from run_integrated_retreat_v118 import load_raw_world
  from mixed_bridge_v95 import SIGNS
  initialize_worker();print('RAW_WORLD_LOADING',flush=True);ENGINE=load_raw_world(json.loads((ROOT/'build/scene_placement_corrected_v1.json').read_text()),SIGNS)
  for rid,model in list(ENGINE.world.models.items()):ENGINE.world.models[rid]=getattr(model,'source',model)
  print('RAW_WORLD_READY',len(queue),flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker) as executor:
   futures=[executor.submit(evaluate,job) for job in queue]
   for f in concurrent.futures.as_completed(futures):
    row=f.result();rows[lookup[row['id']]]=row;report()
 assert_frozen_geometry_cache()
 if before!=extended_source_hashes():raise ValueError('Source assets changed')
 seal(bindings);atomic(OUT/'source_after.json',dict(source_sha256=extended_source_hashes(),unchanged=True));value=report(True);print('LIBRARY300_COMPLETE',value['coverage_all_rows'],flush=True)

if __name__=='__main__':main()
