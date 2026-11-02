"""A weld is accepted only with a complete online-validated straight retreat."""
import numpy as np


def opening_prefix_valid(probe, amplitude):
    # The same half-mm opening grid as v117. This is reject-only: every
    # successful complete action still executes the authoritative v117 gate.
    for driver in -.5*np.arange(1,int(round(2*amplitude))+1):
        if not probe(float(driver)):
            return False
    return True


def first_complete(candidates,evaluate):
    records=[]
    for candidate in candidates:
        result=evaluate(candidate);records.append(result)
        if result.get('passed'):
            if not result.get('path') or not result['path'].get('passed'):
                raise ValueError('Complete action requires a passed retreat path')
            return result,records
    return None,records


def jobs_for_stage(rows,stage):
    if stage=='stationary':return [i for i,r in enumerate(rows) if r['relative_gun_candidates'] and not r['stationary_done']]
    if stage!='positioner':raise ValueError(stage)
    if any(r['relative_gun_candidates'] and not r['stationary_done'] for r in rows):raise ValueError('Stationary complete-action barrier required')
    return [i for i,r in enumerate(rows) if r['relative_gun_candidates'] and not r['passed'] and not r.get('positioner_done')]


def positioner_angles():
    priority=[sign*a for a in (30,60,90,120,150) for sign in (1,-1)]+[180]
    rest=[sign*a for a in range(5,180,5) for sign in (1,-1)]
    return priority+[a for a in rest if a not in priority]


def coarse_kinematic(model,start,q,direction,length=300.):
    """Reject only: sparse continuation never accepts a final path."""
    from analytic_ik_v2 import analytical_continuous_ik
    previous=np.asarray(q)
    for s in np.linspace(0.,length,21)[1:]:
        target=start.copy();target[:3,3]+=s*direction
        fit=analytical_continuous_ik(model,target,previous,max_joint_change_deg=35.)
        if not fit['passed']:return False
        previous=np.asarray(fit['q_rad'])
    return True


def try_configuration(engine,weld,candidate,q,parks,settings,cache):
    import time
    started=time.perf_counter()
    if started>=settings.get('deadline',float('inf')):return dict(passed=False,closed_passed=False,reason='cooperative_budget_expired',attempts=[],path=None)
    from run_arm import full_probe
    from analytic_ik_v2 import analytical_seed_bank
    from probe_near_retreat_v72 import release_neighbors
    from run_direct import geometry_prefix
    import integrated_retreat_v117 as online
    rid=candidate['robot_id'];other=next(r for r in engine.world.models if r!=rid)
    model=engine.world.models[rid];moved,target=engine.target(weld,candidate);q=np.asarray(q);start=model.tcp(q)
    result=dict(passed=False,closed_passed=False,candidate=candidate,q_rad=q.tolist(),attempts=[],path=None)
    if not model.limits_valid(q) or np.max(abs(start-target))>1e-6:
        result['reason']='invalid_source_weld_pose';return result
    for park in parks[other]:
        if time.perf_counter()>=settings.get('deadline',float('inf')):result['reason']='cooperative_budget_expired';return result
        park=np.asarray(park)
        check=full_probe(engine,rid,q,other,park,0.,moved,'closed')
        caps=[pid for pid,(_,kind,_) in engine.parts[rid].items() if kind=='electrode']
        if not caps or not check['passed'] or not all(check.get('certificates',{}).get(pid,{}).get('passed',False) for pid in caps):continue
        result['closed_passed']=True
        selected=dict(candidate,q_rad=q.tolist(),other_robot_id=other,other_q_rad=park.tolist(),
            positioner_deg=settings['angle_deg'],target_tcp_matrix=target.ravel().tolist(),actual_tcp_matrix=start.ravel().tolist())
        if not opening_prefix_valid(lambda driver:full_probe(engine,rid,q,other,park,driver,moved,'opening')['passed'],min(settings['amplitudes'])):
            result['opening_prefix_rejected_parks']=result.get('opening_prefix_rejected_parks',0)+1
            continue
        for amplitude in settings['amplitudes']:
            if time.perf_counter()>=settings.get('deadline',float('inf')):result['reason']='cooperative_budget_expired';return result
            driver=-float(amplitude);key=('open',rid,candidate['roll_deg'],candidate['axis_sign'],driver)
            if key not in cache:
                opened=engine.state(rid,start,driver,moved,engine.world.static,'opening')
                cache[key]=opened
            else:opened=cache[key]
            if opened['blocking_pairs']:continue
            context=engine.release_context(rid,start,moved,dict(open_state=opened),driver_mm=driver)
            reference=np.asarray(context['preferred_outward_normal'])
            bank=diverse_directions(release_neighbors(start,reference,context['supports'],2522),settings['directions'])
            for direction in bank:
                if time.perf_counter()>=settings.get('deadline',float('inf')):result['reason']='cooperative_budget_expired';return result
                trial=dict(amplitude_mm=amplitude,direction_world=direction.tolist());result['attempts'].append(trial)
                end=start.copy();end[:3,3]+=300.*direction
                ends,_=analytical_seed_bank(model,end,[q])
                if not ends:trial['reason']='endpoint_ik_rejected';continue
                if not coarse_kinematic(model,start,q,direction):trial['reason']='coarse_kinematic_rejected';continue
                key=('geom',rid,candidate['roll_deg'],candidate['axis_sign'],driver,tuple(direction))
                if key not in cache:
                    blocked=False
                    for s in np.arange(10.,300.1,25.):
                        pose=start.copy();pose[:3,3]+=s*direction
                        if engine.state(rid,pose,driver,moved,engine.world.static,'separated')['blocking_pairs']:blocked=True;break
                    cache[key]=dict(passed=False,reason='coarse_tool_collision') if blocked else geometry_prefix(engine,rid,start,moved,driver,direction,300.)
                geometry=cache[key]
                if not geometry['passed']:trial['reason']=geometry.get('reason', 'geometry_certificate_failed');continue
                if not any(full_probe(engine,rid,np.asarray(f['q_rad']),other,park,driver,moved,'separated')['passed'] for f in ends):
                    trial['reason']='endpoint_source_collision';continue
                path=online.lift_path(engine,weld,selected,selected,dict(direction_world=direction.tolist(),direction_frame='current_world',length_mm=300.),driver_mm=driver)
                trial.update(reason=path.get('reason','passed'),passed=path['passed'],last_accepted_travel_mm=path.get('last_accepted_travel_mm'))
                if path['passed']:
                    path.update(weld_id=weld['id'],geometry_certificate=geometry)
                    online.check_path_contract(path,selected,300.)
                    result.update(passed=True,path=path,selected=selected,elapsed_s=time.perf_counter()-started)
                    return result
    result.update(reason='no_complete_action_in_configuration_budget',elapsed_s=time.perf_counter()-started)
    return result


def asset_mesh_paths(scene,root):
    paths=[]
    for asset in scene['assets']:
        if asset.get('url'):paths.append(root/'build'/asset['url'])
        for kind in ('gun','robot'):
            if kind in asset:
                if asset[kind].get('source_detail'):paths.append(root/asset[kind]['source_detail'])
                paths.extend(root/'build'/part['url'] for part in asset[kind].get('link_models',{}).values())
    return paths


def diverse_directions(bank,count):
    bank=np.asarray(bank)
    if not len(bank):return bank.reshape((-1,3))
    order=[0];available=np.ones(len(bank),bool);available[0]=False;nearest=1.-bank@bank[0]
    while len(order)<min(count,len(bank)):
        j=int(np.argmax(np.where(available,nearest,-np.inf)));order.append(j);available[j]=False;nearest=np.minimum(nearest,1.-bank@bank[j])
    return bank[order]
