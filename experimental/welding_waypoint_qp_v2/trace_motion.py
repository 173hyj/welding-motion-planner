"""Same exact retreat acceptance, with actual failed and last accepted q traces."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'welding_tool_exit_v1'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'welding_variable_open_v1'))
from run_arm import full_probe, resolve_exit_direction
from arm_path import continuous_ik, interpolation_probes, line_error
from contracts import ContactTracker
from variable_core import driver_samples


def lift_path(engine, weld, candidate, selected, exit_path, *, driver_mm, opening_frames):
    driver_samples(driver_mm)
    rid = candidate['robot_id']; model = engine.world.models[rid]
    q0 = np.asarray(selected['q_rad']); other = selected['other_robot_id']
    park = np.asarray(selected['other_q_rad'])
    moved, _ = engine.target(weld,candidate)
    start = model.tcp(q0); direction = resolve_exit_direction(exit_path,engine.delta)
    length = float(exit_path['length_mm'])
    if (not opening_frames or opening_frames[0]['q_rad'] != selected['q_rad'] or
        opening_frames[-1]['driver_mm'] != driver_mm or
        any(f['q_rad'] != selected['q_rad'] for f in opening_frames)):
        raise ValueError('Opening prefix must end at requested driver and lock archived q0')
    frames = list(opening_frames)
    def rejected(reason,s,failed_q=None,**details):
        return dict(passed=False,reason=reason,travel_mm=float(s),failure_travel_mm=float(s),
                    failure_q_rad=np.asarray(failed_q).tolist() if failed_q is not None else None,
                    last_accepted_q_rad=frames[-1]['q_rad'],last_accepted_travel_mm=frames[-1]['travel_mm'],
                    accepted_retreat_waypoints=[f for f in frames if f['travel_mm']>0.],**details)
    maximum_line_error = 0.; maximum_rotation_error = 0.; maximum_joint_step = 0.
    open_state = engine.state(rid,start,driver_mm,moved,engine.world.static,'opening')
    context = engine.release_context(rid,start,moved,dict(open_state=open_state), driver_mm=driver_mm)
    for support in context['supports']:
        if np.asarray(support['outward_normal'])@direction <= 1e-6:
            return rejected('actual_open_state_requires_other_release_direction',0.,q0)
    tracker = ContactTracker(context['initial_touching'])
    q = q0.copy(); travel = 0.; step = 2.
    while travel < length-1e-8:
        requested = min(length,travel+step)
        target = start.copy(); target[:3,3] += requested*direction
        fit = continuous_ik(model,target,q)
        if not fit['passed']:
            if step > .100001:
                step = max(.1,step/2.); continue
            return rejected('continuous_ik_failed',requested,fit.get('q_rad'),failure=fit)
        next_q = np.asarray(fit['q_rad']); probes = interpolation_probes(q,next_q)
        probe_data = []; shape_ok = True
        for qi in probes[1:]:
            actual = model.tcp(qi)
            s = float((actual[:3,3]-start[:3,3])@direction)
            lateral = line_error(actual[:3,3],start[:3,3],direction)
            rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@actual[:3,:3]).as_rotvec())))
            if lateral > .05 or rotation > .01 or s < travel-1e-5 or s > requested+.05:
                shape_ok = False; break
            probe_data.append((qi,actual,s,lateral,rotation))
        if not shape_ok:
            if step > .100001:
                step = max(.1,step/2.); continue
            return rejected('actual_joint_interpolation_off_straight_line',requested,next_q)
        for qi,actual,s,lateral,rotation in probe_data:
            phase = 'release' if s <= 5.000001 else 'separated'
            state = full_probe(engine,rid,qi,other,park,driver_mm,moved,phase,context)
            if not state['passed']:
                return rejected('full_retreat_blocked',s,qi,failure=state)
            if not tracker.advance(state['touching'],s):
                return rejected('release_contact_reappeared_or_window_exceeded',s,qi)
            maximum_line_error = max(maximum_line_error,lateral)
            maximum_rotation_error = max(maximum_rotation_error,rotation)
            frames.append(dict(q_rad=qi.tolist(),driver_mm=driver_mm,phase=phase,
                               tcp_xyz_mm=actual[:3,3].tolist(),travel_mm=s))
        maximum_joint_step = max(maximum_joint_step,float(np.degrees(np.max(abs(next_q-q)))))
        q = next_q; travel = requested; step = min(2.,step*2.)
    clearance = engine.minimum_distance(rid,model.tcp(q),driver_mm,engine.world.static)
    if clearance < 5.:
        return rejected('actual_endpoint_clearance_below_5mm',length,q,clearance_mm=clearance)
    return dict(passed=True,driver_mm=driver_mm,robot_id=rid,other_robot_id=other,other_q_rad=park.tolist(),
        positioner_deg=selected['positioner_deg'],candidate=candidate,
        q_start_rad=q0.tolist(),q_end_rad=q.tolist(),direction_world=direction.tolist(),
        length_mm=length,endpoint_clearance_mm=clearance,frames=frames,
        maximum_sampled_line_error_mm=maximum_line_error,
        maximum_sampled_rotation_error_deg=maximum_rotation_error,
        maximum_ik_step_joint_change_deg=maximum_joint_step,
        actual_joint_interpolation_checked=True,
        scope='Sampled source-FK joint interpolation and both complete source robots. '
              'Opening/release/arm edges have no continuous collision certificate. '
              'Conditional on existing static assembly-component exemptions; positioner rotation not planned.')
