import numpy as np
import run_joint_weld_retreat_v134 as search


def test_feedback_excludes_only_current_failed_weld_configuration():
    candidate=dict(robot_id='r',roll_deg=5,axis_sign=1)
    old=dict(candidate,q_rad=[0.]*6,positioner_deg=30)
    assert search.blocked_configuration(candidate,[0.]*6,30,old)
    assert not search.blocked_configuration(candidate,[.1]*6,30,old)
    assert not search.blocked_configuration(candidate,[0.]*6,35,old)
    assert not search.blocked_configuration(dict(candidate,roll_deg=10),[0.]*6,30,old)


def test_candidate_replacement_invalidates_all_incident_edges_only():
    import feedback_reselection_v134 as feedback
    edges={(0,1):1,(1,0):2,(1,2):3,(2,3):4}
    assert feedback.invalidate_incident(edges,1)=={(2,3):4}
