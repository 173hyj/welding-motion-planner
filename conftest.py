"""Explicitly separate tests requiring the undistributed industrial dataset."""
from pathlib import Path
import pytest
PRIVATE_INTEGRATION = {'tests/test_source_fk_audit_v17.py::SourceFKAuditTests::test_source_fk_agrees_archived_world_flange', 'tests/test_kinematics.py::KinematicsTests::test_current_pose_matches_independently_decoded_source_link_matrices', 'tests/test_gun_kinematics.py::SourceGunKinematicsTests::test_source_links_and_closed_pose_are_preserved', 'tests/test_weld_task_planner.py::WeldTaskPlannerTests::test_time_law_identifies_source_limits_and_coupling', 'tests/test_tcp_registration.py::TCPRegistrationTests::test_source_weld_registration_pairs_all_points', 'tests/test_weld_task_planner.py::WeldTaskPlannerTests::test_source_timing_bounds_independent_coupled_and_nonlinear_axes', 'tests/test_weld_task_planner.py::WeldTaskPlannerTests::test_unknown_inventory_cannot_silently_disappear', 'tests/test_kinematics.py::KinematicsTests::test_dependent_joint_limits_use_full_source_polygon', 'tests/test_layout_reach_audit.py::LayoutReachAuditTests::test_robot_anchors_remain_on_each_source_axis', 'tests/test_counterbalance_fk_v17.py::CounterbalanceFKTests::test_cylinder_and_rod_share_direction_of_transformed_source_pins', 'tests/test_scene.py::SceneTests::test_fixture_world_pose_includes_parent_assembly', 'tests/test_tcp_registration.py::TCPRegistrationTests::test_inverse_tcp_relative_frame_hits_source_electrode', 'tests/test_source_fk_audit_v17.py::SourceFKAuditTests::test_source_coupling_factor_is_not_silently_ignored', 'tests/test_plan.py::PlanTests::test_diagnostic_plan_contains_actual_frames_and_explicit_status', 'tests/test_weld_task_planner.py::WeldTaskPlannerTests::test_source_passive_values_reconstruct_closure_transforms', 'tests/test_gun_kinematics.py::SourceGunKinematicsTests::test_opening_follows_source_dependent_arm', 'tests/test_counterbalance_fk_v17.py::CounterbalanceFKTests::test_source_crank_is_equivariant_under_robot_first_axis', 'tests/test_weld_task_planner.py::WeldTaskPlannerTests::test_planner_preserves_all_source_weld_ids_and_declares_unavailable', 'tests/test_counterbalance_audit_v17.py::CounterbalanceAuditTests::test_closure_and_base_rotation_covariance', 'tests/test_scene.py::SceneTests::test_source_coordinate_registration_within_csv_precision', 'tests/test_gun_kinematics.py::SourceGunKinematicsTests::test_real_mesh_partition_matches_independent_original_gun'}

def pytest_addoption(parser):
    parser.addoption('--run-private-integration', action='store_true', default=False,
                     help='Run industrial-data integration checks; provide source/, build/ and local .venv first.')

def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-private-integration'):
        return
    for item in items:
        if item.nodeid in PRIVATE_INTEGRATION:
            item.add_marker(pytest.mark.skip(reason='Requires undistributed industrial dataset; opt in with --run-private-integration'))
