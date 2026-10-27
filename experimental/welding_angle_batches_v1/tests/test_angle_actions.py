import importlib.util
import json
import sqlite3
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('angle_actions', Path(__file__).parents[1] / 'actions.py')
actions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(actions)


def test_ranking_diversifies_pose_before_second_winding_and_keeps_exact_q():
    rows = [dict(robot_id='a', roll_deg=r, axis_sign=1, q_rad=[q]*6)
            for r, q in [(0, 0), (0, 1), (90, 2)]]
    ranked = actions.rank_candidates(rows, dict(robot_id='a', roll_deg=0, q_rad=[0]*6))
    assert [r['roll_deg'] for r in ranked] == [0, 90, 0]
    assert ranked[-1]['q_rad'] == [1]*6


def test_catalog_excludes_numeric_retained_and_preserves_branch(tmp_path):
    db = tmp_path/'catalog.sqlite'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE candidates(weld,angle,robot,roll,sign,branch,payload)')
        for branch, origin in enumerate(['analytic', 'numeric_retained', 'analytic_and_numeric']):
            connection.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?)',
                               ('w', 90, 'a', 20, 1, branch, json.dumps(dict(origin=origin, q_rad=[branch]*6))))
    rows = actions.load_candidates(db, 90)
    assert [r['catalog_branch'] for r in rows['w']] == [0, 2]
    assert rows['w'][1]['q_rad'] == [2]*6
    assert actions.required_weld_ids(db) == ['w']


def test_schedule_complete_is_separate_from_action_success():
    summary = actions.action_summary([dict(passed=True), dict(passed=False)], 2, 90)
    assert summary['complete'] is True
    assert summary['all21_actions_passed'] is False
    assert summary['passed'] == 1


def test_previous_opening_is_tried_first_without_omitting_standard_amplitudes():
    assert actions.amplitude_order([10., 30., 60.], -30.) == [30., 10., 60.]
    assert actions.amplitude_order([10., 30., 60.], -20.) == [20., 10., 30., 60.]
