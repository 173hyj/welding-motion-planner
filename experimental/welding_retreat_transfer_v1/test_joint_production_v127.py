"""Exercise the production controller, rather than only its generic helper."""
from types import SimpleNamespace
import numpy as np
import pytest
import run_joint_weld_retreat_v127 as controller


@pytest.fixture
def search(monkeypatch, tmp_path):
    class Model:
        q_initial = np.zeros(6)
        def limits_valid(self, q):
            return True
    models = {'a': Model(), 'b': Model()}
    angles = []
    engine = SimpleNamespace(
        world=SimpleNamespace(models=models),
        environment=SimpleNamespace(layout_collisions=lambda a: {'blocking_pairs': []}, fixed_env=[]),
        set_angle=angles.append,
        target=lambda weld, candidate: (weld, np.eye(4)),
        search=SimpleNamespace(gun_pairs=lambda *a: []))
    monkeypatch.setattr(controller, 'ENGINE', engine)
    monkeypatch.setattr(controller, 'SIGNS', {'a': -1, 'b': 1})
    monkeypatch.setattr(controller, 'OUT', tmp_path)
    monkeypatch.setattr(controller, 'HINTS', {})
    monkeypatch.setattr(controller, 'SETTINGS', dict(point_seconds=60., poses=2, configurations=2))
    monkeypatch.setattr(controller, 'branch_parks', lambda *a: {r: np.zeros(6).tolist() for r in models})
    monkeypatch.setattr(controller, 'analytical_seed_bank', lambda *a: ([{'q_rad': [0.]*6}, {'q_rad': [1.]*6}], {}))
    row = dict(weld_id='w', passed=False, stationary_done=False, relative_gun_candidates=2,
               angle_trials=[], allowed=[dict(robot_id='a', roll_deg=r, axis_sign=1) for r in (0, 5)])
    return row, angles


def outcome(candidate, q, passed):
    result = dict(passed=passed, closed_passed=True, attempts=[], path=None)
    if passed:
        result.update(path=dict(passed=True, driver_mm=-10., online_validation={'passed': True}),
                      selected=dict(candidate, q_rad=list(q)))
    return result


def test_production_continues_to_next_ik_when_closed_pose_has_no_retreat(monkeypatch, search):
    row, angles = search
    tried = []
    def evaluate(engine, weld, candidate, q, *args):
        tried.append((candidate['roll_deg'], q[0]))
        return outcome(candidate, q, q[0] == 1.)
    monkeypatch.setattr(controller, 'try_configuration', evaluate)
    _, result = controller.search_point(0, row, {'id': 'w'}, 'stationary')
    assert tried == [(0, 0.), (0, 1.)]
    assert result['passed'] and result['stationary_passed'] and angles == [0]


def test_production_closed_only_failure_moves_to_another_pose_then_positioner(monkeypatch, search):
    row, angles = search
    tried = []
    def evaluate(engine, weld, candidate, q, parks, settings, cache):
        tried.append((settings['angle_deg'], candidate['roll_deg'], q[0]))
        return outcome(candidate, q, settings['angle_deg'] == 10 and candidate['roll_deg'] == 5)
    monkeypatch.setattr(controller, 'try_configuration', evaluate)
    _, stationary = controller.search_point(0, row, {'id': 'w'}, 'stationary')
    assert not stationary['passed'] and stationary['best'] is None
    assert {roll for angle, roll, q in tried} == {0, 5}
    assert controller.jobs_for_stage([stationary], 'positioner') == [0]
    monkeypatch.setattr(controller, 'positioner_angles', lambda: [5, 10])
    _, result = controller.search_point(0, stationary, {'id': 'w'}, 'positioner')
    assert result['passed'] and not result['stationary_passed']
    assert angles == [0, 5, 10]
    assert result['relative_gun_candidates'] == 2 and result['weld_id'] == 'w'
