import numpy as np
from experimental.trajopt_transfer_v1.solver import optimize, SolverOptions


def empty_oracle(x, linearize=True):
    return np.empty(0), np.zeros((0, x.size)) if linearize else None


def sphere_oracle(x, linearize=True):
    points = (x[:-1] + x[1:]) / 2
    norms = np.linalg.norm(points, axis=1)
    g = norms - 0.35
    jac = np.zeros((len(g), x.size))
    if linearize:
        for i, p in enumerate(points):
            grad = p / max(norms[i], 1e-12) / 2
            jac[i, 2*i:2*i+2] = grad
            jac[i, 2*i+2:2*i+4] = grad
    return g, jac if linearize else None


def test_smooths_path_and_keeps_endpoints_exact():
    seed = np.array([[-1., 0.], [-.5, .7], [0., -.3], [.5, .8], [1., 0.]])
    result = optimize(seed, [-1., -1.], [1., 1.], empty_oracle)
    np.testing.assert_array_equal(result.trajectory[[0, -1]], seed[[0, -1]])
    assert result.objective < result.initial_objective
    assert result.feasible
    assert np.max(np.abs(result.trajectory[:, 1])) < 1e-3


def test_curved_seed_stays_clear_of_midpoint_sphere():
    seed = np.array([[-1., 0.], [-.5, .7], [0., .8], [.5, .7], [1., 0.]])
    result = optimize(seed, [-1., -1.], [1., 1.], sphere_oracle,
                      SolverOptions(penalty=200., max_iterations=100))
    assert result.feasible
    assert np.min(sphere_oracle(result.trajectory, False)[0]) >= -1e-5
    assert result.objective < result.initial_objective


def test_callback_rejects_step_and_reports_infeasibility_truthfully():
    seed = np.array([[-1., 0.], [0., .6], [1., 0.]])
    result = optimize(seed, [-1., -1.], [1., 1.], empty_oracle,
                      SolverOptions(max_iterations=4, feasibility_callback=lambda x: False))
    np.testing.assert_array_equal(result.trajectory, seed)
    assert not result.success
    assert not result.feasible
    assert any(not row['accepted'] for row in result.history)


def test_public_objective_uses_options_weights():
    from experimental.trajopt_transfer_v1.solver import objective, Options
    x = np.array([[0.], [1.], [0.]])
    assert objective(x, Options(velocity_weight=2., acceleration_weight=3.)) == 16.


def test_stationary_infeasible_constraint_escalates_penalty_and_reports_limit():
    seed = np.array([[0.], [0.], [0.]])
    rows = []
    def impossible(x, linearize=True):
        return np.array([-1.]), np.zeros((1, x.size)) if linearize else None
    result = optimize(seed, [-1.], [1.], impossible,
                      SolverOptions(penalty=10., max_penalty=100., progress_callback=rows.append))
    assert not result.success
    assert result.status == 'penalty_limit'
    assert [row['penalty'] for row in rows] == [10., 100.]
    assert rows == result.history
