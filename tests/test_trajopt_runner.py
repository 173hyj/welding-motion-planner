import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('trajopt_runner', ROOT/'experimental/trajopt_transfer_v1/run_optimize.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_refinement_preserves_every_original_corner_and_length():
    m = module()
    source = np.array([[0., 0.], [1., 0.], [1., 2.]])
    refined = m.refine_seed(source, 12)
    assert len(refined) == 12
    assert all(any(np.array_equal(p, q) for q in refined) for p in source)
    assert np.isclose(m.path_length(refined), 3.)
    np.testing.assert_array_equal(refined[[0, -1]], source[[0, -1]])


def test_acceptance_requires_fine_check_exact_endpoints_and_no_length_regression():
    m = module()
    base = np.array([[0., 0.], [1., 1.], [2., 0.]])
    good = np.array([[0., 0.], [1., .5], [2., 0.]])
    assert m.accept_candidate(base, good, 2., 1., True)[0]
    assert not m.accept_candidate(base, good, 2., 1., False)[0]
    assert not m.accept_candidate(base, good, 1., 2., True)[0]
    bad = good.copy(); bad[-1, 0] += 1e-10
    assert not m.accept_candidate(base, bad, 2., 1., True)[0]
    long = good.copy(); long[1, 1] = 2.
    assert not m.accept_candidate(base, long, 2., 1., True)[0]
