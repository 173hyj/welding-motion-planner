import numpy as np
import pytest
from export_latest_route_v160 import action_sections


def test_action_sections_keep_every_retreat_sample_and_reverse_opening():
    states = [np.r_[np.zeros(13), d, 0.] for d in (0., -5., -10.)]
    end = states[-1].copy(); end[0] = .2
    paths = dict(frames=[dict(travel_mm=t) for t in (0., 0., 0., 1000.)])
    ingress, close, opening, retreat = action_sections(paths, states + [end])
    np.testing.assert_array_equal(ingress, [end, states[-1]])
    np.testing.assert_array_equal(close, list(reversed(states)))
    np.testing.assert_array_equal(opening, states)
    np.testing.assert_array_equal(retreat, [states[-1], end])


def test_no_retreat_samples_rejected():
    with pytest.raises(ValueError):
        action_sections(dict(frames=[dict(travel_mm=0.)]), [np.zeros(15)])
