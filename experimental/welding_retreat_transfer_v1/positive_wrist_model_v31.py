"""Narrow an IK search domain without modifying the original source robot."""
import numpy as np


class PositiveWristModel:
    def __init__(self, source):
        self.source = source
        self.bounds = np.array(source.bounds, copy=True)
        self.bounds[0, 5] = max(self.bounds[0, 5], 1e-6)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def limits_valid(self, q):
        q = np.asarray(q)
        return (q.shape == (6,) and np.isfinite(q).all() and
                q[5] >= self.bounds[0, 5] and self.source.limits_valid(q))
