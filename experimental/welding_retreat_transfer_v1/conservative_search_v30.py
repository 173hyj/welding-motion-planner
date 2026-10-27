"""Coarse source-mesh proposals, accepted only after the original fine checks."""
import numpy as np
from state_space import interpolate
from world_bridge_v2 import SourceBridge


def certified_search(bridge, propose, *, attempts=4):
    trials = []
    for attempt in range(attempts):
        path, details = propose(attempt)
        certified = path is not None and bridge.certify_path(path)
        trials.append(dict(attempt=attempt, proposal=details, candidate_found=path is not None,
                           fine_certified=certified))
        if certified:
            return path, trials
    return None, trials


class SearchBridge(SourceBridge):
    def __init__(self, engine):
        super().__init__(engine)
        self.rejected_search_edges = set()

    @staticmethod
    def edge_key(a, b):
        a, b = tuple(np.asarray(a, float)), tuple(np.asarray(b, float))
        return (a, b) if a <= b else (b, a)

    def edge_scaled(self, a, b):
        """Proposal only: a positive here never certifies the physical path."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self.edge_key(a, b) in self.rejected_search_edges:
            return False
        for u in (0., 1., .5, .25, .75):
            if not self.valid_scaled(a + (b - a) * u):
                return False
        coarse = np.r_[np.full(12, np.deg2rad(1.)), np.deg2rad(1.), 2., 2.]
        return all(self.valid_scaled(q) for q in interpolate(a, b, coarse / self.scale))

    def certify_path(self, path):
        if not path:
            return False
        if len(path) == 1:
            return self.valid_scaled(path[0])
        for a, b in zip(path, path[1:]):
            if not SourceBridge.edge_scaled(self, a, b):
                self.rejected_search_edges.add(self.edge_key(a, b))
                return False
        return True
