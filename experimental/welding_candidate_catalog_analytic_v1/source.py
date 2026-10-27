"""Analytic IK with unchanged original source-mesh acceptance."""
from experimental.welding_candidate_catalog_v1.source import SourceCatalog
from experimental.welding_candidate_catalog_analytic_v1.analytic import FanucAnalytic
import time


class AnalyticSourceCatalog(SourceCatalog):
    def __init__(self):
        super().__init__(seed_count=0)
        self.analytic={rid:FanucAnalytic(model) for rid,model in self.world.models.items()}

    def solve(self,weld,pose):
        _,target=self._target(weld,pose)
        started=time.monotonic()
        solutions,stats=self.analytic[pose['robot_id']].solve(target)
        stats['elapsed_s']=time.monotonic()-started
        return solutions,stats
