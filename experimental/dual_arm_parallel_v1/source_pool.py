"""Ordered source probes in fork-isolated worlds, without pickling FCL objects."""
import multiprocessing
import sys
import numpy as np

_worker_bridge=None
_worker_checker=None


def _minimal(result):
    return dict(passed=bool(result['passed']),reason=result.get('reason',''),
                blocking_pairs=result.get('blocking_pairs',[]))


def _strict_probe(state):
    return _minimal(_worker_bridge.probe(state))


def _dual_result(checker,state,contexts):
    result=checker.probe(state,contexts)
    minimal=_minimal(result)
    tools=result.get('tool_states',{})
    minimal['touching']={arm:list(tools.get(rid,{}).get('touching',[]))
                         for arm,rid in enumerate(checker.robot_ids)}
    return minimal


def _dual_probe(record):
    return _dual_result(_worker_checker,*record)


class SourcePool:
    """Create only after the parent bridge/checker worlds are fully built.

    Linux workers inherit independent world memory through fork. Only states,
    contexts and compact results cross process pipes; world/FCL objects never
    get pickled. Windows or workers=1 uses ordered sequential probes.
    Parent bridge caches/check counts are deliberately managed by the caller.
    """
    def __init__(self,bridge,checker,workers=8):
        if not isinstance(workers,int) or workers<=0:
            raise ValueError('positive integer worker count required')
        self.bridge=bridge;self.checker=checker;self.workers=workers
        self._closed=False;self._joined=False;self._pool=None
        if sys.platform!='win32' and workers>1:
            global _worker_bridge,_worker_checker
            _worker_bridge=bridge;_worker_checker=checker
            self._pool=multiprocessing.get_context('fork').Pool(workers)

    def _check_open(self):
        if self._closed:raise RuntimeError('source pool is closed')

    def strict_states(self,states):
        self._check_open()
        records=list(states)
        if self._pool is None:return [_minimal(self.bridge.probe(s)) for s in records]
        return self._pool.map(_strict_probe,records)

    def dual_states(self,records):
        self._check_open()
        records=list(records)
        if self._pool is None:return [_dual_result(self.checker,*r) for r in records]
        return self._pool.map(_dual_probe,records)

    def close(self):
        if not self._closed:
            if self._pool is not None:self._pool.close()
            self._closed=True

    def join(self):
        if not self._closed:self.close()
        if not self._joined:
            if self._pool is not None:self._pool.join()
            self._joined=True

    def __enter__(self):
        self._check_open();return self

    def __exit__(self,exc_type,exc,tb):
        if exc_type is not None and self._pool is not None:
            self._pool.terminate();self._closed=True
        else:self.close()
        self.join()


def prefill_fine_cache(bridge,scaled_path,pool):
    """Prefill exactly the source fine/early samples, preserving false entries.

    SourceBridge.edge_scaled must still run afterward as the authoritative
    certificate. Probe results populate the same exact physical tuple keys
    used by valid_scaled; no rounded keys or sparse positive certificates.
    """
    from experimental.welding_retreat_transfer_v1.state_space import interpolate
    path=np.asarray(scaled_path,float)
    if path.ndim!=2 or path.shape[1]!=15 or not len(path) or not np.isfinite(path).all():
        raise ValueError('finite nonempty scaled Nx15 path required')
    pending={}
    def add(sample):
        physical=np.asarray(sample)*bridge.scale
        key=tuple(physical)
        if key not in bridge.cache and key not in pending:pending[key]=physical
    if len(path)==1:add(path[0])
    for a,b in zip(path,path[1:]):
        for u in (0.,1.,.5,.25,.75):add(a+(b-a)*u)
        for sample in interpolate(a,b,bridge.resolutions/bridge.scale):add(sample)
    results=pool.strict_states(pending.values())
    if len(results)!=len(pending):raise RuntimeError('source pool lost probe results')
    failures=0
    for (key,physical),result in zip(pending.items(),results):
        bridge.cache[key]=bool(result['passed'])
        bridge.checks+=1
        if not result['passed']:
            failures+=1
            bridge.last_failure=dict(state=physical.tolist(),**result)
    return dict(samples=len(pending),failures=failures)
