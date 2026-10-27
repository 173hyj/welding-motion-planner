import sys
import numpy as np
import pytest
from experimental.dual_arm_parallel_v1.source_pool import SourcePool, prefill_fine_cache


class FakeBridge:
    def __init__(self): self.calls=[]
    def probe(self,state):
        self.calls.append(float(state[0]))
        passed=state[0]!=2
        return dict(passed=passed,reason='ok' if passed else 'blocked',
                    blocking_pairs=[] if passed else [dict(id='collision')],large='omitted')


class FakeChecker:
    robot_ids=['a','b']
    def __init__(self): self.calls=[]
    def probe(self,state,contexts):
        self.calls.append((float(state[0]),contexts))
        return dict(passed=state[0]!=2,reason='dual',blocking_pairs=[float(state[0])],
                    tool_states={rid:dict(touching=context['touching']) for rid,context in zip(self.robot_ids,contexts) if context is not None})


def data():
    states=np.zeros((4,15));states[:,0]=[3,1,2,0]
    contexts=[(dict(touching=[i]),None) for i in range(4)]
    return states,contexts


def test_serial_probes_every_sample_in_order_preserves_failures_and_contexts():
    states,contexts=data();bridge=FakeBridge();checker=FakeChecker()
    with SourcePool(bridge,checker,workers=1) as pool:
        strict=pool.strict_states(states)
        dual=pool.dual_states(zip(states,contexts))
    assert bridge.calls==[3,1,2,0]
    assert checker.calls==list(zip([3,1,2,0],contexts))
    assert [r['passed'] for r in strict]==[True,True,False,True]
    assert strict[2]['blocking_pairs']==[dict(id='collision')]
    assert 'large' not in strict[0]
    assert [r['touching'] for r in dual]==[{0:[i],1:[]} for i in range(4)]
    assert not dual[2]['passed']


@pytest.mark.skipif(sys.platform=='win32',reason='fork is unavailable on Windows')
def test_fork_results_identical_to_serial_and_world_mutation_stays_in_workers():
    states,contexts=data();bridge=FakeBridge();checker=FakeChecker()
    with SourcePool(FakeBridge(),FakeChecker(),workers=1) as serial:
        expected_s=serial.strict_states(states)
        expected_d=serial.dual_states(zip(states,contexts))
    with SourcePool(bridge,checker,workers=2) as pool:
        assert pool.strict_states(states)==expected_s
        assert pool.dual_states(zip(states,contexts))==expected_d
    assert bridge.calls==[] and checker.calls==[]


def test_closed_pool_rejects_work_and_bad_worker_counts():
    with pytest.raises(ValueError,match='positive'):
        SourcePool(FakeBridge(),FakeChecker(),workers=0)
    pool=SourcePool(FakeBridge(),FakeChecker(),workers=1)
    pool.close();pool.join();pool.close();pool.join()
    with pytest.raises(RuntimeError,match='closed'):
        pool.strict_states(np.zeros((1,15)))


def test_prefill_exact_original_even_samples_and_early_quarters_keeps_false_cache():
    bridge=FakeBridge();bridge.cache={};bridge.scale=np.r_[np.ones(13),100.,100.]
    bridge.resolutions=np.ones(15);bridge.checks=0;bridge.last_failure=None
    path=np.zeros((2,15));path[1,0]=3;path[:,13]=-.25
    bridge.cache[tuple(path[0]*bridge.scale)]=True
    with SourcePool(bridge,FakeChecker(),workers=1) as pool:
        stats=prefill_fine_cache(bridge,path,pool)
    # count=ceil(3)=3 rounds to 4, includes .75/1.5/2.25.
    assert sorted(s[0] for s in bridge.cache)==[0.,.75,1.5,2.25,3.]
    assert all(s[13]==-25 for s in bridge.cache)
    assert stats['samples']==4 and bridge.checks==4
    bridge.cache[tuple(np.r_[np.zeros(13),-25.,0.])]=False
    before=len(bridge.calls)
    with SourcePool(bridge,FakeChecker(),workers=1) as pool:
        assert prefill_fine_cache(bridge,path,pool)['samples']==0
    assert len(bridge.calls)==before


def test_prefill_keeps_probe_failure_in_cache_and_diagnostic():
    bridge=FakeBridge();bridge.cache={};bridge.scale=np.ones(15)
    bridge.resolutions=np.full(15,.5);bridge.checks=0;bridge.last_failure=None
    path=np.zeros((2,15));path[1,0]=3
    with SourcePool(bridge,FakeChecker(),workers=1) as pool:
        stats=prefill_fine_cache(bridge,path,pool)
    failed=np.zeros(15);failed[0]=2
    assert bridge.cache[tuple(failed)] is False and stats['failures']==1
    assert bridge.last_failure['state']==failed.tolist()
    assert bridge.last_failure['reason']=='blocked'
    assert bridge.last_failure['blocking_pairs']==[dict(id='collision')]
