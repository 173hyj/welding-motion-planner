import importlib.util
from pathlib import Path
import numpy as np

SPEC=importlib.util.spec_from_file_location('local_window',Path(__file__).parents[1]/'local_window.py')
window=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(window)


def test_splice_preserves_prefix_suffix_and_offsets_original_clock():
    old=np.arange(8*15,dtype=float).reshape(8,15)
    middle=np.vstack([old[2],np.full(15,999.),old[5]])
    states,times=window.splice_states_times(old,np.arange(8,dtype=float),2,5,middle,[0.,.5,1.])
    np.testing.assert_array_equal(states[:3],old[:3])
    np.testing.assert_array_equal(states[-3:],old[5:])
    np.testing.assert_array_equal(times,[0,1,2,2.5,3,4,5])


def test_metrics_gate_requires_all_three_improvements():
    assert window.accept_metrics(460.,5,540.)
    assert not window.accept_metrics(463.720785,5,540.)
    assert not window.accept_metrics(460.,7,540.)
    assert not window.accept_metrics(460.,5,540.1)


def test_window_indices_use_last_task2_retreat_first_task5_ingress():
    tasks=np.zeros((7,2),int);tasks[:,1]=[2,2,2,3,4,5,5]
    phases=np.full((7,2),'idle',dtype='<U12');phases[:,1]=['retreat','retreat','transfer','weld','weld','ingress','close']
    assert window.window_indices(tasks,phases)==(1,5)
