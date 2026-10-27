import importlib.util
from pathlib import Path
import numpy as np
import pytest

def load():
    path=Path(__file__).resolve().parents[1]/'experimental/dual_arm_parallel_v1/motion.py'
    spec=importlib.util.spec_from_file_location('dual_motion',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_display_timing_respects_existing_joint_gun_and_tcp_speed_assumptions():
    m=load();states=np.zeros((2,7));states[1,0]=np.deg2rad(60);states[1,6]=-80
    assert m.section_times(states,np.array([[0,0,0],[1000,0,0]]))[-1]==5.
    assert m.section_times(states)[-1]==2.

def test_chain_requires_continuous_seams_and_keeps_weld_dwell():
    m=load();s=np.zeros(7);chain=m.new_chain(s)
    m.append_piece(chain,np.array([s,s]),'weld',task='fixed',duration=.6)
    assert chain['times'][-1]==.6
    assert m.piece_at(chain,.3)['phase']=='weld'
    assert m.piece_at(chain,-.1) is None
    wrong=s.copy();wrong[0]=1.
    with pytest.raises(ValueError,match='seam'):
        m.append_piece(chain,np.array([wrong,s]),'transfer')
def test_piece_boundary_survives_round_trip_offset_subtraction():
    from experimental.dual_arm_parallel_v1.motion import piece_at
    chain={'times':[0.,10.6], 'pieces':[{'start':0.,'end':10.,'phase':'close'},
        {'start':10.,'end':10.6,'phase':'weld'}]}
    assert piece_at(chain,np.nextafter(10.,-np.inf))['phase']=='weld'
