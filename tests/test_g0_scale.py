import numpy as np
import pytest
from cfact.generation.g0_scale import source_geometry_scale

def poses(length):
    a=np.repeat(np.eye(4)[None],101,axis=0);a[:,0,3]=np.linspace(0,length,101);return a

def test_scale_is_source_and_geometry_bounded_without_future_effects():
    x=source_geometry_scale(poses(.5),geometry_allowance_m=.015)
    assert x['requested_amplitude_m']==pytest.approx(.015)
    assert x['outcome_blind']
    assert source_geometry_scale(poses(.02),geometry_allowance_m=.03)['requested_amplitude_m']<.003
    assert source_geometry_scale(poses(0),geometry_allowance_m=.03)['requested_amplitude_m']==0

def test_invalid_current_geometry_does_not_fall_back_to_large_uniform_noise():
    with pytest.raises(ValueError):source_geometry_scale(poses(.5),geometry_allowance_m=float('nan'))
