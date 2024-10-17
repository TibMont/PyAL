import pytest
import numpy as np
from PyAL.aggregation_fn import conductivity_aggregation_fn

@pytest.mark.parametrize("x,expected_result", [
    (np.ones((3,2)), np.array([1.24182278, 1.24182278])),
    (np.ones((3,1)), np.array([1.24182278])),
    (np.ones(3), np.array([1.24182278]))
    ])

def test_conductivity_aggregation_fn(x, expected_result):
    delta_beta = 1000/333.15 - 1000/293.15
    result = conductivity_aggregation_fn(x,delta_beta)
    result = np.round(result, 8) 
    assert np.array_equal(result, expected_result)

