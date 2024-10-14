"""
This module contains a collection of aggregation functions that can be used to combine different objectives and optimize 
them together.
"""

def conductivity_aggregation_fn(x, delta_beta):
    """Calculate the ionic conductivity from S0, S1 and S2 objectives from the generalized Arrhenius fit. 

    Parameters
    ----------
    x : nd_array of dimension 2
        Array that contains three columns for each of the S0, S1 and S2 objectives and as many rows as data points.
    delta_beta : float
        Difference between 1/T and 1/T_0.

    Returns
    -------
    nd_array of dimension 1
        The ionic conductivity for each data point.
    """

    if len(x.shape) == 1:
        x = x.reshape(1,-1).T

    conductivity = x[0,:] - delta_beta*x[1,:] - x[2,:]*delta_beta**2
    return conductivity

