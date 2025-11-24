import numpy as np

"""
This module contains a collection of aggregation functions that can be used to combine different objectives and optimize 
them together.
"""


def conductivity_aggregation_fn(
    x, features, uncert=False, delta_beta=0, scaler=None, use_features=True
):
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
        x = x.reshape(1, -1).T

    if isinstance(features, np.ndarray) and use_features == True:
        if scaler != None:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            features = scaler.inverse_transform(features)

        if uncert == False:
            zeros = np.where(features[:, 0] == 0)
            conductivity = (
                x[0, :]
                + np.log10(features[:, 0])
                - delta_beta * x[1, :]
                - x[2, :] * delta_beta**2
            )
            conductivity[zeros] = (
                x[0, zeros] - delta_beta * x[1, zeros] - x[2, zeros] * delta_beta**2
            )
            # print('V:')
            # print(conductivity)

            return conductivity

        else:
            conductivity = x[0, :] + delta_beta * x[1, :] + x[2, :] * delta_beta**2
            # print('Uncert:')
            # print(x)
            # print(conductivity)

            return conductivity

    else:

        if uncert == False:
            conductivity = x[0, :] - delta_beta * x[1, :] - x[2, :] * delta_beta**2
            # print('V:')
            # print(conductivity)

            return conductivity

        else:
            conductivity = x[0, :] + delta_beta * x[1, :] + x[2, :] * delta_beta**2
            # print('Uncert:')
            # print(x)
            # print(conductivity)

            return conductivity


def identity_aggregation_fn(x, features, uncert=False):
    return x
