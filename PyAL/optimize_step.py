"""
This module contains functionality to evaluate the acquisition functions and 
find their optimum using different approaches. The supported approaches are:
- Evaluation on discrete grid
- Evaluation using LBFGS optimizer from Scipy
- Evaluation using PSO optimizer from PySwarms
- Evaluation for single objectives and combined objectives via aggregation functions
"""

# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import copy
import sys
import os
import warnings

import numpy as np
from scipy.stats.qmc import scale
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from pyswarms.single.global_best import GlobalBestPSO

from PyAL.acfn_continuous import EI_con, POI_con, UCB_con, IDEAL_con, GSx_con
from PyAL.acfn_continuous import GSy_con, iGS_con, QBC_con, SGSx_con, std_con, UIDAL_con
from PyAL.acfn_continuous_multi import EI_multi, POI_multi, UCB_multi, IDEAL_multi
from PyAL.acfn_continuous_multi import GSx_multi, GSy_multi, iGS_multi, QBC_multi
from PyAL.acfn_continuous_multi import SGSx_multi, std_multi, UIDAL_multi
from PyAL.acfn_discrete import EI, POI, UCB, IDEAL, GSx, GSy, iGS, SGSx, UIDAL

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = 'ignore::ConvergenceWarning'

def step_discrete(
                    acquisition_function,
                    estimated_observation_y,
                    alpha, mean, std,
                    n_data, data_indices, pool,
                    custom_acfn_input,
                    rng,
):
    """Evaluating an acquisition function on a discrete grid or for discrete values. 
    This is suited for pool-based learning.

    Parametersls
    ----------
    acquisition_function : str or callable
        Name of the acquisition function or a callable. Choose from: ei, poi, 
        ucb, random, std, ideal, uidal, GSx, GSy, iGS, sGSx. 
        QBC must implemented separately.
    estimated_observation_y : nd_array
        Values of the objective for already evaluated data.
    alpha : float
        Hyperparameter for the acquisition function.
    mean : nd_array
        Mean of the prediction for all data points.
    std : nd_array
        Mean of the prediction for all data points. 
    n_data : int
        Number of data points
    data_indices : nd_array
        Array with indices of already evaluated data points from a pool of data
    pool : nd_array
        Pool of data
    custom_acfn_input : dict
        Dictionary that contains which information is used by a custom acquisition function.
    rng : Numpy Random Number Generator
        Random Number Generator object from numpy.

    Returns
    -------
    nd_array
        The values of the acquisition function for all data points in the pool.

    Raises
    ------
    Exception
        Acquisition function is not implemented.
    """
    acquisition = None
    if acquisition_function == 'poi':
        acquisition = POI(mean, std, opt=np.max(estimated_observation_y), max=True, alpha=alpha)
    elif acquisition_function == 'ei':
        acquisition = EI(mean, std, opt=np.max(estimated_observation_y), max=True, alpha=alpha)
    elif acquisition_function == 'ucb':
        acquisition = UCB(mean, std, alpha=alpha)
    elif acquisition_function == 'random':
        acquisition = rng.rand(n_data)
    elif acquisition_function == 'std':
        acquisition = std
    elif acquisition_function =='ideal':
        y_true = np.zeros(len(pool))
        y_true[data_indices] = estimated_observation_y
        acquisition = IDEAL(data_indices, pool, mean, y_true, alpha)
    elif acquisition_function =='uidal':
        acquisition = UIDAL(data_indices, pool, std, alpha)
    elif acquisition_function == 'GSx':
        acquisition = GSx(data_indices, pool)
    elif acquisition_function == 'GSy':
        acquisition = GSy(mean, estimated_observation_y)
    elif acquisition_function == 'iGS':
        acquisition = iGS(data_indices, pool, mean, estimated_observation_y)
    elif acquisition_function == 'SGSx':
        acquisition = SGSx(data_indices, pool, std, alpha)
    elif callable(acquisition_function):
        custom_args = {}
        if 'mean' in custom_acfn_input:
            custom_args['mean'] = mean
        if 'std' in custom_acfn_input:
            custom_args['std'] = std
        if 'pool' in custom_acfn_input:
            custom_args['pool'] = pool
        if 'data_indices' in custom_acfn_input:
            custom_args['data_indices'] = data_indices
        if 'alpha' in custom_acfn_input:
            custom_args['alpha'] = alpha
        if 'y_true' in custom_acfn_input:
            custom_args['y_true'] = y_true
        if 'y_est' in custom_acfn_input:
            custom_args['y_est'] = estimated_observation_y

        acquisition = acquisition_function(custom_args)
    else:
        raise KeyError('Acquisition function "{}" not implemented'.format(acquisition_function))
    return acquisition

def step_continuous(acquisition_function,
                    opt_method,
                    regression_model,
                    estimated_observation_y,
                    estimated_sample_x,
                    estimated_sample_x_poly,
                    custom_acfn_input,
                    alpha,
                    sampler,
                    lim,
                    dimensions,
                    poly_x,
                    n_jobs,
                    pso_options,
                    rng):
    """Evaluating an acquisition function within a continuous intervall.

    Parameters
    ----------
    acquisition_function : str or callable
        Name of the acquisition function or a callable. Choose from: ei, poi, ucb, 
        random, std, ideal, uidal, GSx, GSy, iGS, sGSx, qbc
    opt_method : str
        Choose from scipy (using lbfgs algorithm) or PSO (using PySwarms).
    regression_model : scikit-learn model
        Trained scikit-Learn model
    estimated_observation_y : nd_array
        Values of the objective for already evaluated data.
    estimated_sample_x : nd_array
        Values of the data points for already evaluated data.
    estimated_sample_x_poly : nd_array
        Values of the data points for already evaluated and to PolynomialFeatures 
        transformed data points. This is used only for Linear Regression models.
    custom_acfn_input : dict
        Dictionary that contains which information is used by a custom acquisition function.
    alpha : float
        Hyperparameter for the acquisition function.
    sampler : LatinHypercube 
        Scipys LatinHypercube object.
    lim : list
        Boundaries for model evaluation. 
    dimensions : int
        Number of dimensions of the features.
    poly_x : bool
        Indicates if polynomial features are used.
    n_jobs : int
        Number of jobs for PySwarms.
    pso_options : dict
        Options for PySwarms GlobalBestOptimizer.
    rng : Numpy Random Number Generator
        Random Number Generator object from numpy.

    Returns
    -------
    nd_array
        Feature values for which the acquisition function has its maximum.
    float
        Value of the found maximum of the acquisition function.

    Raises
    ------
    Exception
        Acquisition function not implemented.
    Exception
        Optimization method not implemented.
    """
    cost = None
    new_x = None
    if acquisition_function == 'random':
        x0_unscaled = sampler.random(1)[0]
        x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
        new_x = x0

    elif opt_method == 'scipy':
        lim_t = np.array(lim).T
        x0_unscaled = sampler.random(1)[0]
        x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)

        #Check for the acquisition functions
        if acquisition_function == 'ei':
            res = minimize(EI_con, x0=x0, args=(regression_model,
                                                np.max(estimated_observation_y), alpha),
                        bounds=lim_t)
        elif acquisition_function == 'poi':
            res = minimize(POI_con, x0=x0, args=(regression_model,
                                                 np.max(estimated_observation_y), alpha),
                        bounds=lim_t)
        elif acquisition_function == 'ucb':
            res = minimize(UCB_con, x0=x0, args=(regression_model, alpha),
                        bounds=lim_t)
        elif acquisition_function == 'ideal':
            res = minimize(IDEAL_con, x0=x0, args=(estimated_sample_x, regression_model,
                                                   estimated_observation_y, lim, alpha, poly_x),
                        bounds=lim_t)
        elif acquisition_function == 'uidal':
            res = minimize(UIDAL_con, x0=x0, args=(estimated_sample_x,
                                                   regression_model, alpha),
                        bounds=lim_t)
        elif acquisition_function == 'std':
            res = minimize(std_con, x0=x0, args=(regression_model),
                        bounds=lim_t)
        elif acquisition_function == 'GSx':
            res = minimize(GSx_con, x0=x0, args=(estimated_sample_x),
                    bounds=lim_t)
        elif acquisition_function == 'GSy':
            res = minimize(GSy_con, x0=x0, args=(estimated_observation_y,
                                                 regression_model, poly_x),
                    bounds=lim_t)
        elif acquisition_function == 'iGS':
            res = minimize(iGS_con, x0=x0, args=(estimated_sample_x, estimated_observation_y,
                                                 regression_model, poly_x),
                    bounds=lim_t)
        elif acquisition_function == 'SGSx':
            res = minimize(SGSx_con, x0=x0, args=(estimated_sample_x,
                                                  regression_model, alpha),
                    bounds=lim_t)
        elif acquisition_function == 'qbc':
            models = []
            for _ in range(alpha):
                train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                if isinstance(regression_model, LinearRegression):
                    regression_model.fit(estimated_sample_x_poly[train_index],
                                         estimated_observation_y[train_index])
                else:
                    regression_model.fit(estimated_sample_x[train_index],
                                         estimated_observation_y[train_index])
                models.append(copy.deepcopy(regression_model))
            res = minimize(QBC_con, x0=x0, args=(models, poly_x),
                    bounds=lim_t)
        elif callable(acquisition_function):
            custom_args = []
            if 'x' in custom_acfn_input:
                custom_args.append(estimated_sample_x)
            if 'y' in custom_acfn_input:
                custom_args.append(estimated_observation_y)
            if 'max_y' in custom_acfn_input:
                custom_args.append(np.max(estimated_observation_y))
            if 'regression_model' in custom_acfn_input:
                custom_args.append(regression_model)
            if 'lim' in custom_acfn_input:
                custom_args.append(lim)
            if 'alpha' in custom_acfn_input:
                custom_args.append(alpha)
            if 'poly_x' in custom_acfn_input:
                custom_args.append(poly_x)
            custom_args = tuple(custom_args)
            res = minimize(acquisition_function, x0=x0, args=custom_args,
                    bounds=lim_t)

        else:
            raise KeyError('Acquisition function "{}" not implemented'.format(acquisition_function))

        new_x = res.x
        cost = res.fun

    #Simple Particle Swarm Optimization
    elif opt_method == 'PSO':
        if not isinstance(pso_options, dict):
            pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p':dimensions*10, 'i':200}
        else:
            dict_keys = pso_options.keys()
            if 'c1' not in dict_keys  or 'c2' not in dict_keys or 'w' not in dict_keys:
                if 'p' not in dict_keys or 'i' not in dict_keys:
                    raise KeyError('c1, c2, w, p and i keys must be in pso_options.')
        n_particles = int(pso_options['p'])
        n_iters = int(pso_options['i'])

        lb = lim[0]
        ub = lim[1]
        bounds = [lb,ub]
        init_pos_unscaled = sampler.random(n_particles)
        init_pos = scale(init_pos_unscaled, *lim)
        np.random.seed(rng.randint(0,1000000))
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions,
                                  options=pso_options, bounds=bounds, init_pos=init_pos)
        if acquisition_function == 'ei':
            cost, new_x = optimizer.optimize(EI_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs, model=regression_model,
                                                opt=np.max(estimated_observation_y), alpha=alpha)
        elif acquisition_function == 'poi':
            cost, new_x = optimizer.optimize(POI_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs, model=regression_model,
                                                opt=np.max(estimated_observation_y), alpha=alpha)
        elif acquisition_function == 'ucb':
            cost, new_x = optimizer.optimize(UCB_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs,
                                                model=regression_model, alpha=alpha)
        elif acquisition_function == 'ideal':
            cost, new_x = optimizer.optimize(IDEAL_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs,
                                                x_samples=estimated_sample_x,
                                                model=regression_model,
                                                y_true=estimated_observation_y,
                                                lim=lim, alpha=alpha, poly_x=poly_x)
        elif acquisition_function == 'uidal':
            cost, new_x = optimizer.optimize(UIDAL_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs,
                                                x_samples=estimated_sample_x,
                                                model=regression_model, alpha=alpha)
        elif acquisition_function == 'std':
            cost, new_x = optimizer.optimize(std_con, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, model=regression_model)
        elif acquisition_function == 'GSx':
            cost, new_x = optimizer.optimize(GSx_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs, x_sample=estimated_sample_x)
        elif acquisition_function == 'GSy':
            cost, new_x = optimizer.optimize(GSy_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs,
                                                y_sample=estimated_observation_y,
                                                model = regression_model, poly_x=poly_x)
        elif acquisition_function == 'iGS':
            cost, new_x = optimizer.optimize(iGS_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs, x_sample=estimated_sample_x,
                                                y_sample=estimated_observation_y,
                                                model=regression_model, poly_x=poly_x)
        elif acquisition_function == 'SGSx':
            cost, new_x = optimizer.optimize(SGSx_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs, x_sample=estimated_sample_x,
                                                model=regression_model, alpha=alpha)
        elif acquisition_function == 'qbc':
            models = []
            for _ in range(alpha):
                train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                if isinstance(regression_model, LinearRegression):
                    regression_model.fit(estimated_sample_x_poly[train_index],
                                         estimated_observation_y[train_index])
                else:
                    regression_model.fit(estimated_sample_x_poly[train_index],
                                         estimated_observation_y[train_index])
                models.append(copy.deepcopy(regression_model))
            cost, new_x = optimizer.optimize(QBC_con, iters=n_iters, verbose=False,
                                                n_processes=n_jobs, models=models, poly_x=poly_x)
        elif callable(acquisition_function):
            custom_args = {}
            if 'x' in custom_acfn_input:
                custom_args['x']=estimated_sample_x
            if 'y' in custom_acfn_input:
                custom_args['y']=estimated_observation_y
            if 'max_y' in custom_acfn_input:
                custom_args['max_y']=np.max(estimated_observation_y)
            if 'regression_model' in custom_acfn_input:
                custom_args['regression_model'] = regression_model
            if 'lim' in custom_acfn_input:
                custom_args['lim']=lim
            if 'alpha' in custom_acfn_input:
                custom_args['alpha']=alpha
            if 'poly_x' in custom_acfn_input:
                custom_args['poly_x'] = poly_x

            cost, new_x = optimizer.optimize(acquisition_function, iters=n_iters,
                                             verbose=False, n_processes=n_jobs, **custom_args)

        else:
            raise KeyError('Acquisition function "{}" not implemented'.format(acquisition_function))

    else:
        raise KeyError('Optimization method "{}" not implemented'.format(opt_method))

    return new_x, cost

def step_continous_multi(
                    acquisition_function,
                    opt_method,
                    regression_models,
                    aggregation_function,
                    estimated_observation_y,
                    estimated_observation_y_aggregated,
                    estimated_sample_x,
                    estimated_sample_x_poly,
                    custom_acfn_input,
                    alpha,
                    sampler,
                    lim,
                    dimensions,
                    poly_x,
                    n_jobs,
                    pso_options,
                    rng,
                    n_models,
                    **kwargs):

    """Evaluating an acquisition function within a continuous intervall with the 
    support for aggregation functions to combine multiple objectives and optimize them together.

    Parameters
    ----------
    acquisition_function : str or callable
        Name of the acquisition function or a callable. Choose from: ei, poi, ucb,
        random, std, ideal, uidal, GSx, GSy, iGS, sGSx, qbc.
    opt_method : str
        Choose from scipy (using lbfgs algorithm) or PSO (using PySwarms).
    regression_models : list of scikit-learn model
        Trained scikit-Learn models. One is needed for each objective.
    aggregation_function : callable
        Function that combines all individual objectives together.
    estimated_observation_y : nd_array
        Values of the objective for already evaluated data.
    estimated_observation_y_aggregated : nd_array
        Values of the aggregated objective for already evaluated data.
    estimated_sample_x : nd_array
        Values of the data points for already evaluated data.
    estimated_sample_x_poly : nd_array
        Values of the data points for already evaluated and to PolynomialFeatures 
        transformed data points. This is used only for Linear Regression models.
    custom_acfn_input : dict
        Dictionary that contains which information is used by a custom acquisition function.
    alpha : float
        Hyperparameter for the acquisition function.
    sampler : LatinHypercube 
        Scipys LatinHypercube object.
    lim : list
        Boundaries for model evaluation. 
    dimensions : int
        Number of dimensions of the features.
    poly_x : bool
        Indicates if polynomial features are used.
    n_jobs : int
        Number of jobs for PySwarms.
    pso_options : dict
        Options for PySwarms GlobalBestOptimizer.
    rng : Numpy Random Number Generator
        Random Number Generator object from numpy.
    n_models : int
        Number of objectives and models.

    Returns
    -------
    nd_array
        Feature values for which the acquisition function has its maximum.
    float
        Value of the found maximum of the acquisition function.

    Raises
    ------
    Exception
        Acquisition function not implemented.
    Exception
        Optimization method not implemented.
    """
    cost = None
    new_x = None
    if acquisition_function == 'random':
        x0_unscaled = sampler.random(1)[0]
        x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
        new_x = x0

    elif opt_method == 'scipy':
        lim_t = np.array(lim).T
        x0_unscaled = sampler.random(1)[0]
        x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)

        optargs = kwargs.values()

        #Check for the acquisition functions
        if acquisition_function == 'ei':
            res = minimize(EI_multi, x0=x0, args=(regression_models, aggregation_function,
                                                  np.max(estimated_observation_y_aggregated),
                                                  alpha, True, *optargs),
                        bounds=lim_t)
        elif acquisition_function == 'poi':
            res = minimize(POI_multi, x0=x0, args=(regression_models, aggregation_function,
                                                   np.max(estimated_observation_y_aggregated),
                                                   alpha, True, *optargs),
                        bounds=lim_t)
        elif acquisition_function == 'ucb':
            res = minimize(UCB_multi, x0=x0, args=(regression_models, aggregation_function,
                                                   alpha, *optargs),
                        bounds=lim_t)
        elif acquisition_function == 'ideal':
            res = minimize(IDEAL_multi, x0=x0, args=(estimated_sample_x, regression_models,
                                                     estimated_observation_y_aggregated, lim,
                                                     aggregation_function, alpha, poly_x, *optargs),
                        bounds=lim_t)
        elif acquisition_function == 'uidal':
            res = minimize(UIDAL_multi, x0=x0, args=(estimated_sample_x, regression_models,
                                                     aggregation_function, alpha, *optargs),
                        bounds=lim_t)
        elif acquisition_function == 'std':
            res = minimize(std_multi, x0=x0, args=(regression_models, aggregation_function,
                                                   *optargs),
                        bounds=lim_t)

        elif acquisition_function == 'GSx':
            res = minimize(GSx_multi, x0=x0, args=(estimated_sample_x),
                    bounds=lim_t)
        elif acquisition_function == 'GSy':
            res = minimize(GSy_multi, x0=x0, args=(estimated_observation_y_aggregated,
                                                   regression_models, aggregation_function,
                                                   poly_x, *optargs),
                    bounds=lim_t)
        elif acquisition_function == 'iGS':
            res = minimize(iGS_multi, x0=x0, args=(estimated_sample_x,
                                                   estimated_observation_y_aggregated,
                                                   regression_models, aggregation_function,
                                                   poly_x, *optargs),
                    bounds=lim_t)
        elif acquisition_function == 'SGSx':
            res = minimize(SGSx_multi, x0=x0, args=(estimated_sample_x, regression_models,
                                                    aggregation_function, alpha, *optargs),
                    bounds=lim_t)

        elif acquisition_function == 'qbc':
            s_models = []
            for i in range(n_models):
                alpha_models = []
                for _ in range(alpha):
                    train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                    if isinstance(regression_models[i], LinearRegression):
                        regression_models[i].fit(estimated_sample_x_poly[train_index],
                                                 estimated_observation_y[i][train_index])
                    else:
                        regression_models[i].fit(estimated_sample_x[train_index],
                                                 estimated_observation_y[i][train_index])
                    alpha_models.append(copy.deepcopy(regression_models[i]))
                s_models.append(alpha_models)
            res = minimize(QBC_multi, x0=x0, args=(s_models, aggregation_function, poly_x,
                                                   *optargs),
                    bounds=lim_t)
        elif callable(acquisition_function):
            custom_args = []
            if 'x' in custom_acfn_input:
                custom_args.append(estimated_sample_x)
            if 'y' in custom_acfn_input:
                custom_args.append(estimated_observation_y_aggregated)
            if 'max_y' in custom_acfn_input:
                custom_args.append(np.max(estimated_observation_y_aggregated))
            if 'regression_models' in custom_acfn_input:
                custom_args.append(regression_models)
            if 'lim' in custom_acfn_input:
                custom_args.append(lim)
            if 'alpha' in custom_acfn_input:
                custom_args.append(alpha)
            if 'poly_x' in custom_acfn_input:
                custom_args.append(poly_x)

            #Aggregation function is always used and provided as the last argument
            custom_args.append(aggregation_function)
            #Provide arguments for the aggregation function
            for optarg in optargs:
                custom_args.append(optarg)
            custom_args = tuple(custom_args)
            res = minimize(acquisition_function, x0=x0, args=custom_args,
                           bounds=lim_t)
        else:
            raise KeyError('Acquisition function "{}" not implemented'.format(acquisition_function))

        new_x = res.x
        cost = res.fun

    #Simple Particle Swarm Optimization
    elif opt_method == 'PSO':
        if not isinstance(pso_options, dict):
            pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p':dimensions*10, 'i':200}
        else:
            dict_keys = pso_options.keys()
            if 'c1' not in dict_keys  or 'c2' not in dict_keys or 'w' not in dict_keys:
                if 'p' not in dict_keys or 'i' not in dict_keys:
                    raise KeyError('c1, c2, w, p and i keys must be in pso_options.')
        n_particles = int(pso_options['p'])
        n_iters = int(pso_options['i'])

        lb = lim[0]
        ub = lim[1]
        bounds = [lb,ub]
        init_pos_unscaled = sampler.random(n_particles)
        init_pos = scale(init_pos_unscaled, *lim)
        np.random.seed(rng.randint(0,1000000))
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions,
                                  options=pso_options, bounds=bounds, init_pos = init_pos)
        if acquisition_function == 'ei':
            cost, new_x = optimizer.optimize(EI_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, model=regression_models,
                                             aggregation_function=aggregation_function,
                                             opt=np.max(estimated_observation_y_aggregated),
                                             alpha=alpha, **kwargs)
        elif acquisition_function == 'poi':
            cost, new_x = optimizer.optimize(POI_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, model=regression_models,
                                             aggregation_function=aggregation_function,
                                             opt=np.max(estimated_observation_y_aggregated),
                                             alpha=alpha, **kwargs)
        elif acquisition_function == 'ucb':
            cost, new_x = optimizer.optimize(UCB_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, model=regression_models,
                                             aggregation_function=aggregation_function,
                                             alpha=alpha, **kwargs)
        elif acquisition_function == 'ideal':
            cost, new_x = optimizer.optimize(IDEAL_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, x_samples=estimated_sample_x,
                                             model=regression_models,
                                             y_true=estimated_observation_y_aggregated, lim=lim,
                                             aggregation_function=aggregation_function,
                                             alpha=alpha, poly_x=poly_x, **kwargs)

        elif acquisition_function == 'uidal':
            cost, new_x = optimizer.optimize(UIDAL_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, x_samples=estimated_sample_x,
                                             model=regression_models,
                                             aggregation_function=aggregation_function,
                                             alpha=alpha, **kwargs)
        elif acquisition_function == 'std':
            cost, new_x = optimizer.optimize(std_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, model=regression_models,
                                             aggregation_function=aggregation_function,
                                             **kwargs)

        elif acquisition_function == 'GSx':
            cost, new_x = optimizer.optimize(GSx_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, x_sample=estimated_sample_x)
        elif acquisition_function == 'GSy':
            cost, new_x = optimizer.optimize(GSy_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs,
                                             y_sample=estimated_observation_y_aggregated,
                                             model = regression_models,
                                             aggregation_function=aggregation_function,
                                             poly_x = poly_x, **kwargs)
        elif acquisition_function == 'iGS':
            cost, new_x = optimizer.optimize(iGS_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, x_sample=estimated_sample_x,
                                             y_sample=estimated_observation_y_aggregated,
                                             model=regression_models,
                                            aggregation_function=aggregation_function,
                                            poly_x = poly_x, **kwargs)
        elif acquisition_function == 'SGSx':
            cost, new_x = optimizer.optimize(SGSx_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, x_sample=estimated_sample_x,
                                             model=regression_models,
                                             aggregation_function=aggregation_function,
                                             alpha=alpha, **kwargs)
        elif acquisition_function == 'qbc':
            s_models = []
            for i in range(n_models):
                alpha_models = []
                for _ in range(alpha):
                    train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                    if isinstance(regression_models[i], LinearRegression):
                        regression_models[i].fit(estimated_sample_x_poly[train_index],
                                                 estimated_observation_y[i][train_index])
                    else:
                        regression_models[i].fit(estimated_sample_x[train_index],
                                                 estimated_observation_y[i][train_index])
                    alpha_models.append(copy.deepcopy(regression_models[i]))
                s_models.append(alpha_models)
            cost, new_x = optimizer.optimize(QBC_multi, iters=n_iters, verbose=False,
                                             n_processes=n_jobs, models=s_models,
                                             aggregation_function=aggregation_function,
                                             poly_x = poly_x, **kwargs)
        elif callable(acquisition_function):
            custom_args = {}
            if 'x' in custom_acfn_input:
                custom_args['x']=estimated_sample_x
            if 'y' in custom_acfn_input:
                custom_args['y']=estimated_observation_y_aggregated
            if 'max_y' in custom_acfn_input:
                custom_args['max_y']=np.max(estimated_observation_y_aggregated)
            if 'regression_models' in custom_acfn_input:
                custom_args['regression_models'] = regression_models
            if 'lim' in custom_acfn_input:
                custom_args['lim']=lim
            if 'alpha' in custom_acfn_input:
                custom_args['alpha']=alpha
            if 'poly_x' in custom_acfn_input:
                custom_args['poly_x'] = poly_x

            cost, new_x = optimizer.optimize(acquisition_function, iters=n_iters, verbose=False,
                                             n_processes=n_jobs,
                                             aggregation_function=aggregation_function,
                                             **custom_args, **kwargs)
        else:
            raise KeyError('Acquisition function "{}" not implemented'.format(acquisition_function))

    else:
        raise KeyError('Optimization method "{}" not implemented'.format(opt_method))

    return new_x, cost
