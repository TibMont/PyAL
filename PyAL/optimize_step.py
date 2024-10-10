# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import numpy as np

from scipy.stats.qmc import scale

from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression

from PyAL.acfn_continuous import EI_con, POI_con, UCB_con, IDEAL_con, GSx_con, GSy_con, iGS_con, QBC_con, SGSx_con, std_con, UIDAL_con
from PyAL.acfn_continuous_multi import EI_multi, POI_multi, UCB_multi, IDEAL_multi, GSx_multi, GSy_multi, iGS_multi, QBC_multi, SGSx_multi, std_multi, UIDAL_multi
from PyAL.acfn_discrete import EI, POI, UCB, IDEAL, GSx, GSy, iGS, QBC, SGSx, UIDAL

import copy
from pyswarms.single.global_best import GlobalBestPSO

import sys
import os
import warnings

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')


def step_discrete(
                    acquisition_function, 
                    estimated_observation_y, 
                    alpha, mean, std,
                    n_data, data_indices, pool,
                    custom_acfn_input,
                    rng,
):
    
    if acquisition_function == 'poi':
        acquisition = POI(mean, std, opt=np.max(estimated_observation_y), max=True, alpha=alpha)
    elif acquisition_function == 'ei':
        acquisition = EI(mean, std, opt=np.max(estimated_observation_y), max=True, alpha=alpha)
    elif acquisition_function == 'ucb':
        acquisition = UCB(mean, std, alpha=alpha)
        acquisition += 1000
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
        raise Exception('Acquisition function not implemented')
    
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
    
    if acquisition_function == 'random':
        x0_unscaled = sampler.random(1)[0]
        x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
        new_x = x0

    elif opt_method == 'scipy':
        lim_t = np.array(lim).T
        x0_unscaled = sampler.random(1)[0]
        x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
        
        #Check for the acquisition functions
        #TODO: implement custom acquisition function
        if acquisition_function == 'ei':
            res = minimize(EI_con, x0=x0, args=(regression_model, np.max(estimated_observation_y), alpha), 
                        bounds=lim_t)
        elif acquisition_function == 'poi':
            res = minimize(POI_con, x0=x0, args=(regression_model, np.max(estimated_observation_y), alpha), 
                        bounds=lim_t)
        elif acquisition_function == 'ucb':
            res = minimize(UCB_con, x0=x0, args=(regression_model, alpha), 
                        bounds=lim_t)
        elif acquisition_function == 'ideal':
            res = minimize(IDEAL_con, x0=x0, args=(estimated_sample_x, regression_model, estimated_observation_y, lim, alpha, poly_x), 
                        bounds=lim_t)
        elif acquisition_function == 'uidal':
            res = minimize(UIDAL_con, x0=x0, args=(estimated_sample_x, regression_model, alpha), 
                        bounds=lim_t)
        elif acquisition_function == 'std':
            res = minimize(std_con, x0=x0, args=(regression_model), 
                        bounds=lim_t)
        elif acquisition_function == 'GSx':
            res = minimize(GSx_con, x0=x0, args=(estimated_sample_x),
                    bounds=lim_t)
        elif acquisition_function == 'GSy':
            res = minimize(GSy_con, x0=x0, args=(estimated_observation_y, regression_model, poly_x),
                    bounds=lim_t)
        elif acquisition_function == 'iGS':
            res = minimize(iGS_con, x0=x0, args=(estimated_sample_x, estimated_observation_y, regression_model, poly_x),
                    bounds=lim_t)
        elif acquisition_function == 'SGSx':
            res = minimize(SGSx_con, x0=x0, args=(estimated_sample_x, regression_model, alpha),
                    bounds=lim_t)
        elif acquisition_function == 'qbc':
            models = []
            for _ in range(alpha):
                train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                if isinstance(regression_model, LinearRegression):
                    regression_model.fit(estimated_sample_x_poly[train_index], estimated_observation_y[train_index])
                else:
                    regression_model.fit(estimated_sample_x[train_index], estimated_observation_y[train_index])
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
            raise Exception('Acquisition function not implemented')
    
        new_x = res.x
        cost = res.fun

    #Simple Particle Swarm Optimization
    #TODO: enable to customly choose hyperparameters
    elif opt_method == 'PSO':
        if not isinstance(pso_options, dict):
            pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p':dimensions*10, 'i':200}
        else:
            dict_keys = pso_options.keys()
            if 'c1' not in dict_keys  or 'c2' not in dict_keys or 'w' not in dict_keys or 'p' not in dict_keys or 'i' not in dict_keys:
                raise Exception('c1, c2, w, p and i keys must be in pso_options.')
        n_particles = int(pso_options['p'])
        n_iters = int(pso_options['i'])

        lb = lim[0]
        ub = lim[1]
        bounds = [lb,ub]
        init_pos_unscaled = sampler.random(n_particles)
        init_pos = scale(init_pos_unscaled, *lim)
        np.random.seed(rng.randint(0,1000000))
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=pso_options, 
                            bounds=bounds, init_pos=init_pos)
        if acquisition_function == 'ei':
            cost, new_x = optimizer.optimize(EI_con, iters=n_iters, verbose=False, n_processes=n_jobs, 
                                                model=regression_model, 
                                                opt=np.max(estimated_observation_y), alpha=alpha)
        elif acquisition_function == 'poi':
            cost, new_x = optimizer.optimize(POI_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                model=regression_model, 
                                                opt=np.max(estimated_observation_y), alpha=alpha)
        elif acquisition_function == 'ucb':
            cost, new_x = optimizer.optimize(UCB_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                model=regression_model, alpha=alpha)
        elif acquisition_function == 'ideal':
            cost, new_x = optimizer.optimize(IDEAL_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_samples=estimated_sample_x,
                                            model=regression_model, y_true=estimated_observation_y, lim=lim, alpha=alpha, poly_x=poly_x)
        elif acquisition_function == 'uidal':
            cost, new_x = optimizer.optimize(UIDAL_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_samples=estimated_sample_x,
                                            model=regression_model, alpha=alpha)
        elif acquisition_function == 'std':
            cost, new_x = optimizer.optimize(std_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                            model=regression_model)
        elif acquisition_function == 'GSx':
            cost, new_x = optimizer.optimize(GSx_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_sample=estimated_sample_x)
        elif acquisition_function == 'GSy':
            cost, new_x = optimizer.optimize(GSy_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                y_sample=estimated_observation_y,
                                                model = regression_model, poly_x=poly_x)
        elif acquisition_function == 'iGS':
            cost, new_x = optimizer.optimize(iGS_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_sample=estimated_sample_x,
                                                y_sample=estimated_observation_y, model=regression_model, poly_x=poly_x)
        elif acquisition_function == 'SGSx':
            cost, new_x = optimizer.optimize(SGSx_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_sample=estimated_sample_x, model=regression_model, alpha=alpha)
        elif acquisition_function == 'qbc':
            models = []
            for _ in range(alpha):
                train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                if isinstance(regression_model, LinearRegression):
                    regression_model.fit(estimated_sample_x_poly[train_index], estimated_observation_y[train_index])
                else:
                    regression_model.fit(estimated_sample_x_poly[train_index], estimated_observation_y[train_index])
                models.append(copy.deepcopy(regression_model))
            cost, new_x = optimizer.optimize(QBC_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                models=models, poly_x=poly_x)
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
            
            cost, new_x = optimizer.optimize(acquisition_function, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                **custom_args)
        
        else:
            raise Exception('Acquisition function not implemented')
        
    else:
        raise Exception('Optimization method not implemented')
    
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
        #TODO: implement custom acquisition function
        if acquisition_function == 'ei':
            res = minimize(EI_multi, x0=x0, args=(regression_models, aggregation_function, np.max(estimated_observation_y_aggregated), alpha, True, *optargs), 
                        bounds=lim_t)
        elif acquisition_function == 'poi':
            res = minimize(POI_multi, x0=x0, args=(regression_models, aggregation_function, np.max(estimated_observation_y_aggregated), alpha, True, *optargs), 
                        bounds=lim_t)
        elif acquisition_function == 'ucb':
            res = minimize(UCB_multi, x0=x0, args=(regression_models, aggregation_function, alpha, *optargs), 
                        bounds=lim_t)
        elif acquisition_function == 'ideal':
            res = minimize(IDEAL_multi, x0=x0, args=(estimated_sample_x, regression_models, estimated_observation_y_aggregated, lim, aggregation_function, alpha, poly_x, *optargs), 
                        bounds=lim_t)
        elif acquisition_function == 'uidal':
            res = minimize(UIDAL_multi, x0=x0, args=(estimated_sample_x, regression_models, aggregation_function, alpha, *optargs), 
                        bounds=lim_t)
        elif acquisition_function == 'std':
            res = minimize(std_multi, x0=x0, args=(regression_models, aggregation_function, *optargs), 
                        bounds=lim_t)
        
        elif acquisition_function == 'GSx':
            res = minimize(GSx_multi, x0=x0, args=(estimated_sample_x),
                    bounds=lim_t)
        elif acquisition_function == 'GSy':
            res = minimize(GSy_multi, x0=x0, args=(estimated_observation_y_aggregated, regression_models, aggregation_function, poly_x, *optargs),
                    bounds=lim_t)
        elif acquisition_function == 'iGS':
            res = minimize(iGS_multi, x0=x0, args=(estimated_sample_x, estimated_observation_y_aggregated, regression_models, aggregation_function, poly_x, *optargs),
                    bounds=lim_t)
        elif acquisition_function == 'SGSx':
            res = minimize(SGSx_multi, x0=x0, args=(estimated_sample_x, regression_models, aggregation_function, alpha, *optargs),
                    bounds=lim_t)    

        elif acquisition_function == 'qbc':
            S_models = []
            for i in range(n_models):
                alpha_models = []
                for _ in range(alpha):
                    train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                    if isinstance(regression_models[i], LinearRegression):
                        regression_models[i].fit(estimated_sample_x_poly[train_index], estimated_observation_y[i][train_index])
                    else:
                        regression_models[i].fit(estimated_sample_x[train_index], estimated_observation_y[i][train_index])
                    alpha_models.append(copy.deepcopy(regression_models[i]))
                S_models.append(alpha_models)
            res = minimize(QBC_multi, x0=x0, args=(S_models, aggregation_function, poly_x, *optargs),
                    bounds=lim_t)
        else:
            raise Exception('Acquisition function not implemented')
    
        new_x = res.x
        cost = res.fun

    #Simple Particle Swarm Optimization
    #TODO: enable to customly choose hyperparameters
    elif opt_method == 'PSO':
        if not isinstance(pso_options, dict):
            pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p':dimensions*10, 'i':200}
        else:
            dict_keys = pso_options.keys()
            if 'c1' not in dict_keys  or 'c2' not in dict_keys or 'w' not in dict_keys or 'p' not in dict_keys or 'i' not in dict_keys:
                raise Exception('c1, c2, w, p and i keys must be in pso_options.')
        n_particles = int(pso_options['p'])
        n_iters = int(pso_options['i'])

        lb = lim[0]
        ub = lim[1]
        bounds = [lb,ub]
        init_pos_unscaled = sampler.random(n_particles)
        init_pos = scale(init_pos_unscaled, *lim)
        np.random.seed(rng.randint(0,1000000))
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=pso_options, 
                            bounds=bounds, init_pos = init_pos)
        if acquisition_function == 'ei':
            cost, new_x = optimizer.optimize(EI_multi, iters=n_iters, verbose=False, n_processes=n_jobs, 
                                                model=regression_models, aggregation_function=aggregation_function,
                                                opt=np.max(estimated_observation_y_aggregated), alpha=alpha, **kwargs)
        elif acquisition_function == 'poi':
            cost, new_x = optimizer.optimize(POI_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                model=regression_models, aggregation_function=aggregation_function,
                                                opt=np.max(estimated_observation_y_aggregated), alpha=alpha, **kwargs)
        elif acquisition_function == 'ucb':
            cost, new_x = optimizer.optimize(UCB_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                model=regression_models, aggregation_function=aggregation_function, 
                                                alpha=alpha, **kwargs)
        elif acquisition_function == 'ideal':
            cost, new_x = optimizer.optimize(IDEAL_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_samples=estimated_sample_x,
                                            model=regression_models, y_true=estimated_observation_y_aggregated, lim=lim, aggregation_function=aggregation_function,
                                            alpha=alpha, poly_x=poly_x, **kwargs)
            
        elif acquisition_function == 'uidal':
            cost, new_x = optimizer.optimize(UIDAL_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_samples=estimated_sample_x,
                                            model=regression_models, aggregation_function=aggregation_function, alpha=alpha, **kwargs)
        elif acquisition_function == 'std':
            cost, new_x = optimizer.optimize(std_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                            model=regression_models, aggregation_function=aggregation_function, **kwargs)
            
        elif acquisition_function == 'GSx':
            cost, new_x = optimizer.optimize(GSx_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_sample=estimated_sample_x)
        elif acquisition_function == 'GSy':
            cost, new_x = optimizer.optimize(GSy_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                y_sample=estimated_observation_y_aggregated,
                                                model = regression_models, aggregation_function=aggregation_function, poly_x = poly_x, **kwargs)
        elif acquisition_function == 'iGS':
            cost, new_x = optimizer.optimize(iGS_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_sample=estimated_sample_x,
                                                y_sample=estimated_observation_y_aggregated, model=regression_models, 
                                                aggregation_function=aggregation_function, poly_x = poly_x, **kwargs)
        elif acquisition_function == 'SGSx':
            cost, new_x = optimizer.optimize(SGSx_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                x_sample=estimated_sample_x, model=regression_models, aggregation_function=aggregation_function, alpha=alpha, **kwargs)
        elif acquisition_function == 'qbc':
            S_models = []
            for i in range(n_models):
                alpha_models = []
                for _ in range(alpha):
                    train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                    if isinstance(regression_models[i], LinearRegression):
                        regression_models[i].fit(estimated_sample_x_poly[train_index], estimated_observation_y[i][train_index])
                    else:
                        regression_models[i].fit(estimated_sample_x[train_index], estimated_observation_y[i][train_index])
                    alpha_models.append(copy.deepcopy(regression_models[i]))
                S_models.append(alpha_models)
            cost, new_x = optimizer.optimize(QBC_multi, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                models=S_models, aggregation_function=aggregation_function, poly_x = poly_x, **kwargs)
        else:
            raise Exception('Acquisition function not implemented')
        
    else:
        raise Exception('Optimization method not implemented')
    
    return new_x, cost