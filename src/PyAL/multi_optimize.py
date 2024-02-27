import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import scale
from scipy.stats import norm

from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.model_selection import cross_validate
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import copy
from pyswarms.single.global_best import GlobalBestPSO

import sys
import os
import warnings

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')

def conductivity_aggregation_fn(x, delta_beta, uncert=False):

    if len(x.shape) == 1:
        x = x.reshape(1,-1).T

    #print(x.shape)
    #print(x)
    if uncert == False:
        conductivity = x[0,:] - delta_beta*x[1,:] - delta_beta*x[2,:]**2
        return conductivity

    else:
        uncertainty = x[0,:] + delta_beta*x[1,:] + delta_beta*x[2,:]**2
        return uncertainty



def GSx_con(x, x_sample):

    if len(x.shape) == 1:
        x = x.reshape(1,-1)

    n_samples = len(x_sample)
    n_pool = len(x)
    min_dist = np.zeros(n_pool)
    distances = np.zeros((n_pool, n_samples))

    if n_pool == 1:
        for i in range(n_samples):
            dist = np.sum((x-x_sample[i])**2)
            distances[:,i] = dist
    else:
        for i in range(n_samples):
            dist = np.sum((x-x_sample[i])**2, axis=-1)
            distances[:,i] = dist

    min_dist = np.min(distances, axis=-1)

    return -min_dist

def GSy_con(x, y_sample, model, aggregation_function, poly_x, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        y_individual = np.zeros((len(model), 1))
    else:
        y_individual = np.zeros((len(model), len(x)))
    for i in range(len(model)):
        if isinstance(poly_x, PolynomialFeatures):
            x_poly = poly_x.fit_transform(x)
            y_individual[i] = model[i].predict(x_poly)
        else:
            y_individual[i] = model[i].predict(x)
    
    if len(args) != 0:
        y = aggregation_function(y_individual, *args)
    else:
        y = aggregation_function(y_individual, **kwargs)

    n_samples = len(y_sample)
    #print('N samples')
    #print(n_samples)
    #print('y samples')
    #print(y_sample)
    #print('x')
    #print(x)
    #print('y')
    #print(y)
    n_pool = len(x)
    min_dist = np.zeros(n_pool)
    distances = np.zeros((n_pool, n_samples))

    if n_pool == 1:
        for i in range(n_samples):
            dist = (y-y_sample[i])**2
            distances[:,i] = dist
    else:
        for i in range(n_samples):
            dist = (y-y_sample[i])**2
            distances[:,i] = dist

    min_dist = np.min(distances, axis=-1)
    #print('Dist')
    #print(distances)
    #print('Min dist')
    #print(min_dist)
    #print()
    return -min_dist

def iGS_con(x, x_sample, y_sample, model, aggregation_function, poly_x=None, *args, **kwargs):
    gsx = GSx_con(x,x_sample)
    gsy = GSy_con(x, y_sample, model, aggregation_function, poly_x, *args, **kwargs)

    min_dist = gsx*gsy
    return -min_dist
    
def IDEAL_con_old(x, x_samples, model, aggregation_function, alpha=1, *args, **kwargs):
    #print(grid.shape)
    #print(grid)
    #print(uncertainty)
    if len(x.shape) == 1:
        n_pool = 1
        x = x.reshape(1,-1)
    else:
        n_pool = len(x)
    n_samples = len(x_samples)

    w = np.zeros([n_pool, n_samples])
    SiD = np.zeros(n_pool)
    Z = np.zeros(n_pool)
    SW = np.ones(n_pool)

    for i in range(n_samples):
        dist = np.sum((x-x_samples[i])**2, axis=-1)
        w[:,i]= np.exp(-dist)/dist
        SiD[:] += 1/dist

    Z = np.arctan(1/SiD)*2/np.pi

    SW = np.sum(w, axis=1)

    v = w/SW.reshape(-1,1)

    uncertainty_individual = np.zeros((len(model), n_pool))
    for i in range(len(model)):
        _, uncertainty_individual[i] = model[i].predict(x, return_std=True)

    if len(args) != 0:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, *args)
    else:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, **kwargs)

    vk = np.sum(v*uncertainty.reshape(-1,1), axis=1)
    ideal = vk+alpha*Z
    o = np.isnan(ideal)
    ideal[o] = 0
    return -ideal

def IDEAL_con(x, x_samples, model, y_true, lim, aggregation_function, alpha=1, poly_x=None, tol=1e-8, *args, **kwargs):
    if len(x.shape) == 1:
        n_pool = 1
        x = x.reshape(1,-1)
    else:
        n_pool = len(x)

    n_samples = len(x_samples)
    
    xmin = np.array(lim[0])
    xmax = np.array(lim[1])
    x_scaled = 2/(xmax-xmin) * (x - (xmax+xmin)/2)
    x_samples = 2/(xmax-xmin) * (x_samples - (xmax+xmin)/2)

    w = np.zeros([n_pool, n_samples])
    SiD = np.zeros(n_pool)
    Z = np.zeros(n_pool)
    SW = np.ones(n_pool)

    mask_idx = []
    identical = []

    for i in range(n_samples):
        dist = np.sum((x_scaled-x_samples[i])**2, axis=-1)

        mask = np.where(dist>tol)
        mask_rev = np.where(dist<=tol)[0]
        if len(mask_rev) > 0:
            for m in mask_rev:
                mask_idx.append(m)
            identical.append(i)

        w[:,i] = np.exp(-dist)/dist
        SiD+= 1/dist

    Z = np.arctan(1/SiD)*2/np.pi
    SW = np.sum(w[:,0:n_samples], axis=1)

    mask_idx = np.array(mask_idx, dtype=np.int32)
    identical = np.array(identical, dtype=np.int32)
    if len(mask_idx)>0:

        Z[mask_idx] = 0

    mean_individual = np.zeros((len(model), n_pool))

    for i in range(len(model)):
        if isinstance(poly_x, PolynomialFeatures):
            x_poly = poly_x.fit_transform(x)
            mean_individual[i,...] = model[i].predict(x_poly)
        else:
            mean_individual[i,...] = model[i].predict(x)
    
    if len(args) != 0:
        mean = aggregation_function(mean_individual, *args)
    else:
        mean = aggregation_function(mean_individual, **kwargs)

    if len(mean.shape) == 1:
        mean = mean.reshape(-1,1)
    Yhat = mean

    ff = np.zeros(n_pool)
    ny = mean.shape[1]

    Ymax = np.max(y_true)
    Ymin = np.min(y_true)
    Ymax += 1.e-10
    Yscale = (Ymax - Ymin) / 2.
    dY2 = (2 * Yscale) ** 2

    for i in range(ny):
        vk = w / SW.reshape(-1, 1)
        vk[mask_idx] = 0

        ff += np.sum((vk) * (
            (Yhat[:,i,np.newaxis] - y_true) ** 2), axis=1) / dY2
    
    ideal = ff+alpha*Z

    return -ideal


def UCB_con(x, model, aggregation_function, alpha=0.5, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        n_pool = 1
    else:
        n_pool = len(x)

    mean_individual = np.zeros((len(model), n_pool))
    uncertainty_individual = np.zeros((len(model), n_pool))
    for i in range(len(model)):
        mean_individual[i], uncertainty_individual[i] = model[i].predict(x, return_std=True)

    if len(args) != 0:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, *args)
        mean = aggregation_function(mean_individual, *args)
    else:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, **kwargs)   
        mean = aggregation_function(mean_individual, **kwargs)

    ucb = mean+alpha*uncertainty
    return -ucb

def POI_con(x, model, aggregation_function, opt, alpha, max=True, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        n_pool = 1
    else:
        n_pool = len(x)

    
    mean_individual = np.zeros((len(model), n_pool))
    uncertainty_individual = np.zeros((len(model), n_pool))
    for i in range(len(model)):
        mean_individual[i], uncertainty_individual[i] = model[i].predict(x, return_std=True)

    if len(args) != 0:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, *args)
        mean = aggregation_function(mean_individual, *args)
    else:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, **kwargs)
        mean = aggregation_function(mean_individual, **kwargs)

    if max==False:
        f_min = opt
        probs = norm.cdf((f_min-mean-alpha)/(uncertainty+1e-9))
    else:
        f_max = opt
        probs = norm.cdf((mean-f_max-alpha)/(uncertainty+1e-9))
    return -probs

def EI_con(x, model, aggregation_function, opt, alpha=0.5, max=True, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        n_pool = 1
    else:
        n_pool = len(x)
    
    mean_individual = np.zeros((len(model), n_pool))
    uncertainty_individual = np.zeros((len(model), n_pool))
    for i in range(len(model)):
        mean_individual[i], uncertainty_individual[i] = model[i].predict(x, return_std=True)

    if len(args) != 0:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, *args)
        mean = aggregation_function(mean_individual, *args)
    else:
        uncertainty = aggregation_function(uncertainty_individual, uncert=True, **kwargs)
        mean = aggregation_function(mean_individual, **kwargs)

    if max==False:
        f_min = opt
        cdf = norm.cdf((f_min-mean)/uncertainty)
        pdf = norm.pdf((f_min-mean)/uncertainty)
        ei = alpha*(f_min-mean)*cdf + (1-alpha)*uncertainty*pdf
    else:
        f_max = opt
        cdf = norm.cdf((mean-f_max)/uncertainty)
        pdf = norm.pdf((mean-f_max)/uncertainty)
        ei = alpha*(mean-f_max)*cdf + (1-alpha)*uncertainty*pdf
    return -ei

def QBC_con(x, models, aggregation_function, poly_x, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        n_pool = 1
    else:
        n_pool = len(x)

    mean_qbc_individual = np.zeros((len(models), len(models[0]), n_pool))
    #bootstrapping approach to train models
    for i in range(len(models)):
        model = models[i]
        for j, mod in enumerate(model):
            if isinstance(poly_x, PolynomialFeatures):
                x_poly = poly_x.fit_transform(x)
                m = mod.predict(x_poly)
            else:
                m = mod.predict(x)
            mean_qbc_individual[i,j,...] += m

    mean_qbc = np.zeros((len(models[0]), n_pool))

    for i in range(mean_qbc_individual.shape[1]):
        if len(args) != 0:
            mean_qbc[i] = aggregation_function(mean_qbc_individual[:,i,:], *args)
        else:
            mean_qbc[i] = aggregation_function(mean_qbc_individual[:,i,:], **kwargs)

    result = np.zeros(n_pool)
    for i in range(n_pool):
        result[i] = np.sum( mean_qbc[:,i] - np.mean(mean_qbc[:,i]) )

    return -result


def UIDAL_con(x, x_samples, model, aggregation_function, alpha=1, *args, **kwargs):
    #print(grid.shape)
    #print(grid)
    #print(uncertainty)
    if len(x.shape) == 1:
        n_pool = 1
        x = x.reshape(1,-1)
    else:
        n_pool = len(x)
    n_samples = len(x_samples)

    w = np.zeros([n_pool, n_samples])
    SiD = np.zeros(n_pool)
    Z = np.zeros(n_pool)
    SW = np.ones(n_pool)

    for i in range(n_samples):
        dist = np.sum((x-x_samples[i])**2, axis=-1)
        w[:,i]= np.exp(-dist)/dist
        SiD[:] += 1/dist

    Z = np.arctan(1/SiD)*2/np.pi

    SW = np.sum(w, axis=1)

    v = w/SW.reshape(-1,1)

    individual_uncertainty = np.zeros((len(model), n_pool))

    for i in range(len(model)):
        _, u = model[i].predict(x, return_std=True)
        individual_uncertainty[i,...] = u

    if len(args) != 0:
        uncertainty = aggregation_function(individual_uncertainty, uncert=True,  *args)
    else:
        uncertainty = aggregation_function(individual_uncertainty, uncert=True, **kwargs)
    
    vk = np.sum(v*uncertainty.reshape(-1,1), axis=1)
    ideal = vk+alpha*Z
    o = np.isnan(ideal)
    ideal[o] = 0
    return -ideal   

def SGSx_con(x, x_sample, model, aggregation_function, alpha=0.5, *args, **kwargs):
    if len(x.shape) == 1:
        n_pool = 1
        x = x.reshape(1,-1)
    else:
        n_pool = len(x)
    gsx = GSx_con(x,x_sample)

    individual_std = np.zeros((len(model), n_pool))

    for i in range(len(model)):
        m, s = model[i].predict(x, return_std=True)
        individual_std[i,...] = s

    if len(args) != 0:
        std = aggregation_function(individual_std, uncert=True,  *args)
    else:
        std = aggregation_function(individual_std, uncert=True, **kwargs)
    
    sgsx = np.power(std,alpha)*np.power(-gsx,(1-alpha))
    return -sgsx 

def std_con(x, model, aggregation_function, *args, **kwargs):
    if len(x.shape) == 1:
        n_pool = 1
        x = x.reshape(1,-1)
    else:
        n_pool = len(x)

    individual_std = np.zeros((len(model), n_pool))

    for i in range(len(model)):
        m, s = model[i].predict(x, return_std=True)
        individual_std[i,...] = s

    if len(args) != 0:
        std = aggregation_function(individual_std, uncert=True,  *args)
    else:
        std = aggregation_function(individual_std, uncert=True, **kwargs)

    return -std

########################################################################################################

def run_continuous_batch_learning_multi(models, 
           aggregation_function,
           regression_models,
           acquisition_function = 'ideal',
           opt_method = 'PSO',
           pool = None, 
           batch_size = 1,
           noise=0.1,
           initial_samples=2, 
           active_learning_steps=10,
           lim=[-1,1],
           alpha=0,
           n_jobs=1,
           random_state=None, 
           initialization='random',
           pso_options = None,
           fictive_noise_level=0,
           poly_degree = 3,
           calculate_test_metrics = True,
           verbose=True,
           single_update = False,
           **kwargs
           ):
    '''
    Multi-objective batch-mode Active Learning in continuous parameter space. This method is similar to 
    'run_continuous_batch_learning' but also enables to take into account multiple objective functions. 
    Multiple objectives are handeled by an aggregation function, which combines the individual output for each data point of 
    each objective into a single output value. The aggregation functions needs to be provided customly.
    For each objective function it is also possible to use an individual model. E.g. we can use Gaussian Process Regression for objective 1 and 
    Linear Regression for objective 2.

    The active learning acquisition functions are modified in order to also take the aggregation function as an argument. Therefore, active learning 
    acquisition functions from outside this module are not compatible.

    The algorithm generates initial data automatically using either a random approach or using the model-free GSx approach. 

    Parameters
    ----------
    models : List of model class
        Models to generate true data for each objective. One model per objective needs to be provided.
    aggregation_function : callable
        Function to aggregate multiple outputs for various objective functions. The function needs to get an np_array as input and also 
        needs a parameter 'uncert' which tells the function if it should aggregate uncertainty or not, in case we want to aggregate uncertainty different
        than the mean prediction.
    regression_models: sklearn model, list
        scikit-learn regression models. Either a single model can be given or a list of models with a length corresponding to the number of objectives.
    acquissition_function : str, optional
        Acquisition function. The default is 'ideal'.
    opt_method : str, optional
        The method to optimize the acquisition function. Choose from 'lbfgs' and 'PSO'. The default is 'PSO'.
    pool : nd_array, None, optional
        Array containing pool of data for testing. For 'None' a grid with 100 points in each dimension is created. 
        This is only used when 'calculate_test_metrics' is True. The default is 'None'.
    batch_size : int, optional
        Batch size for Active Learning. The model is updated only with true data when a batch is completed. The default value is 1.
    noise : float, optional
        Noise in observation. The default is 0.1.
    initial_samples : nd_array, int, optional
        If an integer is provided: Number of initial samples to draw. The default is 2.
        If an nd_array is provided: Initial data points. The parameter 'initialization' must be 'data'.
    active_learning_steps : int, optional
        Number of active learning steps to perform. The default is 10.
    lim : list, optional
        Boundaries for model evaluation. Only used when pool=None. The default is [-1,1].
    alpha : float
        Hyperparameter for acquisition function. The default is 0.
    n_jobs : int, optional
        Number of cores to use for parallel evaluation of PSO. Currently not used, PSO runs only in serial mode. The default is 1.
    random_state: int, optional
        Set random state. The default is None.
    initialization : str, optional
        Initialization method for generating initial data. Choose from 'random' and 'GSx'. 'random' uses Latin Hypercube sampling 
        to generate the initial dat points. 'GSx' draws randomly the first data point and then uses the model-free GSx method to sample
        the other initial data points. If data is chosen, the initial data is assumed to be provided by
        'initial_samples'. The default value is 'random'.
    pso_options : dict, optional
        Dictionary with parameters for the Particle Swarm Optimization. Only used when opt_method is 'PSO'. For 'None' default values are used. 
        ['c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p':dimensions*10, 'i':200]. c1 and c2 are swarm parmeters, w is the inertia, p gives the number
        of particles, i is the number of iterations. The default is 'None'.
    fictive_noise_level : float, optional
        Noise level for non-GPR models used for assuming predictions as true values to enable batch-wise learning.
    calculate_test_metrics : bool, optional
        Can be used for testing Active Learning algorithms for known models. Test metrics are calculated automatically. If 'False' no test metrics are 
        calculated and the AL runs in deployement mode.
    verbose : bool, optional
        Whether to print additional information. The default value is True.
    **kwargs : various, optional
        Keyword arguments for the aggregation function.

    Returns
    -------
    samples : nd_array
        Numpy array which contains the unscaled features for data points in the order in which they were selected.
    results : pd_DataFrame
        Pandas DataFrame containing the columns: 'm', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 
        'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'. Data is stored for each active learning step, 
        where 'm' gives the number of data used for training, 'mean_MSE_train' and 'mean_MSE_test' the MSE of the training and test set, respectively,
        and 'max_observation' the maximum value observed so far. MAE corresponds to the Mean Absolute Error and MaxE to the maximum absolute error.

    '''
    #Check consistency of parameters
    if isinstance(regression_models, list):
        if len(regression_models) != len(models):
            raise Exception('Inconsistent number of data and regression models: {}, {}'.format(len(models), len(regression_model)))
    else:
        regression_models = [copy.deepcopy(regression_models) for i in range(len(models))]
    
    for regression_model in regression_models:
        if isinstance(regression_model, LinearRegression):
            if acquisition_function not in ['random', 'GSx', 'GSy', 'iGS', 'ideal', 'qbc']:
                print('Acquisition functin is not implemented for Linear Regression: {}'.format(acquisition_function))
                exit()
        elif not isinstance(regression_model, GPR):
            if acquisition_function not in ['random', 'GSx', 'GSy', 'iGS', 'ideal', 'qbc']:
                print('Acquisition functin is not implemented for Non-GPR models: {}'.format(acquisition_function))
                exit()
        else:
            if acquisition_function not in ['random', 'GSx', 'GSy', 'iGS', 'ideal', 'qbc', 'ei', 'ucb', 'poi', 'std', 'uidal', 'SGSx']:
                print('Acquisition function not implemented: {}'.format(acquisition_function))
                exit()

    #Set random number generator
    rng = np.random.RandomState(seed=random_state)

    #Set polynomial feature transformer
    poly_transformer = PolynomialFeatures(degree=poly_degree)

    dimensions = models[0].n_features
    n_models = len(models)

    #Generate a pool of sample data points for testing
    if calculate_test_metrics == True:
        if not isinstance(pool, np.ndarray):
            x = []
            for i in range(dimensions):
                x.append(np.linspace(*lim, 10))

            pool = np.meshgrid(*x)
            pool = np.array(pool).T
            pool = pool.reshape(len(x[0]**dimensions), dimensions)

        pool_poly = poly_transformer.fit_transform(pool)

    #Check if noise is int or float, noise will be applied to every model individually
    if isinstance(noise, int) or isinstance(noise, float):
        noise_old = noise
        noise = [noise for _ in range(n_models)]
        if verbose:
            print('Noise converted: ')
            print('from {} to {}'.format(noise_old, noise))

    if calculate_test_metrics: 
        #Number of data points in pool
        n_data = len(pool)
        y_true = np.zeros((n_models, n_data))
        for i in range(n_models):
            model = models[i]
            y_true[i, ...] = model.evaluate(pool, noise = noise[i])

        y_true_aggregated = aggregation_function(y_true, **kwargs)


    #Generate initial data
    if initialization == 'random':
        if isinstance(initial_samples, int):
            sampler = LHS(d=dimensions)
            sample_x_unscaled = sampler.random(initial_samples)
            sample_x = scale(sample_x_unscaled, *lim)
        else:
            raise Exception('initial_samples must be an integer for initialization method random')
    elif initialization == 'GSx':
        if isinstance(initial_samples, int):
            sample_x, _ = run_continuous_batch_learning_multi(models, 
            aggregation_function, 
            regression_models,
            acquisition_function = 'GSx',
            opt_method = opt_method,
            pool = pool, 
            batch_size = 1,
            noise=noise,
            initial_samples=1, 
            active_learning_steps=initial_samples-1,
            lim=lim,
            alpha=alpha,
            n_jobs=n_jobs,
            random_state=random_state,
            initialization='random',
            pso_options=pso_options,
            poly_degree = poly_degree,
            calculate_test_metrics=False,
            verbose=False
            )
        else:
            raise Exception('initial_samples must be an integer for initialization method GSx')
    elif initialization == 'data':
        if isinstance(initial_samples, np.ndarray):
            sample_x = initial_samples
            initial_samples = len(sample_x)
        else:
            raise Exception('initial_samples must be a nd_array for initialization method data')
    else:
        raise Exception('Initialization method not implemented')

    observation_y = np.zeros((n_models, len(sample_x)))
    for i in range(n_models):
        observation_y[i, ...] = models[i].evaluate(sample_x, noise=noise[i])

    observation_y_aggregated = aggregation_function(observation_y, **kwargs)

    #To save the metrics
    scores_train = np.zeros((active_learning_steps+1,3))
    max_value = np.zeros((active_learning_steps+1,1))
    n_observations = np.linspace(initial_samples, initial_samples+(active_learning_steps)*batch_size,
                            active_learning_steps+1)
    
    if calculate_test_metrics:
        #To save the metrics
        scores_test = np.zeros((active_learning_steps+1,3))

    #Fit initial model
    mean = np.zeros((n_models, len(pool)))
    std = np.zeros((n_models, len(pool)))

    mean_train = np.zeros((n_models, len(sample_x)))
    std_train = np.zeros((n_models, len(sample_x)))

    for i in range(n_models):
        if isinstance(regression_models[i], LinearRegression):
            sample_x_poly = poly_transformer.fit_transform(sample_x)
            regression_models[i].fit(sample_x_poly, observation_y[i])
        else:
            regression_models[i].fit(sample_x, observation_y[i])

        #Initial model predictions for test set
        if calculate_test_metrics:  
            if isinstance(regression_models[i], GPR):
                mean[i,...], std[i,...] = regression_models[i].predict(pool,return_std=True)
            elif isinstance(regression_models[i], LinearRegression):
                mean[i,...] = regression_models[i].predict(pool_poly)
            else:
                mean[i,...] = regression_model[i].predict(pool)

        #Initial model predictions for training set
        if isinstance(regression_models[i], GPR):
            mean_train[i,...], std_train[i,...] = regression_models[i].predict(sample_x,return_std=True)
        elif isinstance(regression_models[i], LinearRegression):
            mean_train[i,...] = regression_models[i].predict(sample_x_poly)
        else:
            mean_train[i,...] = regression_models[i].predict(sample_x)
    
    #Save scores
    if calculate_test_metrics:
        mean_aggregated = aggregation_function(mean, **kwargs)
        scores_test[0,0] = mean_squared_error(y_true_aggregated, mean_aggregated)
        scores_test[0,1] = mean_absolute_error(y_true_aggregated, mean_aggregated)
        scores_test[0,2] = max_error(y_true_aggregated, mean_aggregated)

    mean_train_aggregated = aggregation_function(mean_train, **kwargs)
    scores_train[0,0] = mean_squared_error(observation_y_aggregated, mean_train_aggregated)
    scores_train[0,1] = mean_absolute_error(observation_y_aggregated, mean_train_aggregated)
    scores_train[0,2] = max_error(observation_y_aggregated, mean_train_aggregated)
    max_value[0,0] = np.max(observation_y_aggregated)


    #Active Learning loop starts here
    ###############################################################

    #Start active learning
    for a in range(active_learning_steps):

        #Save batch results separately, since observation is only estimated
        batch_sample = np.zeros((batch_size, dimensions))
        estimated_observation_y = observation_y.copy()
        estimated_sample_x = sample_x.copy()
        estimated_observation_y_aggregated = observation_y_aggregated.copy()
        for regression_model in regression_models:
            if isinstance(regression_model, LinearRegression):
                estimated_sample_x_poly = sample_x_poly.copy()
        
        for j in range(batch_size):

            #For the first sample in a batch we can use the model with which we evaluated the scores
            if j != 0:
                mean = np.zeros((n_models, len(pool)))
                std = np.zeros((n_models, len(pool)))

                for i in range(n_models):
                    if isinstance(regression_model, LinearRegression):
                        regression_models[i].fit(estimated_sample_x_poly, estimated_observation_y[i])
                    else:
                        regression_models[i].fit(estimated_sample_x, estimated_observation_y[i])
                    #mean[i, ...], std[i, ...] = regression_model.predict(pool,return_std=True)

                #mean_aggregated = aggregation_function(mean, **kwargs)

            #Choose from optimization routines
            #TODO: enable more customizability of parameters for optimization routines
                        
            if isinstance(regression_model, LinearRegression):
                poly_x = poly_transformer
            else:
                poly_x = None

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
                    res = minimize(EI_con, x0=x0, args=(regression_models, aggregation_function, np.max(estimated_observation_y_aggregated), alpha, True, *optargs), 
                                bounds=lim_t)
                elif acquisition_function == 'poi':
                    res = minimize(POI_con, x0=x0, args=(regression_models, aggregation_function, np.max(estimated_observation_y_aggregated), alpha, True, *optargs), 
                                bounds=lim_t)
                elif acquisition_function == 'ucb':
                    res = minimize(UCB_con, x0=x0, args=(regression_models, aggregation_function, alpha, *optargs), 
                                bounds=lim_t)
                elif acquisition_function == 'ideal':
                    res = minimize(IDEAL_con, x0=x0, args=(estimated_sample_x, regression_models, estimated_observation_y, lim, aggregation_function, alpha, poly_x, *optargs), 
                                bounds=lim_t)
                elif acquisition_function == 'uidal':
                    res = minimize(UIDAL_con, x0=x0, args=(estimated_sample_x, regression_models, aggregation_function, alpha, *optargs), 
                                bounds=lim_t)
                elif acquisition_function == 'std':
                    res = minimize(std_con, x0=x0, args=(regression_models, aggregation_function, *optargs), 
                                bounds=lim_t)
                
                elif acquisition_function == 'GSx':
                    res = minimize(GSx_con, x0=x0, args=(estimated_sample_x),
                            bounds=lim_t)
                elif acquisition_function == 'GSy':
                    res = minimize(GSy_con, x0=x0, args=(estimated_observation_y_aggregated, regression_models, aggregation_function, poly_x, *optargs),
                            bounds=lim_t)
                elif acquisition_function == 'iGS':
                    res = minimize(iGS_con, x0=x0, args=(estimated_sample_x, estimated_observation_y_aggregated, regression_models, aggregation_function, poly_x, *optargs),
                            bounds=lim_t)
                elif acquisition_function == 'SGSx':
                    res = minimize(SGSx_con, x0=x0, args=(estimated_sample_x, regression_models, aggregation_function, alpha, *optargs),
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
                    res = minimize(QBC_con, x0=x0, args=(S_models, aggregation_function, poly_x, *optargs),
                            bounds=lim_t)
                else:
                    raise Exception('Acquisition function not implemented')
            
                new_x = res.x

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
                optimizer = GlobalBestPSO(n_particles=30, dimensions=dimensions, options=pso_options, 
                                    bounds=bounds)
                if acquisition_function == 'ei':
                    cost, new_x = optimizer.optimize(EI_con, iters=200, verbose=False, n_processes=n_jobs, 
                                                     model=regression_models, aggregation_function=aggregation_function,
                                                     opt=np.max(estimated_observation_y_aggregated), alpha=alpha, **kwargs)
                elif acquisition_function == 'poi':
                    cost, new_x = optimizer.optimize(POI_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     model=regression_models, aggregation_function=aggregation_function,
                                                     opt=np.max(estimated_observation_y_aggregated), alpha=alpha, **kwargs)
                elif acquisition_function == 'ucb':
                    cost, new_x = optimizer.optimize(UCB_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     model=regression_models, aggregation_function=aggregation_function, 
                                                     alpha=alpha, **kwargs)
                elif acquisition_function == 'ideal':
                    cost, new_x = optimizer.optimize(IDEAL_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     x_samples=estimated_sample_x,
                                                    model=regression_models, y_true=estimated_observation_y, lim=lim, aggregation_function=aggregation_function,
                                                    alpha=alpha, poly_x=poly_x, **kwargs)
                    
                elif acquisition_function == 'uidal':
                    cost, new_x = optimizer.optimize(UIDAL_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                     x_samples=estimated_sample_x,
                                                    model=regression_models, aggregation_function=aggregation_function, alpha=alpha, **kwargs)
                elif acquisition_function == 'std':
                    cost, new_x = optimizer.optimize(std_con, iters=n_iters, verbose=False, n_processes=n_jobs,
                                                    model=regression_models, aggregation_function=aggregation_function, **kwargs)
                    
                elif acquisition_function == 'GSx':
                    cost, new_x = optimizer.optimize(GSx_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     x_sample=estimated_sample_x)
                elif acquisition_function == 'GSy':
                    cost, new_x = optimizer.optimize(GSy_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     y_sample=estimated_observation_y_aggregated,
                                                     model = regression_models, aggregation_function=aggregation_function, poly_x = poly_x, **kwargs)
                elif acquisition_function == 'iGS':
                    cost, new_x = optimizer.optimize(iGS_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     x_sample=estimated_sample_x,
                                                     y_sample=estimated_observation_y_aggregated, model=regression_models, 
                                                     aggregation_function=aggregation_function, poly_x = poly_x, **kwargs)
                elif acquisition_function == 'SGSx':
                    cost, new_x = optimizer.optimize(SGSx_con, iters=n_iters, verbose=False, n_processes=n_jobs,
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
                    cost, new_x = optimizer.optimize(QBC_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     models=S_models, aggregation_function=aggregation_function, poly_x = poly_x, **kwargs)
                else:
                    raise Exception('Acquisition function not implemented')
                
            else:
                raise Exception('Optimization method not implemented')

            #Assume estimated predictions

            mean_new = np.zeros((n_models, len(new_x.reshape(1,-1))))
            std_new = np.zeros((n_models, len(new_x.reshape(1,-1))))
            estimated_observation_new = np.zeros((n_models, len(new_x.reshape(1,-1))))

            for i in range(n_models):
                if isinstance(regression_models[i], GPR):
                    mean_new[i,...], std_new[i,...] = regression_models[i].predict(new_x.reshape(1,-1), return_std=True)
                elif isinstance(regression_models[i], LinearRegression):
                    new_x_poly = poly_transformer.fit_transform(new_x.reshape(1,-1))
                    mean_new[i,...] = regression_models[i].predict(new_x_poly)
                    std_new[i,...] = fictive_noise_level
                else:
                    mean_new[i,...] = regression_models[i].predict(new_x.reshape(1,-1))
                    std_new = fictive_noise_level
                estimated_observation_new[i,...] = mean_new[i,]+rng.normal(0, std_new[i], size=1)
            
            estimated_observation_new_aggregated = aggregation_function(estimated_observation_new, **kwargs)

            
            #Store the new estimated observations
            estimated_sample_x = np.vstack([estimated_sample_x, new_x])
            for regression_model in regression_models:
                if isinstance(regression_model, LinearRegression):
                    estimated_sample_x_poly = poly_transformer.fit_transform(estimated_sample_x)
            estimated_observation_y = np.hstack([estimated_observation_y, estimated_observation_new])
            estimated_observation_y_aggregated = np.hstack([estimated_observation_y_aggregated, 
                                                            estimated_observation_new_aggregated])
            batch_sample[j,...] = new_x

        #Active learning loop ends here
        #########################################################################################################
        
        # Updated pool with batch data
        sample_x = np.vstack([sample_x, batch_sample])

        if single_update:
            #transform results to a pandas DataFrame
            if calculate_test_metrics:
                results = np.hstack([n_observations[0], scores_train[0].T, scores_test[0].T, max_value[0].T]).reshape(1,-1)
                print(results)
                results = pd.DataFrame(results, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
            else:
                results = np.hstack([n_observations[0], scores_train[0].T, max_value[0].T])
                results = pd.DataFrame(results, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
            
            return sample_x, results


        if isinstance(regression_model, LinearRegression):
            sample_x_poly = poly_transformer.fit_transform(sample_x)

        observation_new = np.zeros((n_models, len(batch_sample)))
        for i in range(n_models):
            observation_new[i,...] = models[i].evaluate(batch_sample, noise=noise[i])

        observation_y = np.hstack([observation_y, observation_new])

        observation_new_aggregated = aggregation_function(observation_new, **kwargs)
        observation_y_aggregated = np.hstack([observation_y_aggregated, observation_new_aggregated])

        # Fit new model with updated real training set
        mean = np.zeros((n_models, len(pool)))
        std = np.zeros((n_models, len(pool)))

        mean_train = np.zeros((n_models, len(sample_x)))
        std_train = np.zeros((n_models, len(sample_x)))

        #Calculate metrics
        if verbose: print('Active learning step: {}'.format(a))
        for i in range(n_models):
            if isinstance(regression_models[i], LinearRegression):
                regression_models[i].fit(sample_x_poly, observation_y[i])
            else:
                regression_models[i].fit(sample_x, observation_y[i])

            if calculate_test_metrics:
                if isinstance(regression_models[i], GPR):
                    mean[i,...], std[i,...] = regression_models[i].predict(pool,return_std=True)
                elif isinstance(regression_models[i], LinearRegression):
                    mean[i,...] = regression_models[i].predict(pool_poly)
                else:
                    mean[i,...] = regression_models[i].predict(pool)

            if isinstance(regression_models[i], GPR):
                mean_train[i,...], std_train[i,...] = regression_models[i].predict(sample_x,return_std=True)
            elif isinstance(regression_models[i], LinearRegression):
                mean_train[i,...] = regression_models[i].predict(sample_x_poly)
            else:
                mean_train[i,...] = regression_models[i].predict(sample_x)
            
            if verbose: 
                print('Model {}'.format(i))
                if isinstance(regression_models[i], Pipeline):
                    if isinstance(regression_models[i].named_steps['model'], GPR):
                        print(regression_models[i].named_steps['model'].kernel_)
                    else:
                        print(regression_models[i])
                else:
                    if isinstance(regression_models[i], GPR):
                        print(regression_models[i].kernel_)
                    else:
                        print(regression_models[i])

        if calculate_test_metrics:
            mean_aggregated = aggregation_function(mean, **kwargs)
            scores_test[a+1,0] = mean_squared_error(y_true_aggregated, mean_aggregated)
            scores_test[a+1,1] = mean_absolute_error(y_true_aggregated, mean_aggregated)
            scores_test[a+1,2] = max_error(y_true_aggregated, mean_aggregated)

        mean_train_aggregated = aggregation_function(mean_train, **kwargs)
        #print('New iteration')
        #print(sample_x.shape)
        #print(observation_y_aggregated.shape)
        #print(mean_train_aggregated.shape)
        scores_train[a+1,0] = mean_squared_error(observation_y_aggregated, mean_train_aggregated)
        scores_train[a+1,1] = mean_absolute_error(observation_y_aggregated, mean_train_aggregated)
        scores_train[a+1,2] = max_error(observation_y_aggregated, mean_train_aggregated)
        max_value[a+1,0] = np.max(observation_y_aggregated)
        
    #transform results to a pandas DataFrame
    if calculate_test_metrics:
        results = np.vstack([n_observations, scores_train.T, scores_test.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
    else:
        results = np.vstack([n_observations, scores_train.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
    
    return sample_x, results