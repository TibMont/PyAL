# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import copy
import sys
import os
import warnings

import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import scale
from scipy.stats import norm


from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from PyAL.optimize_step import step_continous_multi
import PyAL.utils as utils

import logging

logger = logging.getLogger('basic_logger')

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')


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
           custom_acfn_input = {},
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

    logger.info('Setting up Active Learning')

    #Check consistency of parameters
    if isinstance(regression_models, list):
        if len(regression_models) != len(models):
            raise Exception('Inconsistent number of data and regression models: {}, {}'.format(len(models), len(regression_model)))
    else:
        regression_models = [copy.deepcopy(regression_models) for i in range(len(models))]
    
    reg_models_pure = []
    for regression_model in regression_models:
        reg_models_pure.append(utils.check_model(regression_model, acquisition_function))

    #Set random number generator
    if isinstance(random_state, int) or random_state==None:
        rng = np.random.RandomState(seed=random_state)

    #Set polynomial feature transformer
    poly_transformer = PolynomialFeatures(degree=poly_degree)

    dimensions = models[0].n_features
    n_models = len(models)

    #Check if noise is int or float, noise will be applied to every model individually
    if isinstance(noise, int) or isinstance(noise, float):
        noise_old = noise
        noise = [noise for _ in range(n_models)]
        if verbose:
            print('Noise converted: ')
            print('from {} to {}'.format(noise_old, noise))
        logger.info('Noise converted: ')
        logger.info('from {} to {}'.format(noise_old, noise))

    #Generate a pool of sample data points for testing
    if calculate_test_metrics:
        logger.info('Test metrics will be calculated.')
        if not isinstance(pool, np.ndarray):
            pool = utils.generate_pool(dimensions, lim)
    
        #Number of data points in pool
        n_data = len(pool)
        y_true = np.zeros((n_models, n_data))
        for i in range(n_models):
            model = models[i]
            y_true[i, ...] = model.evaluate(pool, noise = noise[i])

        y_true_aggregated = aggregation_function(y_true, **kwargs)
    
    else:
        logger.info('Test metrics will not be calculated.')

    sampler = LHS(d=dimensions, seed=random_state)
    #Generate initial data
    logger.info('Initialization method: {}'.format(initialization))
    if initialization == 'random':
        if isinstance(initial_samples, int):
            sample_x_unscaled = sampler.random(initial_samples)
            sample_x = scale(sample_x_unscaled, *lim)
        else:
            raise Exception('initial_samples must be an integer for initialization method random')
    elif initialization == 'GSx':
        if isinstance(initial_samples, int):
            sample_x, _, _ = run_continuous_batch_learning_multi(models, 
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
            random_state=rng,
            initialization='random',
            pso_options=pso_options,
            poly_degree = poly_degree,
            custom_acfn_input = custom_acfn_input,
            calculate_test_metrics=False,
            verbose=False,
            **kwargs
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
    logger.info('Initialization finished.')

    observation_y = np.zeros((n_models, len(sample_x)))
    for i in range(n_models):
        observation_y[i, ...] = models[i].evaluate(sample_x, noise=noise[i])

    observation_y_aggregated = aggregation_function(observation_y, **kwargs)

    #To save the metrics
    scores_train = np.zeros((active_learning_steps+1,3))
    max_value = np.zeros((active_learning_steps+1,1))
    n_observations = np.linspace(initial_samples, initial_samples+(active_learning_steps)*batch_size,
                            active_learning_steps+1)
    
    mean_train = np.zeros((n_models, len(sample_x)))
    std_train = np.zeros((n_models, len(sample_x)))
    for i in range(n_models):
        regression_models[i] = utils.fit_model(sample_x, observation_y[i], 
                                               regression_models[i], poly_transformer)
        #Initial model predictions for training set
        mean_train[i,...], std_train[i,...] = utils.make_prediction(sample_x, regression_models[i], 
                                                                    poly_transformer)
    
    mean_train_aggregated = aggregation_function(mean_train, **kwargs)
    scores_train[0,...] = utils.calculate_errors(observation_y_aggregated.flatten(), mean_train_aggregated.flatten())
    max_value[0,0] = np.max(observation_y_aggregated)
    
    if calculate_test_metrics:
        scores_test = np.zeros((active_learning_steps+1,3))

        mean = np.zeros((n_models, len(pool)))
        std = np.zeros((n_models, len(pool)))

        for i in range(n_models):
            mean[i,...], std[i,...] = utils.make_prediction(pool, regression_models[i]
                                                            poly_transformer)
            
        #Save scores
        mean_aggregated = aggregation_function(mean, **kwargs)
        scores_test[0,...] = utils.calculate_errors(y_true_aggregated.flatten(), mean_aggregated.flatten())


    #Active Learning loop starts here
    ###############################################################
    logger.info('Start Active Learning')
    logger.info('Optimization method: {}'.format(opt_method))
    logger.info('Acquisition function: {}'.format(acquisition_function))

    #Start active learning
    for a in range(active_learning_steps):
        logger.info('Step {}'.format(a+1))

        #Save batch results separately, since observation is only estimated
        batch_sample = np.zeros((batch_size, dimensions))
        estimated_observation_y = observation_y.copy()
        estimated_sample_x = sample_x.copy()
        estimated_observation_y_aggregated = observation_y_aggregated.copy()
        
        for j in range(batch_size):
            #For the first sample in a batch we can use the model with which we evaluated the scores
            if j != 0:
                #Fit models
                for i in range(n_models):
                    regression_models[i] = utils.fit_model(estimated_sample_x, estimated_observation_y[i],
                                                           regression_models[i], poly_transformer)


            for i in range(n_models):        
                if isinstance(reg_models_pure[i], LinearRegression):
                    poly_x = poly_transformer
                else:
                    poly_x = None

            new_x, _ = step_continous_multi(
                    acquisition_function, 
                    opt_method, 
                    regression_models, 
                    aggregation_function,
                    estimated_observation_y,
                    estimated_observation_y_aggregated, 
                    estimated_sample_x,
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
                    **kwargs)

            #Assume estimated predictions

            mean_new = np.zeros((n_models, len(new_x.reshape(1,-1))))
            std_new = np.zeros((n_models, len(new_x.reshape(1,-1))))
            estimated_observation_new = np.zeros((n_models, len(new_x.reshape(1,-1))))

            for i in range(n_models):
                mean_new[i,...], std_new[i,...] = utils.make_prediction(new_x, regression_models[i],
                                                                        poly_transformer, fictive_noise_level)
                estimated_observation_new[i,...] = mean_new[i,]+rng.normal(0, std_new[i], size=1)
            
            estimated_observation_new_aggregated = aggregation_function(estimated_observation_new, **kwargs)

            #Store the new estimated observations
            estimated_sample_x = np.vstack([estimated_sample_x, new_x])
            estimated_observation_y = np.hstack([estimated_observation_y, estimated_observation_new])
            estimated_observation_y_aggregated = np.hstack([estimated_observation_y_aggregated, 
                                                            estimated_observation_new_aggregated])
            batch_sample[j,...] = new_x

        #Active learning loop ends here
        #########################################################################################################

        # Updated pool with batch data
        sample_x = np.vstack([sample_x, batch_sample])

        observation_new = np.zeros((n_models, len(batch_sample)))
        for i in range(n_models):
            observation_new[i,...] = models[i].evaluate(batch_sample, noise=noise[i])

        observation_y = np.hstack([observation_y, observation_new])

        observation_new_aggregated = aggregation_function(observation_new, **kwargs)
        observation_y_aggregated = np.hstack([observation_y_aggregated, observation_new_aggregated])

        # Fit new model with updated real training set
        mean_train = np.zeros((n_models, len(sample_x)))
        std_train = np.zeros((n_models, len(sample_x)))

        #Calculate metrics
        if verbose: print('Active learning step: {}'.format(a))

        for i in range(n_models):
            regression_models[i] = utils.fit_model(sample_x, observation_y[i], 
                                                   regression_models[i], poly_transformer)
            
            mean_train[i,...], std_train[i,...] = utils.make_prediction(sample_x, regression_models[i],
                                                                        poly_transformer)

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

        mean_train_aggregated = aggregation_function(mean_train, **kwargs)
        scores_train[a+1,...] = utils.calculate_errors(observation_y_aggregated.flatten(), mean_train_aggregated.flatten())
        max_value[a+1,0] = np.max(observation_y_aggregated)

        if calculate_test_metrics:
            mean = np.zeros((n_models, len(pool)))
            std = np.zeros((n_models, len(pool)))

            mean[i,...], std[i,...] = utils.make_prediction(pool, regression_models[i], poly_transformer)
            mean_aggregated = aggregation_function(mean, **kwargs)
            scores_test[a+1,...] = utils.calculate_errors(y_true_aggregated.flatten(), mean_aggregated.flatten())

        if single_update:
            #transform results to a pandas DataFrame
            if calculate_test_metrics:
                results = np.hstack([n_observations[0], scores_train[0].T, scores_test[0].T, max_value[0].T]).reshape(1,-1)
                results = pd.DataFrame(results, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
            else:
                results = np.hstack([n_observations[0], scores_train[0].T, max_value[0].T]).reshape(1,-1)
                results = pd.DataFrame(results, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
            
            return sample_x, observation_y, results
    
    logger.info('Finished Active Learning')

    #transform results to a pandas DataFrame
    if calculate_test_metrics:
        results = np.vstack([n_observations, scores_train.T, scores_test.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
    else:
        results = np.vstack([n_observations, scores_train.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
    
    return sample_x, observation_y, results