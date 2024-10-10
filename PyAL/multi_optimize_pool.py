# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import numpy as np
import pandas as pd

from PyAL.acfn_discrete import EI, POI, UCB, IDEAL, GSx, GSy, iGS, QBC, SGSx, UIDAL

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

from PyAL.optimize_step import step_discrete

import sys
import os
import warnings

import logging

logger = logging.getLogger('basic_logger')

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')

    
def QBC_multi(x, models, aggregation_function, poly_x, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)

    n_models = len(models[0])
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
            mean_qbc_individual[i, j, ...] += m

    mean_qbc = np.zeros((len(models[0]), n_pool))

    for i in range(mean_qbc_individual.shape[1]):
        mean_qbc[i] = aggregation_function(mean_qbc_individual[:,i,:], **kwargs)

    result = np.zeros(n_pool)
    for i in range(n_pool):
        result[i] = np.sum( mean_qbc[:,i] - np.mean(mean_qbc[:,i]) )

    return result
    

def run_batch_learning_multi(models,
           aggregation_function, 
           regression_models,
           acquisition_function = 'ideal',
           pool = None, 
           batch_size = 1,
           noise=0.1,
           initial_samples=2, 
           active_learning_steps=10,
           lim=[-1,1],
           alpha=0,
           random_state=None,
           return_samples=False,
           initialization='random',
           test_set = None,
           poly_degree = 3,
           fictive_noise_level = 0,
           calculate_test_metrics = True,
           single_update=False,
           verbose=False,
           **kwargs
           ):
    '''
    Perform batch-mode Active Learning for a discrete parameter space. The algorithm picks automatically initial data points from a given
    pool of discrete possible data points for evaluation using either a random approach or the model-free GSx method.

    Parameters
    ----------
    model : Model class
        Model to generate true data.
    regression_model: sklearn model, str, optional
        scikit-learn regression model.
    acquissition_function : str, optional
        Acquisition function. The default is 'ideal'.
    pool : nd_array, None, optional
        Array containing pool of data. All data is sampled from that pool. Data which is not sampled so far is used for calculating 
        test metrics when test_set is None.
        For 'None' a grid with 100 points in each dimension is created. The default is 'None'.
    batch_size : int, optional
        Batch size for Active Learning. The model is updated only with true data when a batch is completed. The default value is 1.
    noise : float, optional
        Noise in observation. The default is 0.1.
    initial_samples : int, optional
        Number of initial samples to draw from the pool. The default is 2.
    active_learning_steps : int, optional
        Number of active learning steps to perform. The default is 10.
    lim : list, optional
        Boundaries for model evaluation. Only used when pool=None. The default is [-1,1].
    alpha : float
        Hyperparameter for acquisition function. The default is 0.
    random_state: int, optional
        Set random state. The default is None.
    return_samples: bool, optional
        Whether to return the samples drawn so far. The default is False.
    initialization : str, optional
        Initialization method for generating initial data. Choose from 'random' and 'GSx'. 'random' uses Latin Hypercube sampling 
        to generate the initial dat points. 'GSx' draws randomly the first data point and then uses the model-free GSx method to sample
        the other initial data points. The default value is 'random'.
    test_set : nd_array, optional
        Array containing explicit data points for testing. If it is 'None' all unsampled data points in the pool will be used for testing.
        The default value is 'None'.
    calculate_test_metrics : bool, optional
        Can be used for testing Active Learning algorithms for known models. Test metrics are calculated automatically. If 'False' no test metrics are 
        calculated and the AL runs in deployement mode.

    Returns
    -------
    results : pd_DataFrame
        Pandas DataFrame containing the columns: 'm', 'mean_mse_test', 'max_observation'. Data is stored for each active learning step, 
        where 'm' gives the number of data used for training, 'mean_mse_test' the MSE of the test set and 'max_observation' the maximum 
        value observed so far.
    sample_x : nd_array, optional
        Array that contains the sampled data points (in terms of their features). Only returned when return_samples is True. 

    '''
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

    poly_transformer = PolynomialFeatures(degree=poly_degree)

    dimensions = models[0].n_features
    n_models = len(models)

    if isinstance(noise, int) or isinstance(noise, float):
        noise_old = noise
        noise = [noise for _ in range(n_models)]
        if verbose:
            print('Noise converted: ')
            print('from {} to {}'.format(noise_old, noise))
        logger.info('Noise converted: ')
        logger.info('from {} to {}'.format(noise_old, noise))

    #Generate a pool of sample data points
    if not isinstance(pool, np.ndarray):
        x = []
        for i in range(dimensions):
            x.append(np.linspace(*lim, 20))

        pool = np.meshgrid(*x)
        pool = np.array(pool).T
        pool = pool.reshape(len(x[0]**dimensions), dimensions)

    pool_poly = poly_transformer.fit_transform(pool)

    #Number of data points in pool

    n_data = len(pool)

    if calculate_test_metrics:
        if isinstance(test_set, np.ndarray):
            n_data_test = len(test_set)
            y_true_test = np.zeros((n_models, n_data_test))
            test_set_poly = poly_transformer.fit_transform(test_set)
            for i in range(n_models):
                model = models[i]
                y_true_test[i, ...] = model.evaluate(test_set, noise = noise[i])
            extra_test_set = True
            y_true_test_aggregated = aggregation_function(y_true_test, **kwargs)
        else:
            n_data_test = len(pool)
            y_true = np.zeros((n_models, n_data_test))
            for i in range(n_models):
                model = models[i]
                y_true[i, ...] = model.evaluate(pool, noise = noise[i])
            extra_test_set = False
            y_true_aggregated = aggregation_function(y_true, **kwargs)
    
    #Randomly pick data points in pool for initial observations
    if initialization == 'random':
        rand_num = rng.randint(0,n_data, size=initial_samples)
    elif initialization == 'GSx':
        initial_data, _ = run_batch_learning_multi(models, 
           aggregation_function,
           regression_models,
           acquisition_function = 'GSx',
           pool=pool, 
           batch_size=1,
           noise=noise,
           initial_samples=1, 
           active_learning_steps=initial_samples-1,
           lim=lim,
           alpha=alpha,
           random_state=random_state,
           return_samples=return_samples,
           initialization='random',
           test_set=test_set,
           poly_degree=poly_degree,
           **kwargs
           )

        idx = []
        for i in range(len(initial_data)):
            index = np.where(np.sum(pool, axis=1)==np.sum(initial_data[i]))[0][0]
            idx.append(index)
        
        rand_num = np.array(idx)
    else:
        raise Exception('Initializiation method not implemented')

    sample_x = pool[rand_num]
    sample_x_poly = poly_transformer.fit_transform(sample_x)
    observation_y = np.zeros((n_models, len(sample_x)))
    for i in range(n_models):
        observation_y[i, ...] = models[i].evaluate(sample_x, noise=noise[i])

    observation_y_aggregated = aggregation_function(observation_y, **kwargs)

    data_indices = rand_num.copy()

    #To save the metrics
    scores_train = np.zeros((active_learning_steps+1,3))
    max_value = np.zeros((active_learning_steps+1,1))
    n_observations = np.linspace(initial_samples, initial_samples+(active_learning_steps)*batch_size,
                                 active_learning_steps+1)
    
    scores_train_individual = []
    max_value_individual = []
    for i in range(n_models):
        scores_train_individual.append(scores_train.copy())
        max_value_individual.append(max_value.copy())


    if calculate_test_metrics:
        scores_test = np.zeros((active_learning_steps+1,3))

        scores_test_individual = []
        for i in range(n_models):
            scores_test_individual.append(scores_train.copy())


    for i in range(n_models):
        if isinstance(regression_models[i], LinearRegression):
            regression_models[i].fit(sample_x_poly, observation_y[i])
        else:
            regression_models[i].fit(sample_x, observation_y[i])

    #Fit initial model
    mean = np.zeros((n_models, len(pool)))
    std = np.zeros((n_models, len(pool)))

    if calculate_test_metrics:
        if extra_test_set == False:
            mask = np.ones(y_true.size, dtype=bool)
            mask[data_indices] = False
            mask_indices = np.where(mask==True)[0]
            test_set = pool[mask_indices]
            test_set_poly = pool_poly[mask_indices]
            y_true_test = y_true[mask_indices]
            y_true_test_aggregated = y_true_aggregated[mask_indices]
            
        mean_test = np.zeros((n_models, len(test_set)))
        std_test = np.zeros((n_models, len(test_set)))

    for i in range(n_models):
        #Calculate metrics
        if calculate_test_metrics:
            if isinstance(regression_models[i], GPR):   
                mean_test[i,...], std_test[i,...] = regression_models[i].predict(test_set,return_std=True)
            elif isinstance(regression_models[i], LinearRegression):
                mean_test[i,...] = regression_models[i].predict(test_set_poly)
            else:
                mean_test[i,...] = regression_models[i].predict(test_set)
        

        if isinstance(regression_models[i], GPR):   
            mean[i,...], std[i,...] = regression_models[i].predict(pool,return_std=True)
        elif isinstance(regression_models[i], LinearRegression):
            mean[i,...] = regression_models[i].predict(pool_poly)
        else:
            mean[i,...] = regression_models[i].predict(pool)

    if calculate_test_metrics:
        mean_test_aggregated = aggregation_function(mean_test, **kwargs)
        scores_test[0,0] = mean_squared_error(y_true_test_aggregated, mean_test_aggregated)
        scores_test[0,1] = mean_absolute_error(y_true_test_aggregated, mean_test_aggregated)
        scores_test[0,2] = max_error(y_true_test_aggregated, mean_test_aggregated)

        for i in range(n_models):
            scores_test_individual[i][0,0] = mean_squared_error(y_true_test[i], mean_test[i])
            scores_test_individual[i][0,1] = mean_absolute_error(y_true_test[i], mean_test[i])
            scores_test_individual[i][0,2] = max_error(y_true_test[i], mean_test[i])

    #Save scores
    mean_train_aggregated = aggregation_function(mean, **kwargs)
    std_train_aggregated = aggregation_function(std, uncert=True, **kwargs)
    scores_train[0,0] = mean_squared_error(observation_y_aggregated, mean_train_aggregated[data_indices])
    scores_train[0,1] = mean_absolute_error(observation_y_aggregated, mean_train_aggregated[data_indices])
    scores_train[0,2] = max_error(observation_y_aggregated, mean_train_aggregated[data_indices])
    max_value[0,0] = np.max(observation_y_aggregated)

    for i in range(n_models):
        scores_train_individual[i][0,0] = mean_squared_error(observation_y[i], mean[i][data_indices])
        scores_train_individual[i][0,1] = mean_absolute_error(observation_y[i], mean[i][data_indices])
        scores_train_individual[i][0,2] = max_error(observation_y[i], mean[i][data_indices])
        max_value_individual[i][0,0] = np.max(observation_y[i])

    
    #Start active learning
    for a in range(active_learning_steps):

        batch_indices = np.zeros(batch_size, dtype=np.int32)
        estimated_observation_y = observation_y.copy()
        estimated_sample_x = sample_x.copy()
        estimated_observation_y_aggregated = observation_y_aggregated.copy()
        for regression_model in regression_models:
            if isinstance(regression_model, LinearRegression):
                estimated_sample_x_poly = sample_x_poly.copy()
        
        for j in range(batch_size):

            if j != 0:
                mean = np.zeros((n_models, len(pool)))
                std = np.zeros((n_models, len(pool)))
                for i in range(n_models):
                    if isinstance(regression_models[i], LinearRegression):
                        regression_models[i].fit(estimated_sample_x_poly, estimated_observation_y[i])
                    else:
                        regression_models[i].fit(estimated_sample_x, estimated_observation_y[i])
                    if isinstance(regression_models[i], GPR):
                        mean[i,...], std[i,...] = regression_models[i].predict(pool,return_std=True)
                    elif isinstance(regression_models[i], LinearRegression):
                        mean[i,...] = regression_models[i].predict(pool_poly)
                    else:
                        mean[i,...] = regression_models[i].predict(pool)

                mean_train_aggregated = aggregation_function(mean, **kwargs)
                std_train_aggregated = aggregation_function(std, uncert=True, **kwargs)

            mask = np.ones(pool.shape[0], dtype=bool)
            mask[data_indices] = False
            mask_indices = np.where(mask==True)[0]

            mask_z = np.ones(pool.shape[0], dtype=np.int32)
            mask_z[data_indices] = 0

            #Optional plotting for debugging
            #plt.plot(sample_x, observation_y, 'o')
            #plt.plot(pool, y_true, 'k')
            #plt.plot(pool, mean)
            #plt.show()

            if acquisition_function == 'qbc':
                S_models = []
                for i in range(n_models):
                    alpha_models = []
                    for _ in range(alpha):
                        train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                        if isinstance(regression_models[i], LinearRegression):
                            poly_x = poly_transformer
                            regression_models[i].fit(estimated_sample_x_poly[train_index], estimated_observation_y[i][train_index])
                        else:
                            poly_x = None
                            regression_models[i].fit(estimated_sample_x[train_index], estimated_observation_y[i][train_index])
                        alpha_models.append(copy.deepcopy(regression_model))
                    S_models.append(alpha_models)
                imp = QBC_multi(pool, S_models, aggregation_function, poly_x, **kwargs)
                imp = imp*mask_z
                index = np.where(imp==np.max(imp))[0]
            else:
                acquisition = step_discrete(
                    acquisition_function, 
                    estimated_observation_y, 
                    alpha, mean_train_aggregated, std_train_aggregated,
                    n_data, data_indices, pool,
                    rng,
                )

            acquisition_masked = acquisition*mask_z
            index = np.where(acquisition_masked[mask]==np.max(acquisition_masked[mask]))[0]
            
            if len(index) > 1:
                ind = rng.randint(0,len(index),1)
                index = np.array(index[ind])
            data_indices = np.concatenate([data_indices, index])
                
            estimated_x_max = pool[index]
            estimated_sample_x = np.vstack([estimated_sample_x, estimated_x_max])
            for regression_model in regression_models:
                if isinstance(regression_model, LinearRegression):
                    estimated_sample_x_poly = poly_transformer.fit_transform(estimated_sample_x)

            #Assume estimated predictions
                
            mean_new = np.zeros((n_models, len(estimated_x_max.reshape(1,-1))))
            std_new = np.zeros((n_models, len(estimated_x_max.reshape(1,-1))))
            estimated_observation_new = np.zeros((n_models, len(estimated_x_max.reshape(1,-1))))

            for i in range(n_models):
                if isinstance(regression_model, GPR):
                    mean_new[i,...], std_new[i,...] = regression_models[i].predict(pool[index].reshape(1,-1), return_std=True)
                elif isinstance(regression_models[i], LinearRegression):
                    new_x_poly = poly_transformer.fit_transform(pool[index].reshape(1,-1))
                    mean_new[i,...] = regression_models[i].predict(new_x_poly)
                    std_new[i,...] = fictive_noise_level
                else:
                    mean_new[i,...] = regression_models[i].predict(pool[index].reshape(1,-1))
                    std_new[i,...] = fictive_noise_level

                estimated_observation_new[i,...] = mean_new[i]+rng.normal(0, std_new[i], size=1)

            estimated_observation_new_aggregated = aggregation_function(estimated_observation_new, **kwargs)

            estimated_observation_y = np.hstack([estimated_observation_y, estimated_observation_new])
            estimated_observation_y_aggregated = np.hstack([estimated_observation_y_aggregated, estimated_observation_new_aggregated])
            batch_indices[j] = index

        #Active learning loop ends here
        ########################################################################################################

        # Updated pool with batch data
        x_max = pool[batch_indices]
        sample_x = np.vstack([sample_x, x_max])

        if single_update:
            #transform results to a pandas DataFrame
            if calculate_test_metrics:
                results = np.hstack([n_observations[0], scores_train[0].T, scores_test[0].T, max_value[0].T]).reshape(1,-1)
                results = pd.DataFrame(results, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
            else:
                results = np.hstack([n_observations[0], scores_train[0].T, max_value[0].T]).reshape(1,-1)
                results = pd.DataFrame(results, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
            
            return sample_x, results
        
        for regression_model in regression_models:
            if isinstance(regression_model, LinearRegression):
                sample_x_poly = poly_transformer.fit_transform(sample_x)

        observation_new = np.zeros((n_models, len(x_max)))
        for i in range(n_models):
            observation_new[i,...] = models[i].evaluate(x_max, noise=noise[i])
        
        observation_y = np.hstack([observation_y, observation_new])

        observation_new_aggregated = aggregation_function(observation_new, **kwargs)
        observation_y_aggregated = np.hstack([observation_y_aggregated, observation_new_aggregated])

        #Fit new model with updated real training set
        mean = np.zeros((n_models, len(pool)))
        std = np.zeros((n_models, len(pool)))

        if calculate_test_metrics:
            if extra_test_set == False:
                mask = np.ones(y_true.size, dtype=bool)
                mask[data_indices] = False
                mask_indices = np.where(mask==True)[0]
                test_set = pool[mask_indices]
                test_set_poly = pool_poly[mask_indices]
                y_true_test = y_true[mask_indices]
                y_true_test_aggregated = y_true_aggregated[mask_indices]

            mean_test = np.zeros((n_models, len(test_set)))
            std_test = np.zeros((n_models, len(test_set)))

        for i in range(n_models):
            regression_model = regression_models[i]
            if isinstance(regression_model, LinearRegression):
                regression_model.fit(sample_x_poly, observation_y[i])
            else:
                regression_model.fit(sample_x, observation_y[i])

            ###Here###
            if calculate_test_metrics:
                if isinstance(regression_models[i], GPR):
                    mean_test[i,...], std_test[i,...] = regression_models[i].predict(test_set,return_std=True)
                elif isinstance(regression_models[i], LinearRegression):
                    mean_test[i,...] = regression_models[i].predict(test_set_poly)
                else:
                    mean_test[i,...] = regression_models[i].predict(test_set)

            if isinstance(regression_models[i], GPR):
                mean[i,...], std[i,...] = regression_models[i].predict(pool,return_std=True)
            elif isinstance(regression_models[i], LinearRegression):
                mean[i,...] = regression_models[i].predict(pool_poly)
            else:
                mean[i,...] = regression_models[i].predict(pool)
        
        #############
        if calculate_test_metrics:
            mean_test_aggregated = aggregation_function(mean_test, **kwargs)
            scores_test[a+1,0] = mean_squared_error(y_true_test_aggregated, mean_test_aggregated)
            scores_test[a+1,1] = mean_absolute_error(y_true_test_aggregated, mean_test_aggregated)
            scores_test[a+1,2] = max_error(y_true_test_aggregated, mean_test_aggregated)

            for i in range(n_models):
                scores_test_individual[i][a+1,0] = mean_squared_error(y_true_test[i], mean_test[i])
                scores_test_individual[i][a+1,1] = mean_absolute_error(y_true_test[i], mean_test[i])
                scores_test_individual[i][a+1,2] = max_error(y_true_test[i], mean_test[i])

        #Save scores
        mean_train_aggregated = aggregation_function(mean, **kwargs)
        std_train_aggregated = aggregation_function(std, uncert=True, **kwargs)
        scores_train[a+1,0] = mean_squared_error(observation_y_aggregated, mean_train_aggregated[data_indices])
        scores_train[a+1,1] = mean_absolute_error(observation_y_aggregated, mean_train_aggregated[data_indices])
        scores_train[a+1,2] = max_error(observation_y_aggregated, mean_train_aggregated[data_indices])
        max_value[a+1,0] = np.max(observation_y_aggregated)

        for i in range(n_models):
            scores_train_individual[i][a+1,0] = mean_squared_error(observation_y[i], mean[i][data_indices])
            scores_train_individual[i][a+1,1] = mean_absolute_error(observation_y[i], mean[i][data_indices])
            scores_train_individual[i][a+1,2] = max_error(observation_y[i], mean[i][data_indices])
            max_value_individual[i][a+1,0] = np.max(observation_y[i])
        ############            
        
    #transform results to a pandas DataFrame
    if calculate_test_metrics:
        result_dict = {}
        results = np.vstack([n_observations, scores_train.T, scores_test.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
        result_dict['aggregated'] = results.copy()
        
        for i in range(n_models):
            results = np.vstack([n_observations, scores_train_individual[i].T, scores_test_individual[i].T, max_value_individual[i].T])
            results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
            result_dict['model_'+str(i)] = results.copy()
    
    else:
        result_dict = {}
        results = np.vstack([n_observations, scores_train.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
        result_dict['aggregated'] = results.copy()
        
        for i in range(n_models):
            results = np.vstack([n_observations, scores_train_individual[i].T, max_value_individual[i].T])
            results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
            result_dict['model_'+str(i)] = results.copy()

    return sample_x, result_dict