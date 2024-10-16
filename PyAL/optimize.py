# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import scale
from scipy.stats import norm

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from PyAL.acfn_discrete import EI, POI, UCB, IDEAL, GSx, GSy, iGS, QBC, SGSx, UIDAL
from PyAL.optimize_step import step_continuous, step_discrete
from PyAL.utils import check_model

import copy
from pyswarms.single.global_best import GlobalBestPSO

import sys
import os
import warnings

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')

#TODO: Enable mode without calculation of test metrics, which can be directly used for real experiments.

def run_continuous_batch_learning(model, 
           regression_model,
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
           custom_acfn_input = None,
           ):
    '''
    Perform batch-mode Active Learning in continuous parameter space for a given model generating the true data and a give regression model. 
    The algorithm generates initial data automatically using either a random approach or using the model-free GSx approach. 

    Parameters
    ----------
    model : Model class
        Model to generate true data.
    regression_model: sklearn model, str, optional
        scikit-learn regression model.
    acquissition_function : str, optional
        Acquisition function. Can also be a Callable. Then in custom_acfn_input it needs to be specified
        which input the custom function takes. The Callable should only take the input in the described order.
        The default is 'ideal'. 
    opt_method : str, optional
        The method to optimize the acquisition function. Choose from 'lbfgs' and 'PSO'. The default is 'PSO'.
    pool : nd_array, None, optional
        Array containing pool of data that is used for testing. For 'None' a grid with 100 points in each dimension is created. The default is 'None'.
    batch_size : int, optional
        Batch size for Active Learning. The model is updated only with true data when a batch is completed. The default value is 1.
    noise : float, optional
        Noise in observation. The default is 0.1.
    initial_samples : int, optional
        Number of initial samples to generate for the Active Learning. The default is 2.
    active_learning_steps : int, optional
        Number of active learning steps to perform. The default is 10.
    lim : list, optional
        Boundaries for model evaluation. Only used when pool=None. The default is [-1,1].
    alpha : float, optional
        Hyperparameter for acquisition function. The default is 0.
    n_jobs : int, optional
        Number of cores to use for parallel evaluation of PSO. Currently not used, PSO runs only in serial mode. The default is 1.
    random_state: int, optional
        Set random state. The default is None.
    initialization : str, optional
        Initialization method for generating initial data. Choose from 'random' and 'GSx'. 'random' uses Latin Hypercube sampling 
        to generate the initial dat points. 'GSx' draws randomly the first data point and then uses the model-free GSx method to sample
        the other initial data points. The default value is 'random'.
    pso_options : dict, optional
        Dictionary with parameters for the Particle Swarm Optimization. Only used when opt_method is 'PSO'. For 'None' default values are used. 
        ['c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p':dimensions*10, 'i':200]. c1 and c2 are swarm parmeters, w is the inertia, p gives the number
        of particles, i is the number of iterations. The default is 'None'.
    fictive_noise_level : float, optional
        Noise level for non-GPR models used for assuming predictions as true values to enable batch-wise learning.
    calculate_test_metrics : bool, optional
        Can be used for testing Active Learning algorithms for known models. Test metrics are calculated automatically. If 'False' no test metrics are 
        calculated and the AL runs in deployement mode.
    custom_args_input : list, optional
        List of arguments which a custom acquisition function takes. It is important that the callable takes the arguments
        in the described order. The arguments must also be named as described here:

        x0: First argument, does not need to appear in this list.
        x: Values of the features of all evaluated samples
        y: Objective values of all evaluated samples
        max_y: Maximum observed value of all evaluated samples
        regression_model: sklearn model used for the fitting
        lim: boundary 
        alpha: hyperparameter
        poly_x: sklearn poly_transformer. Must be given if a linear regression model should be used with the acquisition function

        The callable must return the value of the acquisition function as a float or as a numpy_array of floats.

    Returns
    -------
    sample_x : nd_array
        Array that contains the sampled data points (in terms of their features). 
    results : pd_DataFrame, optional
        Pandas DataFrame containing the columns: 'm', 'mean_mse_test', 'max_observation'. Data is stored for each active learning step, 
        where 'm' gives the number of data used for training, 'mean_mse_test' the MSE of the test set and 'max_observation' the maximum 
        value observed so far. Only returned when calculate_test_metrics is 'True'.


    '''

    reg_model_pure = check_model(regression_model, acquisition_function)
    
    #Set random number generator
    if isinstance(random_state, int) or random_state==None:
        rng = np.random.RandomState(seed=random_state)

    poly_transformer = PolynomialFeatures(degree=poly_degree)

    #Generate a pool of sample data points for testing
    dimensions = model.n_features
    if calculate_test_metrics == True:
        if not isinstance(pool, np.ndarray):
            x = []
            for i in range(dimensions):
                x.append(np.linspace(*lim, 10))

            pool = np.meshgrid(*x)
            pool = np.array(pool).T
            pool = pool.reshape(len(x[0]**dimensions), dimensions)

        pool_poly = poly_transformer.fit_transform(pool)

    if calculate_test_metrics: 
        y_true = model.evaluate(pool, noise = noise)

    #Generate initial data
    sampler = LHS(d=dimensions, seed=random_state)
    if initialization == 'random':
        sample_x_unscaled = sampler.random(initial_samples)
        sample_x = scale(sample_x_unscaled, *lim)
    elif initialization == 'GSx':
        sample_x, _ = run_continuous_batch_learning(model, 
           regression_model,
           acquisition_function = 'GSx',
           opt_method = opt_method,
           pool = pool, 
           batch_size = 1,
           noise=noise,
           initial_samples=1, 
           active_learning_steps=initial_samples,
           lim=lim,
           alpha=alpha,
           n_jobs=n_jobs,
           random_state=rng,
           initialization='random',
           pso_options=pso_options,
           poly_degree = poly_degree,
           calculate_test_metrics=False,
           custom_acfn_input=custom_acfn_input
           )
    else:
        raise Exception('Initialization method not implemented')


    observation_y = model.evaluate(sample_x, noise=noise)

    scores_train = np.zeros((active_learning_steps+1,3))
    max_value = np.zeros((active_learning_steps+1,1))
    n_observations = np.linspace(initial_samples, initial_samples+(active_learning_steps)*batch_size,
                            active_learning_steps+1)
    if calculate_test_metrics:
        #To save the MSE
        scores_test = np.zeros((active_learning_steps+1,3))

    #Fit initial model
    if isinstance(reg_model_pure, LinearRegression):
        sample_x_poly = poly_transformer.fit_transform(sample_x)
        regression_model.fit(sample_x_poly, observation_y)
    else:
        regression_model.fit(sample_x, observation_y)


    #Initial model predictions  
    if calculate_test_metrics:  
        if isinstance(reg_model_pure, GPR):
            mean, std = regression_model.predict(pool,return_std=True)
        elif isinstance(reg_model_pure, LinearRegression):
            mean = regression_model.predict(pool_poly)
        else:
            mean = regression_model.predict(pool)

        #Save scores
        scores_test[0,0] = mean_squared_error(y_true, mean)
        scores_test[0,1] = mean_absolute_error(y_true, mean)
        scores_test[0,2] = max_error(y_true, mean)

    if isinstance(reg_model_pure, GPR):
        mean_train, std_train = regression_model.predict(sample_x,return_std=True)
    elif isinstance(reg_model_pure, LinearRegression):
        mean_train = regression_model.predict(sample_x_poly)
    else:
        mean_train = regression_model.predict(sample_x)

    scores_train[0,0] = mean_squared_error(observation_y, mean_train)
    scores_train[0,1] = mean_absolute_error(observation_y, mean_train)
    scores_train[0,2] = max_error(observation_y, mean_train)
    max_value[0,0] = np.max(observation_y)

    #Start active learning
    for i in range(active_learning_steps):

        #Save batch results separately, since observation is only estimated
        batch_sample = np.zeros((batch_size, dimensions))
        estimated_observation_y = observation_y.copy()
        estimated_sample_x = sample_x.copy()
        if isinstance(reg_model_pure, LinearRegression):
            estimated_sample_x_poly = sample_x_poly.copy()
        else:
            estimated_sample_x_poly = None
        
        for j in range(batch_size):
            
            if j != 0:
                if isinstance(reg_model_pure, LinearRegression):
                    regression_model.fit(estimated_sample_x_poly, estimated_observation_y)
                else:
                    regression_model.fit(estimated_sample_x, estimated_observation_y)
                #if isinstance(regression_model, GPR):
                #    mean, std = regression_model.predict(pool,return_std=True)
                #elif isinstance(regression_model, LinearRegression):
                #    mean = regression_model.predict(pool_poly)
                #else:
                #    mean = regression_model.predict(pool)

            #Choose from optimization routines
            #TODO: enable more customizability of parameters for optimization routines

            if isinstance(reg_model_pure, LinearRegression):
                poly_x = poly_transformer
            else:
                poly_x = None

            new_x, _ = step_continuous(acquisition_function, 
                            opt_method, regression_model, 
                            estimated_observation_y, estimated_sample_x,estimated_sample_x_poly,
                            custom_acfn_input, alpha, sampler, lim, dimensions, poly_x,n_jobs, pso_options, rng)

            #Assume estimated predictions
            if isinstance(reg_model_pure, GPR):
                mean_new, std_new = regression_model.predict(new_x.reshape(1,-1), return_std=True)
            elif isinstance(reg_model_pure, LinearRegression):
                new_x_poly = poly_transformer.fit_transform(new_x.reshape(-1,1), axis=0)
                mean_new = regression_model.predict(new_x_poly)
                std_new = fictive_noise_level
            else:
                mean_new = regression_model.predict(new_x.reshape(1,-1))
                std_new = fictive_noise_level
            estimated_observation_new = mean_new+rng.normal(0, std_new, size=1)
            
            #Store the new estimated observations
            estimated_sample_x = np.vstack([estimated_sample_x, new_x])
            if isinstance(reg_model_pure, LinearRegression):
                estimated_sample_x_poly = poly_transformer.fit_transform(estimated_sample_x)
            estimated_observation_y = np.hstack([estimated_observation_y, estimated_observation_new])
            batch_sample[j,...] = new_x

        # Updated pool with batch data
        sample_x = np.vstack([sample_x, batch_sample])
        if isinstance(reg_model_pure, LinearRegression):
            sample_x_poly = poly_transformer.fit_transform(sample_x)
        observation_new = model.evaluate(batch_sample, noise=noise)
        observation_y = np.hstack([observation_y, observation_new])

        # Fit new model with updated real training set
        if isinstance(reg_model_pure, LinearRegression):
            regression_model.fit(sample_x_poly, observation_y)
        else:
            regression_model.fit(sample_x, observation_y)

        if calculate_test_metrics:
            if isinstance(reg_model_pure, GPR):
                mean, std = regression_model.predict(pool,return_std=True)
            elif isinstance(reg_model_pure, LinearRegression):
                mean = regression_model.predict(pool_poly)
            else:
                mean = regression_model.predict(pool)

            # Calculate and save scores
            scores_test[i+1,0] = mean_squared_error(y_true, mean)
            scores_test[i+1,1] = mean_absolute_error(y_true, mean)
            scores_test[i+1,2] = max_error(y_true, mean)

        if isinstance(reg_model_pure, GPR):
            mean_train, std_train = regression_model.predict(sample_x,return_std=True)
        elif isinstance(reg_model_pure, LinearRegression):
            mean_train = regression_model.predict(sample_x_poly)
        else:
            mean_train = regression_model.predict(sample_x)

        scores_train[i+1,0] = mean_squared_error(observation_y, mean_train)
        scores_train[i+1,1] = mean_absolute_error(observation_y, mean_train)
        scores_train[i+1,2] = max_error(observation_y, mean_train)
        max_value[i+1,0] = np.max(observation_y)
        
    #transform results to a pandas DataFrame
    if calculate_test_metrics:
        results = np.vstack([n_observations, scores_train.T, scores_test.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
    else:
        results = np.vstack([n_observations, scores_train.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])

    
    return sample_x, results


def run_batch_learning(model, 
           regression_model,
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
           custom_acfn_input = None
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

    reg_model_pure = check_model(regression_model, acquisition_function)
    
    #Set random number generator
    if isinstance(random_state, int) or random_state==None:
        rng = np.random.RandomState(seed=random_state)

    poly_transformer = PolynomialFeatures(degree=poly_degree)


    #Generate a pool of sample data points
    if not isinstance(pool, np.ndarray):
        dimensions = model.n_features
        x = []
        for i in range(dimensions):
            x.append(np.linspace(*lim, 20))

        pool = np.meshgrid(*x)
        pool = np.array(pool).T
        pool = pool.reshape(len(x[0]**dimensions), dimensions)

    pool_poly = poly_transformer.fit_transform(pool)

    n_data = len(pool)
    #Number of data points in pool
    if calculate_test_metrics:
        if isinstance(test_set, np.ndarray):
            test_set_poly = poly_transformer.fit_transform(test_set)
            y_true_test = model.evaluate(test_set, noise = noise)
            extra_test_set = True
        else:
            y_true = model.evaluate(pool, noise = noise)
            extra_test_set = False

    #Randomly pick data points in pool for initial observations
    if initialization == 'random':
        rand_num = rng.randint(0,n_data, size=initial_samples)
    elif initialization == 'GSx':
        initial_data, _ = run_batch_learning(model, 
           regression_model,
           acquisition_function = 'GSx',
           pool=pool, 
           batch_size=1,
           noise=noise,
           initial_samples=1, 
           active_learning_steps=initial_samples-1,
           lim=lim,
           alpha=alpha,
           random_state=rng,
           return_samples=return_samples,
           initialization='random',
           poly_degree=poly_degree,
           custom_acfn_input=custom_acfn_input
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
    observation_y = model.evaluate(sample_x, noise = noise)

    data_indices = rand_num.copy()

    #To save the MSE
    scores_train = np.zeros((active_learning_steps+1,3))
    max_value = np.zeros((active_learning_steps+1,1))
    n_observations = np.linspace(initial_samples, initial_samples+(active_learning_steps)*batch_size,
                                 active_learning_steps+1)
    if calculate_test_metrics:
        scores_test = np.zeros((active_learning_steps+1,3))

    if isinstance(reg_model_pure, LinearRegression):
        regression_model.fit(sample_x_poly, observation_y)
    else:
        regression_model.fit(sample_x, observation_y)

    #Calculate metrics
    if calculate_test_metrics:
        if extra_test_set == False:
            mask = np.ones(y_true.size, dtype=bool)
            mask[data_indices] = False
            mask_indices = np.where(mask==True)[0]
            test_set = pool[mask_indices]
            test_set_poly = pool_poly[mask_indices]
            y_true_test = y_true[mask_indices]

        if isinstance(reg_model_pure, GPR):   
            mean_test, std_test = regression_model.predict(test_set,return_std=True)
        elif isinstance(reg_model_pure, LinearRegression):
            mean_test = regression_model.predict(test_set_poly)
        else:
            mean_test = regression_model.predict(test_set)

        scores_test[0,0] = mean_squared_error(y_true_test, mean_test)
        scores_test[0,1] = mean_absolute_error(y_true_test, mean_test)
        scores_test[0,2] = max_error(y_true_test, mean_test)
        

    if isinstance(reg_model_pure, GPR):   
        mean, std = regression_model.predict(pool,return_std=True)
    elif isinstance(reg_model_pure, LinearRegression):
        mean = regression_model.predict(pool_poly)
        std = None
    else:
        mean = regression_model.predict(pool)
        std = None

    #Save scores
    scores_train[0,0] = mean_squared_error(observation_y, mean[data_indices])
    scores_train[0,1] = mean_absolute_error(observation_y, mean[data_indices])
    scores_train[0,2] = max_error(observation_y, mean[data_indices])
    max_value[0,0] = np.max(observation_y)

    
    #Start active learning
    for i in range(active_learning_steps):

        batch_indices = np.zeros(batch_size, dtype=np.int32)
        estimated_observation_y = observation_y.copy()
        estimated_sample_x = sample_x.copy()
        if isinstance(reg_model_pure, LinearRegression):
            estimated_sample_x_poly = sample_x_poly.copy()
        
        for j in range(batch_size):

            if j != 0:
                if isinstance(reg_model_pure, LinearRegression):
                    regression_model.fit(estimated_sample_x_poly, estimated_observation_y)
                else:
                    regression_model.fit(estimated_sample_x, estimated_observation_y)
                if isinstance(reg_model_pure, GPR):
                    mean, std = regression_model.predict(pool,return_std=True)
                elif isinstance(reg_model_pure, LinearRegression):
                    mean = regression_model.predict(pool_poly)
                else:
                    mean = regression_model.predict(pool)

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

            #Choose an acquisition function

            if acquisition_function == 'qbc':
                models = []
                for _ in range(alpha):
                    train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                    if isinstance(reg_model_pure, LinearRegression):
                        regression_model.fit(estimated_sample_x_poly[train_index], estimated_observation_y[train_index])
                    else:
                        regression_model.fit(estimated_sample_x[train_index], estimated_observation_y[train_index])
                    models.append(copy.deepcopy(regression_model))
                acquisition = QBC(pool, models)

            acquisition = step_discrete(
                    acquisition_function, 
                    estimated_observation_y, 
                    alpha, mean, std,
                    n_data, data_indices, pool,
                    custom_acfn_input,
                    rng)

            #Find maximum of acquisition function for non yet evaluated data points

            #Avoid to sample already sampled data points
            acquisition_masked = acquisition[mask]
            index = np.where(acquisition_masked==np.max(acquisition_masked))[0]
            index = mask_indices[index]
            
            if len(index) > 1:
                ind = rng.randint(0,len(index),1)
                index = np.array(index[ind])
            data_indices = np.concatenate([data_indices, index])
                
            estimated_x_max = pool[index]
            estimated_sample_x = np.vstack([estimated_sample_x, estimated_x_max])
            if isinstance(reg_model_pure, LinearRegression):
                estimated_sample_x_poly = poly_transformer.fit_transform(estimated_sample_x)

            #Assume estimated predictions
            if isinstance(reg_model_pure, GPR):
                mean_new, std_new = regression_model.predict(pool[index].reshape(1,-1), return_std=True)
            elif isinstance(reg_model_pure, LinearRegression):
                new_x_poly = poly_transformer.fit_transform(pool[index].reshape(1,-1))
                mean_new = regression_model.predict(new_x_poly)
                std_new = fictive_noise_level
            else:
                mean_new = regression_model.predict(pool[index].reshape(1,-1))
                std_new = fictive_noise_level

            estimated_observation_new = mean_new+rng.normal(0, std_new, size=1)

            estimated_observation_y = np.hstack([estimated_observation_y, estimated_observation_new])
            batch_indices[j] = index

        # Updated pool
        x_max = pool[batch_indices]
        observation_new = model.evaluate(x_max, noise=noise)
        sample_x = np.vstack([sample_x, x_max])
        if isinstance(reg_model_pure, LinearRegression):
            sample_x_poly = poly_transformer.fit_transform(sample_x)
        observation_y = np.hstack([observation_y, observation_new])

        if isinstance(reg_model_pure, LinearRegression):
            regression_model.fit(sample_x_poly, observation_y)
        else:
            regression_model.fit(sample_x, observation_y)

        if calculate_test_metrics:
            if extra_test_set == False:
                mask = np.ones(y_true.size, dtype=bool)
                mask[data_indices] = False
                mask_indices = np.where(mask==True)[0]
                test_set = pool[mask_indices]
                test_set_poly = pool_poly[mask_indices]
                y_true_test = y_true[mask_indices]

            if isinstance(reg_model_pure, GPR):   
                mean_test, std_test = regression_model.predict(test_set,return_std=True)
            elif isinstance(reg_model_pure, LinearRegression):
                mean_test = regression_model.predict(test_set_poly)
            else:
                mean_test = regression_model.predict(test_set)

            scores_test[i+1,0] = mean_squared_error(y_true_test, mean_test)
            scores_test[i+1,1] = mean_absolute_error(y_true_test, mean_test)
            scores_test[i+1,2] = max_error(y_true_test, mean_test)
        

        if isinstance(reg_model_pure, GPR):   
            mean, std = regression_model.predict(pool,return_std=True)
        elif isinstance(reg_model_pure, LinearRegression):
            mean = regression_model.predict(pool_poly)
        else:
            mean = regression_model.predict(pool)

        #Save scores
        scores_train[i+1,0] = mean_squared_error(observation_y, mean[data_indices])
        scores_train[i+1,1] = mean_absolute_error(observation_y, mean[data_indices])
        scores_train[i+1,2] = max_error(observation_y, mean[data_indices])
        max_value[i+1,0] = np.max(observation_y)
        
    #transform results to a pandas DataFrame
    if calculate_test_metrics:
        results = np.vstack([n_observations, scores_train.T, scores_test.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'mean_MSE_test', 'mean_MAE_test', 'mean_MaxE_test', 'max_observation'])
    else:
        results = np.vstack([n_observations, scores_train.T, max_value.T])
        results = pd.DataFrame(results.T, columns=['m', 'mean_MSE_train', 'mean_MAE_train', 'mean_MaxE_train', 'max_observation'])
    
    return sample_x, results

################################################################################################################
#Old functions

def run_learning(model, 
           regression_model,
           acquisition_function = 'ei',
           pool = None, 
           noise=0.1,
           initial_samples=2, 
           active_learning_steps=10,
           lim=[-1,1],
           alpha=0,
           random_state=None,
           poly_degree = 3
           ):
    '''
    Pool-based active learning algorithm which runs only in sequential mode. This should not be used any more, but instead 
    'run_batch_learning' should be used with a batch_size of 1. This function only provides limited functionality and can be used for testing
    and as a reference for the 'run_batch_learning' function.

    Parameters
    ----------
    model : Model class
        Model to generate true data.
    regression_model: sklearn model, str, optional
        scikit-learn regression model.
    acquissition_function : str, optional
        Acquisition function. The default is 'ei'.
    pool : nd_array, None, optional
        Array containing pool of data. All data is sampled from that pool. Data which is not sampled so far is used for calculating 
        test metrics. For 'None' a grid with 100 points in each dimension is created. The default is 'None'.
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

    Returns
    -------
    results : pd_DataFrame
        Pandas DataFrame containing the columns: 'm', 'mean_mse_test', 'max_observation'. Data is stored for each active learning step, 
        where 'm' gives the number of data used for training, 'mean_mse_test' the MSE of the test set and 'max_observation' the maximum 
        value observed so far.

    '''

    if isinstance(regression_model, LinearRegression):
        if acquisition_function not in ['random', 'GSx', 'ideal', 'qbc']:
            print('Acquisition functin is not implemented for Linear Regression: {}'.format(acquisition_function))
            exit()
    elif not isinstance(regression_model, GPR):
        if acquisition_function not in ['random', 'GSx', 'ideal', 'qbc']:
            print('Acquisition functin is not implemented for Non-GPR models: {}'.format(acquisition_function))
            exit()
    else:
        if acquisition_function not in ['random', 'GSx', 'ideal', 'qbc', 'ei', 'ucb', 'poi']:
            print('Acquisition function not implemented: {}'.format(acquisition_function))
            exit()
    
    #Set random number generator
    rng = np.random.RandomState(seed=random_state)

    poly_transformer = PolynomialFeatures(degree=poly_degree)

    #Generate a pool of sample data points
    if not isinstance(pool, np.ndarray):
        dimensions = model.n_features
        x = []
        for i in range(dimensions):
            x.append(np.linspace(*lim, 1000))

        pool = np.meshgrid(*x)
        pool = np.array(pool).T
        pool = pool.reshape(len(x[0]**dimensions), dimensions)

    pool_poly = poly_transformer.fit_transform(pool)

    #Number of data points in pool
    n_data = len(pool)

    y_true = model.evaluate(pool, noise = noise)

    #Randomly pick data points in pool for initial observations
    rand_num = rng.randint(0,n_data, size=initial_samples)
    sample_x = pool[rand_num]
    sample_x_poly = poly_transformer.fit_transform(sample_x)
    observation_y = y_true[rand_num]

    data_indices = rand_num.copy()

    #To save the MSE
    mse = np.zeros((active_learning_steps,1))
    max_value = np.zeros((active_learning_steps,1))
    n_observations = np.linspace(initial_samples, initial_samples+active_learning_steps-1,active_learning_steps)
    
    #Start active learning
    for i in range(active_learning_steps):

        if isinstance(regression_model, LinearRegression):
            regression_model.fit(sample_x_poly, observation_y)
        else:
            regression_model.fit(sample_x, observation_y)

        mask = np.ones(y_true.size, dtype=bool)
        mask[data_indices] = False

        mask_z = np.ones(y_true.size, dtype=np.int32)
        mask_z[data_indices] = False

        if isinstance(regression_model, GPR):
            mean, std = regression_model.predict(pool,return_std=True)
        elif isinstance(regression_model, LinearRegression):
            mean = regression_model.predict(pool_poly)
        else:
            mean = regression_model.predict(pool)

        #Optional plotting for debugging
        #plt.plot(sample_x, observation_y, 'o')
        #plt.plot(pool, y_true, 'k')
        #plt.plot(pool, mean)
        #plt.show()

        #Choose an acquisition function
        if acquisition_function == 'poi':
            poi = POI(mean, std, opt=np.max(observation_y), max=True, alpha=alpha)
            poi = poi*mask_z
            index = np.where(poi==np.max(poi))[0]
        elif acquisition_function == 'ei':
            ei = EI(mean, std, opt=np.max(observation_y), max=True, alpha=alpha)
            ei = ei*mask_z
            index = np.where(ei==np.max(ei))[0]
        elif acquisition_function == 'ucb':
            ucb = UCB(mean, std, alpha=alpha)
            ucb = ucb*mask_z
            index = np.where(ucb==np.max(ucb))[0]
        elif acquisition_function == 'random':
            index = rng.randint(0, n_data, 1)
        elif acquisition_function =='ideal':
            imp = IDEAL(data_indices, pool, std, alpha)
            imp = imp*mask_z
            index = np.where(imp==np.max(imp))[0]
        elif acquisition_function == 'qbc':
            mean_qbc = 0
            std_qbc = 0
            #bootstrapping approach to train models
            for _ in range(alpha):
                train_index = rng.randint(0,len(sample_x),len(sample_x))
                if isinstance(regression_model, LinearRegression):
                    regression_model.fit(sample_x_poly[train_index], observation_y[train_index])
                else:
                    regression_model.fit(sample_x[train_index], observation_y[train_index])

                if isinstance(regression_model, GPR):
                    m, s = regression_model.predict(pool,return_std=True)
                elif isinstance(regression_model, LinearRegression):
                    m = regression_model.predict(pool_poly)
                else:
                    m = regression_model.predict(pool)
                mean_qbc += m
                std_qbc += s

            mean_qbc /= len(sample_x)
            std_qbc /= len(sample_x)

            std_qbc = std_qbc*mask_z
            index = np.where(std_qbc==np.max(std_qbc))[0]
        elif acquisition_function == 'GSx':
            gsx = GSx(data_indices, pool)
            index = np.where(gsx==np.max(gsx))[0]

        else:
            raise Exception('Acquisition function not implemented')
        
        if len(index) > 1:
            ind = rng.randint(0,len(index),1)
            index = np.array(index[ind])
        data_indices = np.concatenate([data_indices, index])
            
        x_max = pool[index]
        observation_new = y_true[index]

        scores = mean_squared_error(y_true[mask], mean[mask])

        #Save scores
        mse[i,0] = scores
        max_value[i,0] = np.max(observation_y)

        #Update the sample and observations with new data
        sample_x = np.vstack([sample_x, x_max])
        sample_x_poly = poly_transformer.fit_transform(sample_x)
        observation_y = np.hstack([observation_y, observation_new])


    #transform restuls to an pandas DataFrame
    results = np.vstack([n_observations, mse.T, max_value.T])
    results = pd.DataFrame(results.T, columns=['m', 'mean_mse_test', 'max_observation'])
    
    return results