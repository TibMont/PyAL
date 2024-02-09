import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import scale
from scipy.stats import norm

from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

import copy
from pyswarms.single.global_best import GlobalBestPSO

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

def GSy_con(x, y_sample, model, aggregation_function, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        y_individual = np.zeros((len(model), 1))
    else:
        y_individual = np.zeros((len(model), len(x)))
    for i in range(len(model)):
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

def iGS_con(x, x_sample, y_sample, model, aggregation_function, *args, **kwargs):
    gsx = GSx_con(x,x_sample)
    gsy = GSy_con(x, y_sample, model, aggregation_function, *args, **kwargs)

    min_dist = gsx*gsy
    return -min_dist
    
def IDEAL_con(x, x_samples, model, aggregation_function, alpha=1, *args, **kwargs):
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

def QBC_con(x, models, aggregation_function, *args, **kwargs):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        n_pool = 1
    else:
        n_pool = len(x)

    mean_qbc_individual = np.zeros((len(models), n_pool))
    std_qbc_individual = np.zeros((len(models), n_pool))
    #bootstrapping approach to train models
    for i in range(len(models)):
        model = models[i]
        for mod in model:
            m, s = mod.predict(x,return_std=True)
            mean_qbc_individual[i,...] += m
            std_qbc_individual[i,...] += s

    mean_qbc_individual /= len(models[0])
    std_qbc_individual /= len(models[0])

    if len(args) != 0:
        mean_qbc = aggregation_function(mean_qbc_individual, *args)
        std_qbc = aggregation_function(std_qbc_individual, uncert=True, *args)
    else:
        mean_qbc = aggregation_function(mean_qbc_individual, **kwargs)
        std_qbc = aggregation_function(std_qbc_individual, uncert=True, **kwargs)

    return -std_qbc


########################################################################################################

def run_continuous_batch_learning_multi(models, 
           aggregation_function,
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
           verbose=True,
           **kwargs
           ):
    '''
    Multi-objective batch-mode Active Learning in continuous parameter space. This method is similar to 
    'run_continuous_batch_learning' but also enables to take into account multiple objective functions. It lacks some functionality of 
    the previously mentioned function from which it is derived.
    Multiple objectives are handeled by an aggregation function, which combines the individual output for each data point of 
    each objective into a single output value. 
    The active learning functions are modified in order to also take the aggregation function as an argument. Therefore, active learning 
    functions from outside this module are not compatible.

    The algorithm generates initial data automatically using either a random approach or using the model-free GSx approach. 

    Parameters
    ----------
    models : List of model class
        Models to generate true data for each objective.
    aggregation_function : callable
        Function to aggregate multiple outputs for various objective functions.
    regression_model: sklearn model, str, optional
        scikit-learn regression model.
    acquissition_function : str, optional
        Acquisition function. The default is 'ideal'.
    opt_method : str, optional
        The method to optimize the acquisition function. Choose from 'lbfgs' and 'PSO'. The default is 'PSO'.
    pool : nd_array, None, optional
        Array containing pool of data. For 'None' a grid with 100 points in each dimension is created. The default is 'None'.
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
    n_jobs : int, optional
        Number of cores to use for parallel evaluation of PSO. Currently not used, PSO runs only in serial mode. The default is 1.
    random_state: int, optional
        Set random state. The default is None.
    verbose : bool, optional
        Whether to print additional information. The default value is True.
    **kwargs : various, optional
        Keyword arguments for the aggregation function.

    Returns
    -------
    results : pd_DataFrame
        Pandas DataFrame containing the columns: 'm', 'mean_mse_test', 'max_observation'. Data is stored for each active learning step, 
        where 'm' gives the number of data used for training, 'mean_mse_test' the MSE of the test set and 'max_observation' the maximum 
        value observed so far.

    '''
    
    #Set random number generator
    rng = np.random.RandomState(seed=random_state)

    #Generate a pool of sample data points for testing
    dimensions = models[0].n_features
    n_models = len(models)

    if not isinstance(pool, np.ndarray):
        x = []
        for i in range(dimensions):
            x.append(np.linspace(*lim, 10))

        pool = np.meshgrid(*x)
        pool = np.array(pool).T
        pool = pool.reshape(len(x[0]**dimensions), dimensions)

    if isinstance(noise, int) or isinstance(noise, float):
        noise_old = noise
        noise = [noise for _ in range(n_models)]
        if verbose:
            print('Noise converted: ')
            print('from {} to {}'.format(noise_old, noise))

    #Number of data points in pool
    n_data = len(pool)
    y_true = np.zeros((n_models, n_data))
    for i in range(n_models):
        model = models[i]
        y_true[i, ...] = model.evaluate(pool, noise = noise[i])

    y_true_aggregated = aggregation_function(y_true, **kwargs)

    #Generate initial data
    sampler = LHS(d=dimensions)
    sample_x_unscaled = sampler.random(initial_samples)
    sample_x = scale(sample_x_unscaled, *lim)

    observation_y = np.zeros((n_models, len(sample_x)))
    for i in range(n_models):
        observation_y[i, ...] = models[i].evaluate(sample_x, noise=noise[i])

    observation_y_aggregated = aggregation_function(observation_y, **kwargs)

    #To save the MSE
    mse = np.zeros((active_learning_steps+1,1))
    max_value = np.zeros((active_learning_steps+1,1))
    n_observations = np.linspace(initial_samples, initial_samples+(active_learning_steps)*batch_size,
                                 active_learning_steps+1)

    #Fit initial model
    regression_models = []
    mean = np.zeros((n_models, len(pool)))
    std = np.zeros((n_models, len(pool)))

    for i in range(n_models):
        regression_model.fit(sample_x, observation_y[i])
        mean[i,...], std[i,...] = regression_model.predict(pool,return_std=True)
        regression_models.append(copy.deepcopy(regression_model))
    
    mean_aggregated = aggregation_function(mean, **kwargs)
    
    scores = mean_squared_error(y_true_aggregated, mean_aggregated)

    #Save scores
    mse[0,0] = scores
    max_value[0,0] = np.max(observation_y_aggregated)
    ###############################################################
    #ToDO: implement loop over several model functions from here on
    ###############################################################

    #Start active learning
    for a in range(active_learning_steps):

        #Save batch results separately, since observation is only estimated
        batch_sample = np.zeros((batch_size, dimensions))
        estimated_observation_y = observation_y.copy()
        estimated_sample_x = sample_x.copy()
        estimated_observation_y_aggregated = observation_y_aggregated.copy()
        
        for j in range(batch_size):

            #For the first sample in a batch we can use the model with which we evaluated the scores
            if j != 0:
                mean = np.zeros((n_models, len(pool)))
                std = np.zeros((n_models, len(pool)))

                for i in range(n_models):
                    regression_models[i].fit(estimated_sample_x, estimated_observation_y[i])
                    mean[i, ...], std[i, ...] = regression_model.predict(pool,return_std=True)

                mean_aggregated = aggregation_function(mean, **kwargs)

            #Choose from optimization routines
            #TODO: enable more customizability of parameters for optimization routines

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
                    res = minimize(IDEAL_con, x0=x0, args=(estimated_sample_x, regression_models, aggregation_function, alpha, *optargs), 
                                bounds=lim_t)
                elif acquisition_function == 'GSx':
                    res = minimize(GSx_con, x0=x0, args=(estimated_sample_x),
                            bounds=lim_t)
                elif acquisition_function == 'GSy':
                    res = minimize(GSy_con, x0=x0, args=(estimated_observation_y_aggregated, regression_models, aggregation_function, *optargs),
                            bounds=lim_t)
                elif acquisition_function == 'iGS':
                    res = minimize(iGS_con, x0=x0, args=(estimated_sample_x, estimated_observation_y_aggregated, regression_models, aggregation_function, *optargs),
                            bounds=lim_t)
                elif acquisition_function == 'qbc':
                    S_models = []
                    for i in range(n_models):
                        alpha_models = []
                        for _ in range(alpha):
                            train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                            regression_models[i].fit(estimated_sample_x[train_index], estimated_observation_y[i][train_index])
                            alpha_models.append(copy.deepcopy(regression_models[i]))
                        S_models.append(alpha_models)
                    res = minimize(QBC_con, x0=x0, args=(S_models, aggregation_function, *optargs),
                            bounds=lim_t)
                else:
                    raise Exception('Acquisition function not implemented')
            
                new_x = res.x

            #Simple Particle Swarm Optimization
            #TODO: enable to customly choose hyperparameters
            elif opt_method == 'PSO':
                pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
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
                                                    model=regression_models, aggregation_function=aggregation_function,
                                                    alpha=alpha, **kwargs)
                elif acquisition_function == 'GSx':
                    cost, new_x = optimizer.optimize(GSx_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     x_sample=estimated_sample_x)
                elif acquisition_function == 'GSy':
                    cost, new_x = optimizer.optimize(GSy_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     y_sample=estimated_observation_y_aggregated,
                                                     model = regression_models, aggregation_function=aggregation_function, **kwargs)
                elif acquisition_function == 'iGS':
                    cost, new_x = optimizer.optimize(iGS_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     x_sample=estimated_sample_x,
                                                     y_sample=estimated_observation_y_aggregated, model=regression_models, 
                                                     aggregation_function=aggregation_function, **kwargs)
                elif acquisition_function == 'qbc':
                    S_models = []
                    for i in range(n_models):
                        alpha_models = []
                        for _ in range(alpha):
                            train_index = rng.randint(0,len(estimated_sample_x),len(estimated_sample_x))
                            regression_models[i].fit(estimated_sample_x[train_index], estimated_observation_y[i][train_index])
                            alpha_models.append(copy.deepcopy(regression_models[i]))
                        S_models.append(alpha_models)
                    cost, new_x = optimizer.optimize(QBC_con, iters=200, verbose=False, n_processes=n_jobs,
                                                     models=S_models, aggregation_function=aggregation_function, **kwargs)
                else:
                    raise Exception('Acquisition function not implemented')
                
            else:
                raise Exception('Optimization method not implemented')

            #Assume estimated predictions

            mean_new = np.zeros((n_models, len(new_x.reshape(1,-1))))
            std_new = np.zeros((n_models, len(new_x.reshape(1,-1))))
            estimated_observation_new = np.zeros((n_models, len(new_x.reshape(1,-1))))

            for i in range(n_models):
                mean_new[i,...], std_new[i,...] = regression_models[i].predict(new_x.reshape(1,-1), return_std=True)
                estimated_observation_new[i,...] = mean_new[i,]+rng.normal(0, std_new[i], size=1)
            
            estimated_observation_new_aggregated = aggregation_function(estimated_observation_new, **kwargs)

            
            #Store the new estimated observations
            estimated_sample_x = np.vstack([estimated_sample_x, new_x])
            estimated_observation_y = np.hstack([estimated_observation_y, estimated_observation_new])
            estimated_observation_y_aggregated = np.hstack([estimated_observation_y_aggregated, 
                                                            estimated_observation_new_aggregated])
            batch_sample[j,...] = new_x

        #########################################################################################################
        # Updated pool with batch data
        sample_x = np.vstack([sample_x, batch_sample])

        observation_new = np.zeros((n_models, len(batch_sample)))
        for i in range(n_models):
            observation_new[i,...] = models[i].evaluate(batch_sample, noise=noise[i])

        observation_y = np.hstack([observation_y, observation_new])

        observation_new_aggregated = aggregation_function(observation_y, **kwargs)
        observation_y_aggregated = np.hstack([observation_y_aggregated, observation_new_aggregated])

        # Fit new model with updated real training set
        mean_new = np.zeros((n_models, len(pool)))
        std_new = np.zeros((n_models, len(pool)))
        if verbose: print('Active learning step: {}'.format(a))
        for i in range(n_models):
            regression_models[i].fit(sample_x, observation_y[i])
            mean[i,...], std[i,...] = regression_models[i].predict(pool,return_std=True)
            if verbose: 
                print('Model {}'.format(i))
                if isinstance(regression_models[i], Pipeline):
                    print(regression_models[i].named_steps['model'].kernel_)
                else:
                    print(regression_models[i].kernel_)


        mean_aggregated = aggregation_function(mean, **kwargs)

        # Calculate and save scores
        scores = mean_squared_error(y_true_aggregated, mean_aggregated)
        mse[a+1,0] = scores
        max_value[a+1,0] = np.max(observation_y_aggregated)
        
    #transform results to a pandas DataFrame
    results = np.vstack([n_observations, mse.T, max_value.T])
    results = pd.DataFrame(results.T, columns=['m', 'mean_mse_test', 'max_observation'])
    
    return results