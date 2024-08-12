# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
try:
    from IPython.display import HTML
except:
    print('IPython not found.')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import scale
from scipy.stats import norm

from scipy.optimize import minimize
from pyswarms.single.global_best import GlobalBestPSO

from PyAL.acfn_discrete import EI, POI, UCB, IDEAL, GSx, GSy, SGSx, iGS
from PyAL.acfn_continuous import EI_con, POI_con, UCB_con, IDEAL_con, GSx_con, GSy_con, iGS_con, QBC_con, std_con

def max_acquisition(acquisition, grid, rng=None):
    if rng == None:
        rng = np.random.RandomState(seed=None)
    max_acquisition = np.max(acquisition)
    max_index = np.where(acquisition==max_acquisition)[0]
    if len(max_index) > 1:
        ind = rng.randint(0,len(max_index),1)
        max_index = max_index[ind]
    max_x = grid[max_index]
    return max_x, max_acquisition, max_index

def create_animation(
    model,
    gpr,
    kernel, 
    acquisition_function,
    grid_simple,
    n_iterations = 30,
    alpha = 0,
    n_observations = 3,
    noise_level = 0,
    html = True,
    random_state = 42,
    legend = True
):
    rng = np.random.RandomState(seed=random_state)
    
    data_indices = rng.randint(0, len(grid_simple),n_observations)
    sample_x = grid_simple[data_indices]
    observation_y = model.evaluate(sample_x, noise=noise_level)

    y_true = model.evaluate(grid_simple, noise=0)

    #It is good practice to scale features. A pipeline applies scaling before fitting
    pipe = Pipeline([('scaler',StandardScaler()), ('gpr',gpr)])

    #We save every timestep in the collect variables
    sample_x_collect = []
    observation_y_collect = []
    mean_collect = []
    std_collect = []
    acquisition_collect = []
    x_max_collect = []
    acquisition_max_collect = []

    #Run the Active Learning for n_iterations
    for i in range(n_iterations):
        #Make model predictions
        gpr.fit(sample_x, observation_y)
        mean, std = gpr.predict(grid_simple,return_std=True)
        if acquisition_function == 'ei':
            acquisition = EI(mean, std, opt=np.max(observation_y), max=True, alpha=alpha)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'poi':
            acquisition = POI(mean, std, opt=np.max(observation_y), max=True, alpha=alpha)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'ucb':
            acquisition = UCB(mean, std, alpha=alpha)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'random':
            acquisition = rng.random(size=len(grid_simple))
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function =='ideal':
            acquisition = IDEAL(data_indices, grid_simple, mean, y_true, alpha)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'GSx':
            acquisition = GSx(data_indices, grid_simple)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'GSy':
            acquisition = GSy(mean, observation_y)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'iGS':
            acquisition = iGS(data_indices, grid_simple, mean, observation_y)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)
        elif acquisition_function == 'SGSx':
            acquisition = SGSx(data_indices, grid_simple, std)
            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)

        elif acquisition_function == 'qbc':
            mean_qbc = 0
            std_qbc = 0
            #bootstrapping approach to train models
            for _ in range(alpha):
                train_index = rng.randint(0,len(sample_x),len(sample_x))
                gpr.fit(sample_x[train_index], observation_y[train_index])
                m, s = gpr.predict(grid_simple,return_std=True)
                mean_qbc += m
                std_qbc += s

            mean_qbc /= len(sample_x)
            std_qbc /= len(sample_x)
            acquisition = np.array(std_qbc)

            x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)

        else:
            raise Exception('Acquisition function not implemented')
            
        data_indices = np.hstack([data_indices, max_index])

        observation_new = model.evaluate(x_max.reshape(-1,1), noise=noise_level)

        #Save each active learning step
        sample_x_collect.append(sample_x.copy())
        observation_y_collect.append(observation_y.copy())
        mean_collect.append(mean.copy())
        std_collect.append(std.copy())
        acquisition_collect.append(acquisition.copy())
        x_max_collect.append(x_max.copy())
        acquisition_max_collect.append(acquisition_max.copy())

        #Update the sample and observations with new data
        sample_x = np.vstack([sample_x, x_max])
        observation_y = np.hstack([observation_y, observation_new])
    #Create a figure
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[2,1]}, sharex=True)
    fig.set_tight_layout(True)


    def plot_anim(sample_x, observation_y, grid, mean, std, acquisition, max_x, max_acquisition, title=None, legend=True):
        n_data = grid.shape[0]

        #Clear the plot
        ax1.clear()
        ax2.clear()

        #Set title
        if title != None:
            ax1.set_title(title)

        #Plot observations, true model and predicted model
        ax1.plot(grid.reshape(n_data), mean, '-', label='Prediction')
        ax1.plot(grid_simple.reshape(n_data),y_true, 'k-', label='True')
        ax1.plot(sample_x, observation_y,'ko', markerfacecolor='white', label='Observation')
        ax1.set_ylabel('$f(x)$')
        if legend==True:
            ax1.legend(loc='lower center')
        ax1.fill_between(grid.reshape(n_data),mean-std*3,mean+std*3, alpha=0.2)

        #Plot acquisition function
        ax2.plot(grid.reshape(n_data), acquisition)
        ax2.plot(max_x, max_acquisition, 'ro')
        ax2.set_ylabel('Acquisition func.')
        ax2.set_xlabel('$x$')
    
        #Set ylim for acquisition function
        max_acquisition = max(acquisition)+np.abs(max(acquisition))*0.1
        min_acquisition = -max_acquisition
        ax2.set_ylim(min_acquisition, max_acquisition)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    #Create the animation function
    def animation_function(frame, sample_x_collect, observation_y_collect, grid_simple, mean_collect, std_collect, ei_collect, x_max_collect, ei_max_collect, legend=True):
        plot_anim(sample_x_collect[frame], observation_y_collect[frame], grid_simple, mean_collect[frame], std_collect[frame], 
        acquisition_collect[frame], x_max_collect[frame], acquisition_max_collect[frame], title='Iteration {}'.format(frame), legend=legend)
        
    #Create the animation
    anim_created = FuncAnimation(fig, animation_function, frames=n_iterations, interval=1000, fargs=(sample_x_collect, observation_y_collect, grid_simple, mean_collect, 
                                std_collect, acquisition_collect, x_max_collect, acquisition_max_collect, legend))
    plt.close()
    if html == True:
        video = anim_created.to_jshtml()
        html = HTML(video)
        return html
    else:
        return anim_created

    

def create_animation_continuous(
    model,
    gpr,
    kernel, 
    acquisition_function,
    grid_simple,
    n_iterations = 30,
    alpha = 0,
    n_observations = 3,
    noise_level = 0,
    html = True, 
    random_state = 42,
    opt_method = 'lbfgs',
    **kwargs
):
    rng = np.random.RandomState(seed=random_state)

    dimensions = model.n_features

    lim = [min(grid_simple), max(grid_simple)]

    sampler = LHS(d=dimensions)
    sample_x_unscaled = sampler.random(n_observations)
    sample_x = scale(sample_x_unscaled, *lim)
    observation_y = model.evaluate(sample_x, noise=noise_level)

    y_true = model.evaluate(grid_simple, noise=0)

    #It is good practice to scale features. A pipeline applies scaling before fitting
    pipe = Pipeline([('scaler',StandardScaler()), ('gpr',gpr)])

    #We save every timestep in the collect variables
    sample_x_collect = []
    observation_y_collect = []
    mean_collect = []
    std_collect = []
    acquisition_collect = []
    x_max_collect = []
    acquisition_max_collect = []

    #Run the Active Learning for n_iterations
    for i in range(n_iterations):
        #Make model predictions
        gpr.fit(sample_x, observation_y)
        mean, std = gpr.predict(grid_simple,return_std=True)

        kargs = kwargs.values()

        if opt_method == 'lbfgs':
        
            if acquisition_function =='ideal':
                acquisition = IDEAL_con(grid_simple, sample_x, gpr, observation_y ,lim, alpha, **kwargs)
                o = np.where(acquisition == np.nan)
                x0_unscaled = sampler.random(1)[0]
                x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
                res = minimize(IDEAL_con, x0=x0, args=(sample_x, gpr, observation_y ,lim, alpha, *kargs), 
                                bounds=[lim for i in range(dimensions)])
                new_x = res.x
                new_y = np.array([res.fun])
                mean_new, std_new = gpr.predict(new_x.reshape(1,-1), return_std=True)
                observation_new = model.evaluate(new_x.reshape(1,-1), noise=noise_level)
                #observation_new = mean+rng.normal(0, std_new, size=1)
            elif acquisition_function == 'GSx':
                acquisition = GSx_con(grid_simple, sample_x)
                x0_unscaled = sampler.random(1)[0]
                x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
                res = minimize(GSx_con, x0=x0, args=(sample_x),
                                bounds=[lim for i in range(dimensions)])
                new_x = res.x
                new_y = np.array([res.fun])
                mean_new, std_new = gpr.predict(new_x.reshape(1,-1), return_std=True)
                observation_new = model.evaluate(new_x.reshape(1,-1), noise=noise_level)
                #observation_new = mean+rng.normal(0, std_new, size=1)
            elif acquisition_function == 'GSy':
                acquisition = GSy_con(grid_simple, observation_y, gpr)
                x0_unscaled = sampler.random(1)[0]
                x0 = scale(x0_unscaled.reshape(1,-1), *lim).reshape(dimensions)
                res = minimize(GSy_con, x0=x0, args=(observation_y, gpr),
                                bounds=[lim for i in range(dimensions)])
                new_x = res.x
                new_y = np.array([res.fun])
                mean_new, std_new = gpr.predict(new_x.reshape(1,-1), return_std=True)
                observation_new = model.evaluate(new_x.reshape(1,-1), noise=noise_level)
            else:
                raise Exception('Acquisition function not implemented')

        elif opt_method == 'PSO':
            pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
            lb = lim[0]
            ub = lim[1]
            bounds = [lb,ub]
            n_particles = 5
            optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=pso_options, 
                                    bounds=bounds)

            if acquisition_function == 'ideal':
                acquisition = IDEAL_con(grid_simple, sample_x, gpr, observation_y ,lim, alpha, **kwargs)
                cost, new_x = optimizer.optimize(IDEAL_con, iters=50, verbose=False,
                                                x_samples=sample_x,
                                                model=gpr, y_true=observation_y, lim=lim, alpha=alpha, **kwargs)
                new_y = cost
            else:
                raise Exception('Acquisition function not implemented')              

        observation_new = model.evaluate(new_x.reshape(-1,1), noise=noise_level)

        #Save each active learning step
        sample_x_collect.append(sample_x.copy())
        observation_y_collect.append(observation_y.copy())
        mean_collect.append(mean.copy())
        std_collect.append(std.copy())
        acquisition_collect.append(-1*acquisition.copy())
        x_max_collect.append(new_x.copy())
        acquisition_max_collect.append(-1*new_y.copy())

        #Update the sample and observations with new data
        sample_x = np.vstack([sample_x, new_x])
        observation_y = np.hstack([observation_y, observation_new])
    #Create a figure
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[2,1]}, sharex=True)
    fig.set_tight_layout(True)

    def plot_anim(sample_x, observation_y, grid, mean, std, acquisition, max_x, max_acquisition, title=None):
        n_data = grid.shape[0]

        #Clear the plot
        ax1.clear()
        ax2.clear()

        #Set title
        if title != None:
            ax1.set_title(title)

        #Plot observations, true model and predicted model
        ax1.plot(grid.reshape(n_data), mean, '-', label='Prediction')
        ax1.plot(grid_simple.reshape(n_data),y_true, 'k-', label='True')
        ax1.plot(sample_x, observation_y,'ko', markerfacecolor='white', label='Observation')
        ax1.set_ylabel('$f(x)$')
        ax1.legend(loc='upper right', fontsize=10)
        #ax1.fill_between(grid.reshape(n_data),mean-std*3,mean+std*3, alpha=0.2)

        #Plot acquisition function
        ax2.plot(grid.reshape(n_data), acquisition)
        ax2.plot(max_x, max_acquisition, 'ro')
        ax2.set_ylabel('Acquisition func.')
        ax2.set_xlabel('$x$')
    
        #Set ylim for acquisition function
        max_acquisition = max(acquisition)+np.abs(max(acquisition))*0.1
        min_acquisition = -max_acquisition
        ax2.set_ylim(min_acquisition, max_acquisition)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    #Create the animation function
    def animation_function(frame, sample_x_collect, observation_y_collect, grid_simple, mean_collect, std_collect, ei_collect, x_max_collect, ei_max_collect):
        plot_anim(sample_x_collect[frame], observation_y_collect[frame], grid_simple, mean_collect[frame], std_collect[frame], 
        acquisition_collect[frame], x_max_collect[frame], acquisition_max_collect[frame], title='Iteration {}'.format(frame))
        
    #Create the animation
    anim_created = FuncAnimation(fig, animation_function, frames=n_iterations, interval=1000, fargs=(sample_x_collect, observation_y_collect, grid_simple, mean_collect, 
                                std_collect, acquisition_collect, x_max_collect, acquisition_max_collect))
    plt.close()

    if html == True:
        video = anim_created.to_jshtml()
        html = HTML(video)
        return html
    else:
        return anim_created
    