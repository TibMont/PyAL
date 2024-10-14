# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

"""Animation of active learning for simple 1D functions. Can be used for demonstration purposes."""

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
try:
    from IPython.display import HTML
except:
    print('IPython not found.')

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import scale
from scipy.stats import norm

from PyAL.acfn_discrete import QBC
from PyAL.optimize_step import step_discrete, step_continuous

def max_acquisition(acquisition, grid, rng=None):
    """Find the maximum of a function on a discrete grid

    Parameters
    ----------
    acquisition : nd_array
        Array that containes the values of the acquisition function 
        on given grid points.
    grid : nd_array
        Array that contains the grid points.
    rng : Numpy Random Number Generator, optional
        Numpy Random Number Generator object. The default is "None". 

    Returns
    -------
    nd_array
        Grid point for which the acquisition function has its maximum.
    float
        Maximum value of the acquisition function.
    int 
        Index of the grid point for which the acquisition function has its maximum.
    """
    if rng is None:
        rng = np.random.RandomState(seed=None)
    max_value = np.max(acquisition)
    max_index = np.where(acquisition==max_value)[0]
    if len(max_index) > 1:
        ind = rng.randint(0,len(max_index),1)
        max_index = max_index[ind]
    max_x = grid[max_index]
    return max_x, max_value, max_index

def create_animation(
    model,
    gpr,
    acquisition_function,
    grid_simple,
    n_iterations = 30,
    alpha = 0,
    n_observations = 3,
    noise_level = 0,
    html = True,
    random_state = 42,
    legend = True,
    plot_std = True,
    custom_acfn_input = None
):
    """Create an animation for Active Learning on a 1-dimenional function.
    The Active Learning is only done on the discrete grid, which must be provided. 
    This is useful for illustration purposes. 

    Parameters
    ----------
    model : Model class
        PyAL model class that generates the (noisy) data.
    gpr : Sciit-Learn Model
        Scikit-Learn Model, use GaussianProcessRegression model here, 
        since it supports all acquisition functions.
    acquisition_function : str or callable
        Name of the acquisition function or a callable. Choose from: ei, poi, 
        ucb, random, std, ideal, uidal, GSx, GSy, iGS, sGSx. 
        QBC must implemented separately.
    grid_simple : nd_array
        A simple grid in one dimension.
    n_iterations : int, optional
        Number of Active Learning steps, by default 30
    alpha : int, optional
        Hyperparameter of the given acquisition function, by default 0
    n_observations : int, optional
        Number of initial observations, by default 3
    noise_level : int, optional
        Standard deviation of Gaussian noise, by default 0
    html : bool, optional
        Whether to convert the animation to html format, by default True
    random_state : int, optional
        Random state for reproducibility, by default 42
    legend : bool, optional
        Whether to show a legend in the animation, by default True.
    plot_std : bool, optional.
        Whether to plot the standard deviation, by default True.
    custom_acfn_input : dict
        Dictionary that contains which information is used by a custom 
        acquisition function. The default is None.

    Returns
    -------
    matplotlib animation
        Animation of the Active Learning process.
    """
    rng = np.random.RandomState(seed=random_state)

    if custom_acfn_input is None:
        custom_acfn_input = {}

    data_indices = rng.randint(0, len(grid_simple),n_observations)
    sample_x = grid_simple[data_indices]
    observation_y = model.evaluate(sample_x, noise=noise_level)

    y_true = model.evaluate(grid_simple, noise=0)

    #We save every timestep in the collect variables
    sample_x_collect = []
    observation_y_collect = []
    mean_collect = []
    std_collect = []
    acquisition_collect = []
    x_max_collect = []
    acquisition_max_collect = []

    #Run the Active Learning for n_iterations
    for _ in range(n_iterations):
        #Make model predictions
        gpr.fit(sample_x, observation_y)
        mean, std = gpr.predict(grid_simple,return_std=True)

        n_data = len(grid_simple)

        if acquisition_function == 'qbc':
            models = []
            for _ in range(alpha):
                train_index = rng.randint(0,len(sample_x),len(sample_x))
                gpr.fit(sample_x[train_index], observation_y[train_index])
                models.append(copy.deepcopy(gpr))
            acquisition = QBC(grid_simple, models)
        else:
            acquisition = step_discrete(
                        acquisition_function,
                        observation_y,
                        alpha, mean, std,
                        n_data, data_indices, grid_simple,
                        custom_acfn_input,
                        rng,
            )

        x_max, acquisition_max, max_index = max_acquisition(acquisition, grid_simple)

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


    def plot_anim(sample_x, observation_y, grid, mean, std, acquisition, max_x,
                  max_value, title=None, legend=True):
        n_data = grid.shape[0]

        #Clear the plot
        ax1.clear()
        ax2.clear()

        #Set title
        if title is not None:
            ax1.set_title(title)

        #Plot observations, true model and predicted model
        ax1.plot(grid.reshape(n_data), mean, '-', label='Prediction')
        ax1.plot(grid_simple.reshape(n_data),y_true, 'k-', label='True')
        ax1.plot(sample_x, observation_y,'ko', markerfacecolor='white', label='Observation')
        ax1.set_ylabel('$f(x)$')
        if legend:
            ax1.legend(loc='upper right')
        if plot_std:
            ax1.fill_between(grid.reshape(n_data),mean-std*3,mean+std*3, alpha=0.2)

        #Plot acquisition function
        ax2.plot(grid.reshape(n_data), acquisition)
        ax2.plot(max_x, max_value, 'ro')
        ax2.set_ylabel('Acquisition func.')
        ax2.set_xlabel('$x$')

        max_acquisition_ = max(max(acquisition),-1*max_value)*1.1
        min_acquisition_ = -0.1*max_acquisition_

        ax2.set_ylim(min_acquisition_,max_acquisition_)

        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    #Create the animation function
    def animation_function(frame, sample_x_collect, observation_y_collect, grid_simple,
                           mean_collect, std_collect, acquisition_collect, x_max_collect,
                           acquisition_max_collect, legend=True):
        plot_anim(sample_x_collect[frame], observation_y_collect[frame], grid_simple,
                  mean_collect[frame], std_collect[frame], acquisition_collect[frame],
                  x_max_collect[frame], acquisition_max_collect[frame],
                  title='Iteration {}'.format(frame), legend=legend)

    #Create the animation
    anim_created = FuncAnimation(fig, animation_function, frames=n_iterations, interval=1000,
                                 fargs=(sample_x_collect, observation_y_collect, grid_simple,
                                        mean_collect, std_collect, acquisition_collect,
                                        x_max_collect, acquisition_max_collect, legend))
    plt.close()
    if html:
        video = anim_created.to_jshtml()
        html = HTML(video)
        return html
    else:
        return anim_created


def create_animation_continuous(
    model,
    gpr,
    acquisition_function,
    grid_simple,
    n_iterations = 30,
    alpha = 0,
    n_observations = 3,
    noise_level = 0,
    html = True,
    random_state = 42,
    opt_method = 'lbfgs',
    pso_options = None,
    legend = False,
    plot_std = False,
    custom_acfn_input = None,

    ):
    """Create an animation for Active Learning on a 1-dimenional function. 
    The Active Learning is done in continuous space using either LBFGS or PSO optimization.
    A grid must be given do define the boundaries and for plotting.
    This is useful for illustration purposes. 

    Parameters
    ----------
    model : Model class
        PyAL model class that generates the (noisy) data.
    gpr : Sciit-Learn Model
        Scikit-Learn Model, use GaussianProcessRegression model here, 
        since it supports all acquisition functions.
    acquisition_function : str or callable
        Name of the acquisition function or a callable. Choose from: ei, poi, 
        ucb, random, std, ideal, uidal, GSx, GSy, iGS, sGSx. 
        QBC must implemented separately.
    grid_simple : nd_array
        A simple grid in one dimension.
    n_iterations : int, optional
        Number of Active Learning steps, by default 30
    alpha : int, optional
        Hyperparameter of the given acquisition function, by default 0
    n_observations : int, optional
        Number of initial observations, by default 3
    noise_level : int, optional
        Standard deviation of Gaussian noise, by default 0
    html : bool, optional
        Whether to convert the animation to html format, by default True
    random_state : int, optional
        Random state for reproducibility, by default 42
    opt_method : str, optional
        Choose from scipy (using lbfgs algorithm) or PSO (using PySwarms). 
        The default value is "lbfgs".
    pso_options : dict, optional
        Options for PySwarms GlobalBestOptimizer. The default value is "{}".
    legend : bool, optional
        Whether to show a legend in the animation, by default True
    plot_std : bool, optional.
        Whether to plot the standard deviation, by default True.
    custom_acfn_input : dict
        Dictionary that contains which information is used by a custom 
        acquisition function. The default is {}.

    Returns
    -------
    matplotlib animation
        Animation of the Active Learning process.
    """
    rng = np.random.RandomState(seed=random_state)

    if custom_acfn_input is None:
        custom_acfn_input = {}

    dimensions = model.n_features

    lim = [min(grid_simple), max(grid_simple)]

    sampler = LHS(d=dimensions, seed=random_state)
    sample_x_unscaled = sampler.random(n_observations)
    sample_x = scale(sample_x_unscaled, *lim)
    observation_y = model.evaluate(sample_x, noise=noise_level)

    #Append random samples to grid
    n_data = len(grid_simple)
    grid_simple = np.concatenate([grid_simple, sample_x])
    grid_simple = np.sort(grid_simple, axis=0)
    data_indices = []
    for v in sample_x:
        data_indices.append(np.where(grid_simple==v)[0][0])
    data_indices = np.array(data_indices)

    y_true = model.evaluate(grid_simple, noise=0)

    #We save every timestep in the collect variables
    sample_x_collect = []
    observation_y_collect = []
    mean_collect = []
    std_collect = []
    acquisition_collect = []
    x_max_collect = []
    acquisition_max_collect = []
    grid_collect = []
    y_true_collect = []

    #Run the Active Learning for n_iterations
    for _ in range(n_iterations):
        #Make model predictions
        gpr.fit(sample_x, observation_y)
        mean, std = gpr.predict(grid_simple,return_std=True)

        sample_x_poly = None #Needed for LR, which is not allowed here
        poly_x = None #Needed for LR,  which is not allowed here
        n_jobs = 1

        new_x, new_y = step_continuous(acquisition_function,
                    opt_method,
                    gpr,
                    observation_y,
                    sample_x,
                    sample_x_poly,
                    custom_acfn_input,
                    alpha,
                    sampler,
                    lim,
                    dimensions,
                    poly_x,
                    n_jobs,
                    pso_options,
                    rng)

        if acquisition_function == 'qbc':
            models = []
            for _ in range(alpha):
                train_index = rng.randint(0,len(sample_x),len(sample_x))
                gpr.fit(sample_x[train_index], observation_y[train_index])
                models.append(copy.deepcopy(gpr))
            acquisition = QBC(grid_simple, models)
        else:
            acquisition = step_discrete(
                        acquisition_function,
                        observation_y,
                        alpha, mean, std,
                        n_data, data_indices, grid_simple,
                        custom_acfn_input,
                        rng,
            )

        observation_new = model.evaluate(new_x.reshape(-1,1), noise=noise_level)

        #Save each active learning step
        sample_x_collect.append(sample_x.copy())
        observation_y_collect.append(observation_y.copy())
        mean_collect.append(mean.copy())
        std_collect.append(std.copy())

        acquisition_collect.append(acquisition.copy())
        x_max_collect.append(new_x.copy())
        acquisition_max_collect.append(new_y)
        grid_collect.append(grid_simple.copy())
        y_true_collect.append(y_true.copy())

        #Update the sample and observations with new data
        sample_x = np.vstack([sample_x, new_x])
        observation_y = np.hstack([observation_y, observation_new])

        #Append new samples to grid
        n_data = len(grid_simple)
        grid_simple = np.concatenate([grid_simple, new_x.reshape(-1,1)])
        grid_simple = np.sort(grid_simple, axis=0)
        data_indices = []
        for v in sample_x:
            data_indices.append(np.where(grid_simple==v)[0][0])
        data_indices = np.array(data_indices)

        y_true = model.evaluate(grid_simple, noise=0)

    #Create a figure
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios':[2,1]}, sharex=True)
    fig.set_tight_layout(True)

    def plot_anim(sample_x, observation_y, grid, mean, std, acquisition, max_x,
                  max_value, y_true, title=None):
        n_data_plot = grid.shape[0]

        #Clear the plot
        ax1.clear()
        ax2.clear()

        #Set title
        if title is not None:
            ax1.set_title(title)

        #Plot observations, true model and predicted model
        ax1.plot(grid.reshape(n_data_plot), mean, '-', label='Prediction')
        ax1.plot(grid.reshape(n_data_plot),y_true, 'k-', label='True')
        ax1.plot(sample_x, observation_y,'ko', markerfacecolor='white', label='Observation')
        ax1.set_ylabel('$f(x)$')
        if legend:
            ax1.legend(loc='upper right')
        if plot_std:
            ax1.fill_between(grid.reshape(n_data_plot),mean-std*3,mean+std*3, alpha=0.2)

        #Plot acquisition function
        ax2.plot(grid.reshape(n_data_plot), acquisition)
        ax2.plot(max_x, -max_value, 'ro')
        ax2.set_ylabel('Acquisition func.')
        ax2.set_xlabel('$x$')

        max_acquisition_ = max(max(acquisition),-1*max_value)*1.1
        min_acquisition_ = -0.1*max_acquisition_

        ax2.set_ylim(min_acquisition_,max_acquisition_)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    #Create the animation function
    def animation_function(frame, sample_x_collect, observation_y_collect, grid_collect,
                           mean_collect, std_collect, acquisition_collect, x_max_collect,
                           acquisition_max_collect, y_true_collect):
        plot_anim(sample_x_collect[frame], observation_y_collect[frame], grid_collect[frame],
                  mean_collect[frame], std_collect[frame], acquisition_collect[frame],
                  x_max_collect[frame], acquisition_max_collect[frame], y_true_collect[frame],
                  title='Iteration {}'.format(frame))

    #Create the animation
    anim_created = FuncAnimation(fig, animation_function, frames=n_iterations, interval=1000,
                                 fargs=(sample_x_collect, observation_y_collect, grid_collect,
                                        mean_collect, std_collect, acquisition_collect,
                                        x_max_collect, acquisition_max_collect, y_true_collect))
    plt.close()

    if html:
        video = anim_created.to_jshtml()
        html = HTML(video)
        return html
    else:
        return anim_created
