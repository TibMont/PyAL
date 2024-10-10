# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import PolynomialFeatures

import sys
import os
import warnings

import logging

logger = logging.getLogger('basic_logger')

if not sys.warnoptions:
    print('Disabled warnings')
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')

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
        cdf = norm.cdf((f_min-mean-alpha)/uncertainty)
        pdf = norm.pdf((f_min-mean-alpha)/uncertainty)
        ei = (f_min-mean-alpha)*cdf + uncertainty*pdf
    else:
        f_max = opt
        cdf = norm.cdf((mean-f_max-alpha)/uncertainty)
        pdf = norm.pdf((mean-f_max-alpha)/uncertainty)
        ei = (mean-f_max-alpha)*cdf + uncertainty*pdf
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