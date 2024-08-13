# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

"""Acquisition function implementation that can be used with scipy.minimize and pyswarms. 
Since both algorithms search for the minimum the negative acquisition function value is returned."""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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

def GSy_con(x, y_sample, model, poly_x=None):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)

    if isinstance(poly_x, PolynomialFeatures):
        x_poly = poly_x.fit_transform(x)
        y = model.predict(x_poly)
    else:
        y = model.predict(x)

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

def iGS_con(x, x_sample, y_sample, model, poly_x = None):
    gsx = GSx_con(x,x_sample)
    gsy = GSy_con(x, y_sample, model, poly_x)

    min_dist = gsx*gsy
    return -min_dist

def SGSx_con(x, x_sample, model, alpha=0.5):
    gsx = GSx_con(x,x_sample)
    _, std = model.predict(x, return_std=True)
    sgsx = np.power(std,alpha)*np.power(-gsx,(1-alpha))
    return -sgsx

    
def UIDAL_con(x, x_samples, model, alpha=1):
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
    _, uncertainty = model.predict(x, return_std=True)
    vk = np.sum(v*uncertainty.reshape(-1,1), axis=1)
    ideal = vk+alpha*Z
    o = np.isnan(ideal)
    ideal[o] = 0
    return -ideal

def UCB_con(x, model, alpha=0.5):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    mean, uncertainty = model.predict(x, return_std=True)
    ucb = mean+alpha*uncertainty
    return -ucb

def POI_con(x, model, opt, alpha, max=True):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    mean, uncertainty = model.predict(x, return_std=True)

    if max==False:
        f_min = opt
        probs = norm.cdf((f_min-mean-alpha)/(uncertainty+1e-9))
    else:
        f_max = opt
        probs = norm.cdf((mean-f_max-alpha)/(uncertainty+1e-9))
    return -probs

def EI_con(x, model, opt, alpha=0.5, max=True,):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    mean, uncertainty = model.predict(x, return_std=True)

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


def QBC_con_old(x, models):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    mean_qbc = 0
    std_qbc = 0
    #bootstrapping approach to train models
    for model in models:
        m, s = model.predict(x,return_std=True)
        mean_qbc += m
        std_qbc += s

    mean_qbc /= len(models)
    std_qbc /= len(models)

    return -std_qbc

def QBC_con(x, models, poly_x = None):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)

    n_models = len(models)
    n_pool = len(x)

    #print(n_models)
    #print(n_pool)
    
    mean_qbc = np.zeros((n_models, n_pool))
    #bootstrapping approach to train models
    for i in range(n_models):

        if isinstance(poly_x, PolynomialFeatures):
            x_poly = poly_x.fit_transform(x)
            m = models[i].predict(x_poly)
        else:
            m = models[i].predict(x)
        mean_qbc[i] = m
    
    #print('Mean qbc')
    #print(mean_qbc)

    result = np.zeros(n_pool)
    for i in range(n_pool):
        #print('indi')
        #print(mean_qbc[:,i])
        #print('mean')
        #print(np.mean(mean_qbc[:,i]))
        result[i] = np.sum( mean_qbc[:,i] - np.mean(mean_qbc[:,i]) )

    return -result

def std_con(x, model):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    _, std = model.predict(x, return_std=True)

    return -std


def IDEAL_con(x, x_samples, model, y_true, lim, alpha=1,  poly_x=None, tol=1e-8, plot=False):

    if len(x.shape) == 1:
        n_pool = 1
        x = x.reshape(1,-1)
    else:
        n_pool = len(x)
    n_samples = len(x_samples)

    #print('Previous x')
    #print(x)

    #print('Previous sample')
    #print(x_samples)
    
    xmin = np.array(lim[0])
    xmax = np.array(lim[1])
    x_scaled = 2/(xmax-xmin) * (x - (xmax+xmin)/2)
    x_samples = 2/(xmax-xmin) * (x_samples - (xmax+xmin)/2)

    #Try to concatenate both

    #print('Scaled x')
    #print(x)

    #print('Scaled sample')
    #print(x_samples)


    

    #print(x_samples.shape)
    #print(x.shape)

    w = np.zeros([n_pool, n_samples])
    SiD = np.zeros(n_pool)
    Z = np.zeros(n_pool)
    SW = np.ones(n_pool)

    mask_idx = []
    identical = []

    for i in range(n_samples):
        dist = np.sum((x_scaled-x_samples[i])**2, axis=-1)
        #print(dist)

        mask = np.where(dist>tol)
        mask_rev = np.where(dist<=tol)[0]
        if len(mask_rev) > 0:
            for m in mask_rev:
                mask_idx.append(m)
            identical.append(i)

        #w[mask,i] = np.exp(-dist[mask])/dist[mask]
        #w[mask_rev] = 0
        #SiD[mask] += 1/dist[mask]

        w[:,i] = np.exp(-dist)/dist
        SiD+= 1/dist

    Z = np.arctan(1/SiD)*2/np.pi
    SW = np.sum(w[:,0:n_samples], axis=1)

    #print(Z.shape)
    #print(SW.shape)

    mask_idx = np.array(mask_idx, dtype=np.int32)
    identical = np.array(identical, dtype=np.int32)
    if len(mask_idx)>0:
        #print('Idx')
        #print(mask_idx)
        Z[mask_idx] = 0

    if isinstance(poly_x, PolynomialFeatures):
        x_poly = poly_x.fit_transform(x)
        mean = model.predict(x_poly)
    else:
        mean = model.predict(x)
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
        #print(mask_idx)
        #print(identical)
        #print(vk[mask_idx, identical])
        vk[mask_idx] = 0

        #print('Shape')
        #print((Yhat[:,i,np.newaxis] - y_true).shape)

        ff += np.sum((vk) * (
            (Yhat[:,i,np.newaxis] - y_true) ** 2), axis=1) / dY2
    
    ideal = ff+alpha*Z
    #o = np.isnan(ideal)
    #ideal[o] = 0
    #print(vk[:,0])
    #if plot==True and n_pool > 30:
    #    print(vk[:,0])
    #    plt.plot(x, vk)
     #   plt.plot(x_samples, y_true)
    #    plt.show()

    return -ideal