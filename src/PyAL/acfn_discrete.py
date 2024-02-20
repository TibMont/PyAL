import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def UCB(f, uncertainty, alpha=0.5):
    ucb = f+alpha*uncertainty
    return ucb

def EI(f, uncertainty, opt, max=True, alpha=0.5):

    if max==False:
        f_min = np.min(f)
        cdf = norm.cdf((f_min-f)/uncertainty)
        pdf = norm.pdf((f_min-f)/uncertainty)
        ei = alpha*(f_min-f)*cdf + (1-alpha)*uncertainty*pdf
    else:
        f_max = np.max(f)
        cdf = norm.cdf((f-f_max)/uncertainty)
        pdf = norm.pdf((f-f_max)/uncertainty)
        ei = alpha*(f-f_max)*cdf + (1-alpha)*uncertainty*pdf
    return ei
    
def POI(f, uncertainty, opt, alpha, max=True):
    if max==False:
        f_min = np.min(f)
        probs = norm.cdf((f_min-f-alpha)/(uncertainty+1e-9))
    else:
        f_max = np.max(f)
        probs = norm.cdf((f-f_max-alpha)/(uncertainty+1e-9))
    return probs

def UIDAL(I_act, grid, uncertainty, alpha=1):
    #print(grid.shape)
    #print(grid)
    #print(uncertainty)

    n_pool = len(grid)
    n_samples = len(I_act)

    w = np.zeros([n_pool, n_samples])
    SiD = np.zeros(n_pool)
    Z = np.zeros(n_pool)
    SW = np.ones(n_pool)
    xs = grid[I_act]

    for i in range(n_samples):
        dist = np.sum((grid-xs[i])**2, axis=-1)

        #Distances between xs[i] and all other data points
        w[0:I_act[i],i] = np.exp(-dist[0:I_act[i]])/dist[0:I_act[i]]
        w[I_act[i],i] = 0
        w[I_act[i]+1:,i] = np.exp(-dist[I_act[i]+1:])/dist[I_act[i]+1:]

        SiD[0:I_act[i]] += 1/dist[0:I_act[i]]
        SiD[I_act[i]+1:] += 1/dist[I_act[i]+1:]

    mask = np.ones(n_pool, dtype=bool)
    mask[I_act] = False
    Z[mask] = np.arctan(1/SiD)[mask]*2/np.pi
    SW[mask] = np.sum(w[mask][:,0:len(grid)], axis=1)

    v = w[:,0:len(grid)]/SW.reshape(-1,1)
    vk = np.sum(v*uncertainty.reshape(-1,1), axis=1)
    ideal = vk+alpha*Z
    return ideal

def IDEAL(I_act, grid, mean, y_true, alpha=1):
    n_pool = len(grid)
    n_samples = len(I_act)

    xmin = np.min(grid, axis=0)
    xmax = np.max(grid, axis=0)

    #print(xmin)
    #print(xmax)

    #print('Grid Old')
    #print(grid)

    grid = 2/(xmax-xmin) * (grid - (xmax+xmin)/2)
    #print('Grid')
    #print(grid)

    w = np.zeros([n_pool, n_samples])
    SiD = np.zeros(n_pool)
    Z = np.zeros(n_pool)
    SW = np.ones(n_pool)
    xs = grid[I_act]
    
    Y_act = y_true[I_act]

    if len(mean.shape) == 1:
        mean = mean.reshape(-1,1)
    Yhat = mean

    Ymax = np.max(Y_act)
    Ymin = np.min(Y_act)
    Ymax += 1.e-8
    Yscale = (Ymax - Ymin) / 2.
    dY2 = (2 * Yscale) ** 2

    #print('Distances')
    for i in range(n_samples):
        dist = np.sum((grid-xs[i])**2, axis=-1)

        w[0:I_act[i],i] = np.exp(-dist[0:I_act[i]])/dist[0:I_act[i]]
        w[I_act[i]+1:,i] = np.exp(-dist[I_act[i]+1:])/dist[I_act[i]+1:]

        SiD[0:I_act[i]] += 1/dist[0:I_act[i]]
        SiD[I_act[i]+1:] += 1/dist[I_act[i]+1:]

    mask = np.ones(n_pool, dtype=bool)
    mask[I_act] = False
    Z = np.arctan(1/SiD)*2/np.pi

    SW = np.sum(w[:,0:len(grid)], axis=1)

    mask_inverse = np.zeros(n_pool, dtype=bool)
    mask_inverse[I_act] = True

    ff = np.zeros(n_pool)
    
    ny = mean.shape[1]
    #print(ny)
    #print(w.shape)
    #print(SW.reshape(-1, 1).shape)
    #print(Yhat[:, np.newaxis].shape)
    #print(Y_act.shape)
    #print((Yhat[:,0,np.newaxis]-Y_act))

    for i in range(ny):
        vk = (w / SW.reshape(-1, 1))
        ff += np.sum( vk * (
            (Yhat[:,i,np.newaxis] - Y_act) ** 2), axis=1) / dY2
        
    ff += alpha*Z

    ff[I_act] = 0
    #print('New')
    #print(w)
    #print(Z)
    #print(SW)

    #if n_pool > 50:
    #    plt.plot(grid, vk)
    #    plt.plot(grid[I_act], y_true[I_act])
    #    plt.show()
        
    return ff


def GSx(I_act, grid):
    n_samples = len(I_act)
    n_pool = len(grid)

    xs = grid[I_act]
    
    min_dist = np.zeros(n_pool)

    distances = np.zeros((n_pool, n_samples))

    if len(grid.shape) == 1:
        for i in range(n_samples):
            dist = (grid-xs[i])**2
            distances[:,i] = dist
    else:
        for i in range(n_samples):
            dist = np.sum((grid-xs[i])**2, axis=-1)
            distances[:,i] = dist

    min_dist = np.min(distances, axis=-1)

    return min_dist

def GSy(mean, ys):
    n_samples = len(ys)
    n_pool = len(mean)

    min_dist = np.zeros(n_pool)

    distances = np.zeros((n_pool, n_samples))

    if len(mean.shape) == 1:
        for i in range(n_samples):
            dist = (mean-ys[i])**2
            distances[:,i] = dist
    else:
        for i in range(n_samples):
            dist = np.sum((mean-ys[i])**2, axis=-1)
            distances[:,i] = dist

    min_dist = np.min(distances, axis=-1)

    return min_dist

def iGS(I_act, grid, mean, ys):
    gsx = GSx(I_act, grid)
    gsy = GSy(mean, ys)

    return gsx*gsy

def SGSx(I_act, grid, std, alpha=0.5):
    gsx = GSx(I_act, grid)

    sgsx = np.power(std,alpha)*np.power(gsx,(1-alpha))

    return sgsx

def QBC(x, models):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)

    n_models = len(models)
    n_pool = len(x)
    
    mean_qbc = np.zeros((n_models, n_pool))
    #bootstrapping approach to train models
    for i in range(n_models):
        m = models[i].predict(x)
        mean_qbc[i] = m

    result = np.zeros(n_pool)
    for i in range(n_pool):
        result[i] = np.sum( mean_qbc[:,i] - np.mean(mean_qbc[:,i]) )

    return result

