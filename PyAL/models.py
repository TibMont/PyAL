# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.1
# License: MIT license

import numpy as np
import pandas as pd

from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures

#All model classes within this module should implement a function called evaluate
#The evaluate function should take as arguments only the features (called X) and a noise parameter (noise)

class inv_sphere():
    '''
    N-dimensional inverted sphere function. This is similar to the sphere function multiplied by -1
    so that the function has a global maximum instead of a minimum.
    The inverted sphere function is given as:
    $$
    f(x) = -\sum_{i=1}^{d}x_i^2
    $$
    We can also add some noise $\epsilon$:
    $$
    f(x) = -\sum_{i=1}^{d}x_i^2 + \epsilon
    $$

    Parameters:
    -----------
    d : int
        Number of dimensions.
    random_state : int, optional
        Random state for reproducibility. The default value is 1.

    Attributes:
    -----------
    n_features : int
        Number of features.
    rng : RandomNumberGenerator
        Random Number Generator from numpy.
    '''
    def __init__(self, d=1, random_state=None):
        self.n_features=d
        self.rng = np.random.RandomState(seed=random_state)

    def evaluate(self, X, noise=0):
        '''
        Evaluation function.

        Parameters:
        -----------
        X : nd_array
            Input values (coordinates) for every dimension.
        noise : float, optional
            Additional noise to add to the model.
        
        Returns:
        --------
        The function values for each data point.
        '''

        y = np.sum(np.power(X,2), axis=-1)

        n = self.rng.normal(loc=0, scale=noise, size=y.shape)
        y = y+n

        return -y

class inv_rastrigin():
    '''
    N-dimensional inverted rastrigin function. This is similar to the rastrigin function multiplied by -1
    so that the function has a global maximum instead of a minimum.
    The inverted rastrigin function is given as:
    $$
    f(X) = - h \cdot n - \sum_{i=1}^{n} [x_i^2 - h \cos(2 \pi x_i)]
    $$
    We can also add some noise $\epsilon$:
    $$
    f(X) = - h \cdot n - \sum_{i=1}^{n} [x_i^2 - h \cos(2 \pi x_i)]+ \epsilon
    $$

    Parameters:
    -----------
    d : int
        Number of dimensions.
    random_state : int, optional
        Random state for reproducibility. The default value is 1.

    Attributes:
    -----------
    n_features : int
        Number of features.
    h : float, optional
        Parameter for the Rastrigin function. The default value is 10.
    rng : RandomNumberGenerator
        Random Number Generator from numpy.

    '''
    def __init__(self, d=2, h=10, random_state=None):
        self.n_features=d
        self.h = h
        self.rng = np.random.RandomState(seed=random_state)

    def evaluate(self, X, noise=0):
        '''
        Evaluation function.

        Parameters:
        -----------
        X : nd_array
            Input values (coordinates) for every dimension.
        noise : float, optional
            Additional noise to add to the model.
        
        Returns:
        --------
        The function values for each data point.
        '''

        if self.n_features == 1:
            X = X.reshape(self.n_features,*[int(X.shape[0]**(1/self.n_features)) for _ in range(self.n_features)])
            y = self.h*self.n_features + np.sum([(x**2 -self.h*np.cos(2*np.pi*x)) for x in X], axis=0)
        else:
            y = self.h*self.n_features + np.sum([(x**2 -self.h*np.cos(2*np.pi*x)) for x in X], axis=-1)

        n = self.rng.normal(loc=0, scale=noise, size=y.shape)
        y = y+n
        return -y
    
class inv_rosenbrock():
    '''
    N-dimensional inverted rastrigin function, defined for at least two dimensions. This is similar to the rastrigin function multiplied by -1
    so that the function has a global maximum instead of a minimum.
    The inverted rastrigin function is given as:
    $$
    f(X) = - \sum_{i=1}^{n} [100 (x_{i+1} - x_i^2)^2 + (1-x_i)^2 ]
    $$
    We can also add some noise $\epsilon$:
    $$
    f(X) = - \sum_{i=1}^{n} [100 (x_{i+1} - x_i^2)^2 + (1-x_i)^2 ] + \epsilon
    $$
    
    Parameters:
    -----------
    d : int
        Number of dimensions.
    random_state : int, optional
        Random state for reproducibility. The default value is 1.

    Attributes:
    -----------
    n_features : int
        Number of features.
    rng : RandomNumberGenerator
        Random Number Generator from numpy.

    '''
    def __init__(self, d=2, random_state=None):
        if d<2:
            raise Exception('Rosenbrock not defined for dimension lower than 2.')
        self.n_features=d
        self.rng = np.random.RandomState(seed=random_state)

    def evaluate(self, X, noise=0):
        '''
        Evaluation function.

        Parameters:
        -----------
        X : nd_array
            Input values (coordinates) for every dimension.
        noise : float, optional
            Additional noise to add to the model.
        
        Returns:
        --------
        The function values for each data point.
        '''

        N = self.n_features
        y = 0
        for i in range(N-1):
            y += 100*(X[:,i+1]-X[:,i]**2)**2 + (1-X[:,i])**2
        
        n = self.rng.normal(loc=0, scale=noise, size=y.shape)
        y = y+n

        return -y
    
class inv_alos():
    '''
    1-dimensional inverted ALOS ( Agglomeration of Locally Optimized Surrogate) function, taken from https://doi.org/10.48550/arXiv.2303.01560. 
    This is similar to the ALOS function multiplied by -1
    so that the function has a global maximum instead of a minimum.
    The inverted ALOS function is given as:
    $$
    f(X) = - \sin[ 30 (x-0.9)^4 ] \cos[ 2 (x-0.9) ] + (x-0.9)/2
    $$
    We can also add some noise $\epsilon$:
    $$
    f(X) = - \sin[ 30 (x-0.9)^4 ] \cos[ 2 (x-0.9) ] + (x-0.9)/2 + \epsilon
    $$

    Parameters:
    -----------
    d : int
        Number of dimensions.
    random_state : int, optional
        Random state for reproducibility. The default value is 1.

    Attributes:
    -----------
    n_features : int
        Number of features.
    rng : RandomNumberGenerator
        Random Number Generator from numpy.
    '''
    def __init__(self, d=1, random_state=None):
        self.n_features=d
        self.rng = np.random.RandomState(seed=random_state)

    def evaluate(self, X, noise=0):
        '''
        Evaluation function.

        Parameters:
        -----------
        X : nd_array
            Input values (coordinates) for every dimension.
        noise : float, optional
            Additional noise to add to the model.
        
        Returns:
        --------
        The function values for each data point.
        '''

        if self.n_features == 1:
            X = X.flatten()
            y = np.sin(30*(X-0.9)**4)*np.cos(2*(X-0.9))+(X-0.9)/2

        n = self.rng.normal(loc=0, scale=noise, size=y.shape)
        y = y+n

        return -y



class PrefitModel():
    """
    Use a prefit sklearn model as true model. 

    Parameters:
    -----------
    model : sklearn-model
        A prefitted sklearn-model which is used as a model for the true data.
    n_features : int
        Number of features.
    scaler : sklearn Scaler, optional
        Scaler that is used to preprocess data for the model. A prefitted scaler should be used. 
        The default value is ``None``.
    random_state : int, optional 
        Random state for reproducibility. The default value is 1.


    Attributes:
    -----------
     model : sklearn-model
        A prefitted sklearn-model which is used as a model for the true data.
    scaler : sklearn Scaler, optional
        Scaler that is used to preprocess data for the model. A prefitted scaler should be used. 
        The default value is ``None``.
    n_features : int
        Number of features.
    rng : RandomNumberGenerator
        Random Number Generator from numpy.


    """
    def __init__(self, model, n_features, scaler=None, random_state=None):
        self.model = model
        self.scaler = scaler
        self.rng = np.random.RandomState(seed=random_state)
        self.n_features = n_features

    def evaluate(self, grid, noise=0):
        '''
        Evaluation function.

        Parameters:
        -----------
        grid : nd_array
            Input values (coordinates) for every dimension.
        noise : float, optional
            Additional noise to add to the model.
        
        Returns:
        --------
        The function values for each data point.
        '''
        if self.scaler != None:
            grid = self.scaler.transform(grid)
        y = self.model.predict(grid)

        n = self.rng.normal(loc=0, scale=noise, size=len(y))
        y = y + n
        return y


class PolyModel():
    """
    Polynomial Model.
    """
    def __init__(self, weights, features='xy', polynomial_degree=2, scaler=None, random_state=None):
        self.n_features=len(features)
        self.features=features
        self.polynomial_degree=polynomial_degree
        self.weights=self._create_feature_df(weights, features, polynomial_degree)
        self.rng = np.random.RandomState(seed=random_state)
        self.scaler = scaler
        #print('The follwing weights have been assigned:')
        #display(self.weights)

    def _create_feature_df(self, coeffs, features, polynomial_degree):
        final_combinations = []
        for i in range(polynomial_degree+1):
            combi = list(combinations_with_replacement(features, i))
            for c in combi:
                c_str = ''
                for cc in c:
                    c_str = c_str+cc
                if c_str == '':
                    c_str='const'
                final_combinations.append(c_str)
        coeffs = pd.DataFrame([coeffs], columns=final_combinations)
        return coeffs

    def _internal_evaluation(self, grid):
        
        poly =  PolynomialFeatures(degree=self.polynomial_degree)  
        poly_features = poly.fit_transform(grid.transpose())
        if self.scaler != None:
            poly_features = self.scaler.transform(poly_features)
        y = np.sum(poly_features*np.array(self.weights.iloc[0]), axis=1)+self.weights.iloc[0]['const']
        return y

    def evaluate_on_grid(self, n_data, lim=(-2,2)):
        x = [np.linspace(*lim,n_data) for i in range(self.n_features)]
        grid = np.array(np.meshgrid(*x))
        grid = grid.reshape(self.n_features, n_data**self.n_features)
        #y = np.zeros(n_data**self.n_features)

        y = self._internal_evaluation(grid)
        
        return grid.transpose(), y

    def evaluate(self, grid, noise=0):
        grid = np.array(grid).transpose()
        #n_data = len(grid)
        #y = np.zeros(n_data)
        y = self._internal_evaluation(grid)
        n = self.rng.normal(loc=0, scale=noise, size=len(y))
        y = y + n
        return y


class ArrheniusModel():
    """
    An Arrhenius type model with the form:

    .. math:: S = S_0 - S_1 (\\beta-\\beta_0) - S_2 (\\beta-\\beta_0)^2
  
    where :math:`S_0` corresponds to the value of :math:`S` at the onset temperature 
    :math:`T_0`, :math:`S_1` is an activation energy and :math:`S_2` corresponds to deviations from Arrhenius behaviour. 
    :math:`\\beta` is the inverse temperature. A model for each of the :math:`S_i` is needed. 
    """
    def __init__(self, S0, S1, S2, temperature=(1000/293.15), beta_0 = (1000/333.15), random_state=None):
        self.model_S0 = S0
        self.model_S1 = S1
        self.model_S2 = S2
        self.n_features = S0.n_features
        self.rng = np.random.RandomState(seed=random_state)
        self.temperature = temperature
        self.beta_0 = beta_0
        self.delta_beta = temperature-beta_0
        self.weights = {'S0': S0.weights, 'S1': S1.weights, 'S2': S2.weights}

    def evaluate(self, grid, noise):

        S0_res = self.model_S0.evaluate(grid, noise)
        S1_res = self.model_S0.evaluate(grid, noise)
        S2_res = self.model_S0.evaluate(grid, noise)

        sigma = S0_res - self.delta_beta*S1_res - self.delta_beta**2*S2_res
        return sigma
    

class PoolModel():
    """
    A model where data points are taken directly from a pool of already known data.
    """
    def __init__(self, features, objective):
        self.n_features=np.array(features).shape[1]
        self.features=np.array(features)
        self.objective = np.array(objective).flatten()

    def evaluate(self, grid, **kwargs):
        grid = np.array(grid)
        if len(grid.shape)==1:
            grid = grid.reshape(1,-1)
        idx = []
        for i in range(len(grid)):
            index = np.where(np.sum(self.features, axis=1)==np.sum(grid[i]))[0][0]
            idx.append(index)

        idx = np.array(idx)
        return self.objective[idx]
            