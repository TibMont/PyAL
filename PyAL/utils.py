import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

def check_model(regression_model, acquisition_function):
    
    if not callable(acquisition_function):
        
        if isinstance(regression_model, Pipeline):
            reg_model_pure = regression_model['model']
        else:
            reg_model_pure = regression_model

        if isinstance(reg_model_pure, LinearRegression):
            if acquisition_function not in ['random', 'GSx', 'GSy', 'iGS', 'ideal', 'qbc']:
                print('Acquisition functin is not implemented for Linear Regression: {}'.format(acquisition_function))
                exit()
        elif not isinstance(reg_model_pure, GPR):
            if acquisition_function not in ['random', 'GSx', 'GSy', 'iGS', 'ideal', 'qbc']:
                print('Acquisition functin is not implemented for Non-GPR models: {}'.format(acquisition_function))
                exit()
        else:
            if acquisition_function not in ['random', 'GSx', 'GSy', 'iGS', 'ideal', 'qbc', 'ei', 'ucb', 'poi', 'std', 'uidal', 'SGSx']:
                print('Acquisition function not implemented: {}'.format(acquisition_function))
                exit()
    else: 
        print('Using a custom acquisition function. This is an experimental feature. Please be careful.')

    return reg_model_pure

def generate_pool(dimensions, lim, points=20):
    x = []
    for i in range(dimensions):
        x.append(np.linspace(*lim, points))

    pool = np.meshgrid(*x)
    pool = np.array(pool).T
    pool = pool.reshape(len(x[0]**dimensions), dimensions)

    return pool

def fit_model(x, y, regression_model, poly_transformer=None):
    #Check for pipeline, extract the regression model
    if isinstance(regression_model, Pipeline):
        reg_model_pure = regression_model['model']
    else:
        reg_model_pure = regression_model

    if isinstance(reg_model_pure, LinearRegression):
        x_poly = poly_transformer.fit_transform(x, axis=0)
        regression_model.fit(x_poly, y)
    else:
        regression_model.fit(x, y)

    return regression_model

def make_prediction(x, regression_model, poly_transformer=None, fictive_noise_level=0):

    #Check for pipeline, extract the regression model
    if isinstance(regression_model, Pipeline):
        reg_model_pure = regression_model['model']
    else:
        reg_model_pure = regression_model

    #Reshape in case of only 1 sample
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    #Make the prediction
    #We can set a fictive standard deviation, which is by default zero, for non-GPR models
    if isinstance(reg_model_pure, GPR):
        mean_new, std_new = regression_model.predict(x, return_std=True)

    elif isinstance(reg_model_pure, LinearRegression):   

        x_poly = poly_transformer.fit_transform(x, axis=0)
        mean_new = regression_model.predict(x_poly)
        std_new = fictive_noise_level

    else:
        mean_new = regression_model.predict(x)
        std_new = fictive_noise_level

    return mean_new, std_new

def calculate_errors(y_true, mean):
    scores = np.zeros(3)
    scores[0] = mean_squared_error(y_true, mean)
    scores[1] = mean_absolute_error(y_true, mean)
    scores[2] = max_error(y_true, mean)

    return scores

