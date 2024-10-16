from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

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
