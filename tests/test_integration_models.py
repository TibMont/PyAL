import numpy as np

import pytest
from PyAL.models import inv_sphere
from PyAL.optimize import run_continuous_batch_learning
from PyAL.multi_optimize import run_continuous_batch_learning_multi
from PyAL.multi_optimize_pool import run_batch_learning_multi
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

def identity(x, **kwargs):
    return x[0,:]

def test_continuous_batch_learninig_pso():
       random_state = 10
       model_sphere = inv_sphere(d=1, random_state=random_state)
       grid = np.linspace(-5,5,100).reshape(-1,1)

       kernel = RBF()+WhiteKernel()
       reg_model = GPR(kernel)

       samples, _, _ = run_continuous_batch_learning(model_sphere, 
              reg_model,
              acquisition_function = 'ideal',
              opt_method = 'PSO',
              pool = None, 
              batch_size = 1,
              noise=0.1,
              initial_samples=2, 
              active_learning_steps=1,
              lim=[-1,1],
              alpha=0,
              n_jobs=1,
              random_state=random_state,
              initialization='random',
              pso_options = None,
              fictive_noise_level=0,
              poly_degree = 3,
              calculate_test_metrics = True,
              custom_acfn_input = None,
              )

       assert np.round(samples[-1][0], 8) == np.round(-0.06780794497104031, 8)


def test_continuous_batch_learninig_lbfgs():
       random_state = 10
       model_sphere = inv_sphere(d=1, random_state=random_state)
       grid = np.linspace(-5,5,100).reshape(-1,1)

       kernel = RBF()+WhiteKernel()
       reg_model = GPR(kernel)

       samples, _, _ = run_continuous_batch_learning(model_sphere,
              reg_model,
              acquisition_function = 'ideal',
              opt_method = 'scipy',
              pool = None,
              batch_size = 1,
              noise=0.1,
              initial_samples=2,
              active_learning_steps=1,
              lim=[[-1],[1]],
              alpha=0,
              n_jobs=1,
              random_state=random_state,
              initialization='random',
              pso_options = None,
              fictive_noise_level=0,
              poly_degree = 3,
              calculate_test_metrics = True,
              custom_acfn_input = None,
              )

       assert np.round(samples[-1][0], 8) == np.round(-0.06780679378288344, 8)

kernel = RBF()+WhiteKernel()
model = GPR(kernel)
pipe = Pipeline([('model', model)])
@pytest.mark.parametrize("reg_model, calculate_test_metrics, initialization, expected_result, single_update", [
    (model, True, 'random', -0.06780794497104031, True),
    (model, True, 'GSx', 0.18115688540715172, True),
    (model, False, 'GSx',0.19118684612428058, True),
    (pipe, True, 'random', -0.06780794497104031, False)
    ])

def test_continuous_batch_learning_multi_pso(reg_model, calculate_test_metrics, initialization, expected_result, single_update):
       
       random_state = 10
       model_sphere = inv_sphere(d=1, random_state=random_state)

       res = run_continuous_batch_learning_multi([model_sphere], 
              aggregation_function = identity,
              regression_models=[reg_model],
              acquisition_function = 'ideal',
              opt_method = 'PSO',
              pool = None, 
              batch_size = 1,
              noise=0.1,
              initial_samples=2, 
              active_learning_steps=1,
              lim=[[-1],[1]],
              alpha=[0],
              n_jobs=1,
              random_state=random_state,
              initialization=initialization,
              pso_options = None,
              fictive_noise_level=0,
              poly_degree = 3,
              calculate_test_metrics = calculate_test_metrics,
              custom_acfn_input = None,
              single_update=single_update
              )
       samples = res[0]
       assert np.round(samples[-1][0],8) == np.round(expected_result, 8)

def test_run_batch_learning_multi():
       random_state = 10
       model_sphere = inv_sphere(d=2, random_state=random_state)
       #grid = np.linspace(-5,5,100).reshape(-1,1)

       kernel = RBF()+WhiteKernel()
       reg_model = GPR(kernel)

       samples, _, _ = run_batch_learning_multi([model_sphere],
              aggregation_function=identity, 
              regression_models=[reg_model],
              acquisition_function = 'ideal',
              pool = None, 
              batch_size = 1,
              noise=0.1,
              initial_samples=4, 
              active_learning_steps=1,
              lim=[-1,1],
              alpha=[0],
              random_state=random_state,
              return_samples=False,
              initialization='random',
              test_set = None,
              poly_degree = 3,
              custom_acfn_input={},
              fictive_noise_level = 0,
              calculate_test_metrics = True,
              single_update=False,
              verbose=False)
       samples[-1]
       assert np.array_equal(np.round(samples[-1], 8), np.array([0.78947368, -1]))
