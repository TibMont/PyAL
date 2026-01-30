Package Description
===================

Data models
-----------

PyALAF uses so-called data models to provide true data. The data models can either store already labeled data (useful for pool-based sampling) or can label 
data on-the-fly (useful in population-based sampling). When using the on-the-fly generation, also other ML models can be used to provide the true data 
(for example as a toy model). This is mainly useful when comparing and bench marking different acquisition functions for known data sets.

In real AL applications, the data model needs to ask the user to label the data either manually or do the labeling automatically. Some data-models are 
predefined, however, it is also easy to define own data models to allow for large flexibility. This can also enable communication with experiments. 

As predefined data models, the PrefitModel (which uses a scikit-learn model to provide the 'True' objectives values on-the-fly) and the PoolModel 
(which needs a predifined pool of labeled data) are available. 
The data model is a very simple Python class which needs to implement only one function named evaluate. This function takes data points x as input and 
returns the corresponding values of the objective.

Available regression models and model training
----------------------------------------------

Most of scikit-learns models can be used within PyALAF. However, some acquisition functions are limited to Gaussian Process Regression (GPR) models 
because they require a prediction uncertainty, which is only provided by GPR natively. A model can be either defined for all objectives or for each 
objective individually.  

Initialization of regression models
-----------------------------------

In PyALAF the initial data points can be directly generated on-the-fly when starting the AL using either Latin Hypercube Sampling or the GSx method 
(with the first data point selected randomly). It is also possible to provide a pool of initial data points. 
By default the features will not be scaled. However, it is possible to use any of the scikit-learn scaler functionalities. In any case, it is important to 
also select the boundaries for each feature properly, since the optimization algorithm will search for the maximum acquisition function only within given 
boundaries. Thus, the MinMax-Scaler is mostly a good option if features need to be scaled since it ensures fixed boundaries by definition.

Calculation of test scores
--------------------------

In PyALAF it is possible to run the Active Learning routine either with or without calculating error metrics for a test set. If error metrics for a test 
set are calculated the user needs to provide a test set or the test set is selected automatically (a grid of data points for population-based learning or 
all unlabeled data points in pool-based learning). This is mainly useful for testing and comparing different acquisition functions, whereas it is not suitable 
for a real AL application.

Aggregation functions
---------------------

PyALAF allows to perform active learning for multiple objectives with the same data in parallel. The user can simply provide a list of different objectives. 
However, often one does not like to obtain different suggestions of data points for different objectives. Therefore, it is possible to combine different 
objectives into a single new objective which is then used to calculate the acquisition function (see below). However, the acquisition function should not act 
on the predictions of these models itself, 
but instead on the predicted ionic conductivity at a given temperature. An aggregation function simply combines the predictions of the individual 
objectives to the new objectives and can also take additional input, like the temperature, into account. In the context of python programming the aggregation 
function is just a callable python function. Aggregation functions can be defined by the user.

Available acquisition functions
-------------------------------

The GSx, GSy, iGS and IDEAL acquisition functions are directly implemented and available for all scikit-learn regression models. Furthermore, the 
Expected Improvement, Probability of Improvement and Upper Confidence Bound as well as the standard deviation are implemented as acquisition functions 
for Bayesian Optimization using Gaussian Process Regression models. Own acquisition functions can also be implemented. The requested input for the 
custom acquisition functions must be provided. Any of these can be used as input in the given order: Features x, predictions y, maximum found value of y, 
regression models, boundaries and an acquisition function hyper-parameter alpha.

Calculation of the acquisition function
---------------------------------------

The acquisition function is either calculated for the full pool of available data points (pool-based sampling) or for selected data points when searching 
for the optimum in the population-based approach. For both available options, there is a different implementation of the acquisition functions. The simple 
reason behind this is that when used within local or global optimization algorithms the acquisition functions need a slightly different structure. 

Finding the maximum of the acquisition function
-----------------------------------------------

In the pool-base approach, the acquisition function is computed for all data points of the pool. The data points with the highest acquisition function are 
then added to the training set. 
When using a population-based AL approach, it is not possible to calculate the acquisition function for every data point like in the pool-based approach. 
Instead, the maximum of the acquisition function needs to be determined using global optimization algorithms, in particular Particle Swarm Optimization (PSO) 
was implemented. 

In principle it is also possible to use local optimization algorithms like L-BFGS (which is also implemented). Although using L-BFGS for any acquisition 
function improves over random sampling, it performs worse compared to using PSO because in most cases the acquisition function is not smooth and has many 
local optima.

Even if PSO does not find the global optimum, it almost certainly identifies a better optimum than L-BFGS. It is also not important in the HTE setting to 
identify the optimum with high precision (which is very likely with our default values of 10 particles per dimension and only 200 iterations). There is 
always an uncertainty when formulating the electrolytes, which is given by the precision of the used instruments, like pipettes. Thus the maximum number of 
iterations can be reduced to speed up the Active Learning.

Batch-wise sampling
-------------------

PyALAF suggests only one data point at a time for addition to the training set. However, it uses a simple trick to also support batch-model learning. If 
batches of a size larger than 1 are required, PyALAF stores the newly suggested data point first separately and handles the prediction together with its 
uncertainty (if available) as if it was the true observation. Based on this suggestion a new model can be trained to make a second suggestion for a new data 
point and so on. If the number of suggested data points reaches the required batch size, the true values for the objectives of all data points within this 
batch are obtained and the predictions, which previously were only assumed to be true, are replaced by them. 

Stopping of the Active Learning
-------------------------------

PyALAF continues the addition of new batches of data to the training set until the maximum number of batches/active learning steps is reached. It is also 
possible to set the single update keyword to True. Then the AL is stopped after the first batch is completed.

Active Learning results
-----------------------

The active learning routines return all data points in the order in which they were suggested together with the predictions for these data points and a 
(dictionary of) data frame with the training (and optionally test) scores of the models for each batch. As scores the MSE and the MAE are available.
