PyAL - Python based Active Learning with Acquisition Functions
==============================================================

PyAL is a framework for using Active Learning in Python. It is specifically designed to use so-called acquisition functions for Active Learning, as discussed e.g. in [1]_ and [2]_.
The goal of this project is to enable sequential and batch-wise learning for pool and population data.
It can be used for example together with packages like LECA (Liquid Electrolyte Composition Analysis package) to combine Machine Learning based modeling directly with Active Learning.

The LECA package can be found [here](https://github.com/Harrison-Teeg/LECA).

Requirements
============
Python 3.9+

With the following python libraries:

    - Matplotlib 3.8.2+
    - Scikit-Learn 1.3.2+
    - Pandas 2.1.4+
    - Scipy 1.11.4+
    - Openpyxl 3.1.2+
    - Pyswarms 1.3.0+


Installation
============

This package can be installed directly from the repository using the command:

    pip install git+https://github.com/TibMont/PyAL.git

References
==========

.. [1] Wu, D.; Lin, C.-T.; Huang, J. Information Sciences 2019, 474, 90–105., doi: https://doi.org/10.1016/j.ins.2018.09.060.
 
.. [2] Bemporad, A. Information Sciences 2023, 626, 275–292.,doi: https://doi.org/10.1016/j.ins.2023.01.028.


