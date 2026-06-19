PyALAF - The Python Active Learning with Acquisition Functions package
======================================================================

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)
[![Documentation Status](https://readthedocs.org/projects/pythonal/badge/?version=latest)](https://pythonal.readthedocs.io/en/latest/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)]( http://perso.crans.org/besson/LICENSE.html)
[![Tests](https://github.com/TibMont/PyALAF/actions/workflows/tests_main.yml/badge.svg)](https://github.com/TibMont/PyALAF/actions/workflows/tests_main.yml)

PyALAF is a framework for using Active Learning in Python. It is specifically designed to use so-called acquisition functions for Active Learning, as discussed e.g. in [[1]](https://doi.org/10.1016/j.ins.2018.09.060) and [[2]](https://doi.org/10.1016/j.ins.2023.01.028).
The goal of this project is to enable sequential and batch-wise learning for pool and population data.
It can be used for example together with packages like LECA (Liquid Electrolyte Composition Analysis package) to combine Machine Learning-based modeling directly with Active Learning.

The LECA package can be found here: https://github.com/Harrison-Teeg/LECA.

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

    pip install git+https://github.com/TibMont/PyALAF.git

References
==========

[1] Wu, D.; Lin, C.-T.; Huang, J. Information Sciences 2019, 474, 90–105. DOI: https://doi.org/10.1016/j.ins.2018.09.060.
 
[2] Bemporad, A. Information Sciences 2023, 626, 275–292. DOI: https://doi.org/10.1016/j.ins.2023.01.028.

