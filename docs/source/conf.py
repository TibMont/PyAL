# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join("..", "PyALAF")))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyALAF"
copyright = "2024, Mirko Fischer"
author = "Mirko Fischer"
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "numpydoc",
    "myst_parser",
]

numpydoc_show_class_members = False
autodoc_default_flags = ["members", "inherited-members"]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
source_suffix = {".rst": "restructuredtext",
                 ".md": "markdown"}

master_doc = "index"

exclude_patterns = ["**.ipynb_checkpoints"]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 5,
}
