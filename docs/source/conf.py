# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NovAST'
copyright = '2025, Yu Zhu'
author = 'Yu Zhu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # for NumPy/Google style docstrings
    "myst_parser",           # markdown support
    "nbsphinx",              # Jupyter notebooks support
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
exclude_patterns = []
nbsphinx_execute = "never"

autodoc_mock_imports = [
    "scanpy",
    "anndata",
    "dask",
    "dask.array",
    "numpy",
    "scipy",
    "torch",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
