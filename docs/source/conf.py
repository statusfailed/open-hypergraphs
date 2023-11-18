# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Setup -----------------------------------------------------
# Ensure the open_hypergraphs module is in the path so autosummary can load it.
import sys
from pathlib import Path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

# Mock numpy, scipy, and cupy so sphinx can build docs without installing them
# as dependencies.
# See https://blog.rtwilson.com/how-to-make-your-sphinx-documentation-compile-with-readthedocs-when-youre-using-numpy-and-scipy/
from unittest import mock
MOCK_MODULES = ['numpy', 'scipy', 'scipy.sparse', 'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.MagicMock()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# Project name etc.
project = 'open-hypergraphs'
copyright = '2023, Paul Wilson'
author = 'Paul Wilson'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme", # readthedocs theme
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "sphinx.ext.napoleon",
]

# Include both __init__ docstrings and class docstrings
# autoclass_content = "both" # the non-napoleon version of this option
napoleon_include_init_with_doc = True

# Number figures
numfig = True

# class members should be in the order they are written in the file
autodoc_member_order = "bysource"

# references
bibtex_bibfiles = ["refs.bib"]

# let autosummary recurse and generate all modules specified
# https://stackoverflow.com/questions/2701998/
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = []
# html_static_path = ['_static']
