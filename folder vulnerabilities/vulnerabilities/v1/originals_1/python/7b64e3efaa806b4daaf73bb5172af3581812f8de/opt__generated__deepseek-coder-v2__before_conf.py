# -*- coding: utf-8 -*-
from datetime import datetime
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

MOCK_MODULES = ['fastecdsa._ecdsa', 'fastecdsa.curvemath']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- General configuration ------------------------------------------------
extensions = ['sphinx.ext.mathjax', 'sphinx.ext.autodoc']
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
project = 'fastecdsa'
copyright = f"{datetime.now().year}, Anton Kueltz"
author = 'Anton Kueltz'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------
html_theme = 'default'
html_static_path = ['_static']

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {}
latex_documents = [(master_doc, 'fastecdsa.tex', 'fastecdsa Documentation', 'Anton Kueltz', 'manual')]

# -- Options for manual page output ---------------------------------------
man_pages = [(master_doc, 'fastecdsa', 'fastecdsa Documentation', [author], 1)]

# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [(master_doc, 'fastecdsa', 'fastecdsa Documentation', author, 'fastecdsa', 'One line description of project.', 'Miscellaneous')]
