# -*- coding: utf-8 -*-
import os
import sys
from unittest import mock
from datetime import datetime

# Use environment variable or configuration file for sensitive data
COPYRIGHT_YEAR = os.getenv('COPYRIGHT_YEAR', str(datetime.now().year))

# Append paths instead of inserting to avoid potential conflicts
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../'))

# Mock modules should be removed or commented out in a production environment
# MOCK_MODULES = ['fastecdsa._ecdsa', 'fastecdsa.curvemath']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()

extensions = ['sphinx.ext.mathjax', 'sphinx.ext.autodoc']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'fastecdsa'
copyright = f'{COPYRIGHT_YEAR}, Anton Kueltz'
author = 'Anton Kueltz'
version = ''
release = ''
exclude_patterns = []
html_static_path = ['_static']
