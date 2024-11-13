# -*- coding: utf-8 -*-
#
# fastecdsa configuration file
#
import sys
from os import path
from unittest.mock import patch

sys.path.append(path.abspath('.'))
sys.path.append(path.join('.', '..'))

patch('fastecdsa._ecdsa', new=patch.object(ModuleType, '__name__', 'fastecdsa._ecdsa'))
patch('fastecdsa.curvemath', new=patch.object(ModuleType, '__name__', 'fastecdsa.curvemath'))
