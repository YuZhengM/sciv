# -*- coding: UTF-8 -*-

import os
import sys
import hdf5plugin

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_DEFAULT_TIMEOUT"] = "3600"

sys.setrecursionlimit(1000)

if hdf5plugin.version > '6.0.0':
    pass

from . import download as dl
from . import file as fl
from . import model as ml
from . import plot as pl
from . import preprocessing as pp
from . import tool as tl
from . import util as ul

__version__ = f"{ul.project_name}: v{ul.project_version}"
