SKIP_TRAINING = False
import torch
import torch.nn as nn
import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from random import uniform
from math import ceil, floor
from itertools import zip_longest, chain
from typing import Sequence, Optional, Literal, Mapping
import numpy as np

from sklearn.linear_model import Ridge

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, random_split
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from tensordict import TensorDict
from torch.distributions.normal import Normal
from dataclasses import dataclass, astuple, field
from scipy.optimize import minimize
from py_vollib_vectorized import vectorized_implied_volatility
from py_vollib_vectorized import vectorized_black_scholes
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log = logging.getLogger(__name__)
normal = Normal(0, 1)
ridge_rel = 0.1    #  You may change this to 0.1 or 0.5

def worker_init_fn(worker_id):
    np.random.seed(2025 + worker_id)

# define module_lookup (used in FNNlayer)
def _module_lookup(module: str | Module) -> Module:
    """ Look up module by name or return module if already a module.
    """
    modules = torch.nn.__dict__.keys()
    module_dict = dict(zip(map(str.lower, modules), modules))

    if isinstance(module, str):
        return getattr(torch.nn, module_dict[module.lower()])()
    else:
        return module

## B-Spline Basis Function type 1
def B1_Spline13(x,y):
    return np.where( (x<=0.0) & (y>=0.0) & (y<= x+1.0), ((1.0+x-y)**3)/3.0,   # P^1_1
           np.where( (x>=0.0) & (y<=1.0) & (y>=x), ((1.0-y)**2) * (1.0 +3.0*x-y)/3.0,  #P^1_2
           np.where( (x<=1.0) & (y>=0.0) & (y<=x), (1.0-y)*(1.0+3.0*x-2.0*y +3.0*x*y - 3.0*x*x - 2.0*y*y) /3.0,  #P^1_3
           np.where((x >= 1.0) & (y <= 1.0) & (y >= x - 1.0), (1.0 - y) * (1.0 - y) * (4.0 - 3.0 * x + 2.0 * y) /3.0, #P^1_4
           np.where((x <= 2.0) & (y >= 0.0) & (y <= x - 1.0), ((2.0 - x) ** 3)  / 3.0,  #P^1_5
           np.where((x >= -1.0) & (y <= 0.0) & (y >= x), ((1.0 + x) *(1.0 + x) *(1.0 + x -3.0*y)) / 3.0,  #P^1_6
           np.where((x <= 0.0) & (y >= -1.0) & (y <= x), ((1.0 + x) * (1.0 + 2.0*x - 3.0 * y +3.0*x*y - 2.0*x*x - 3.0*y*y)) / 3.0, #P^1_7
           np.where((x >= 0.0) & (y <= 0.0) & (y >= x - 1.0), 1.0/3.0 + x - y - x*x - y*y + x*x*y - x*y*y,  #P^1_8
           np.where((x <= 1.0) & (y >= -1.0) & (y <= x - 1.0), ((2.0 - x + y) * (1.0 + 2.0*x - 2.0 * y -2.0*x*x +x*y - 2.0*y*y)) / 3.0,  #P^1_9
           np.where((x >= 1.0) & (y <= 0.0) & (y >= x - 2.0), ((2.0 - x + y) * (2.0 - x + y) * (2.0 - x - 2.0 * y)) / 3.0, #P^1_10         #
           np.where((x >= -1.0) & (y <= -1.0) & (y >= x - 1.0), ((1.0 + x) * (1.0 + x) * (4.0 - 2.0*x + 3.0 * y)) / 3.0, # P^1_11
           np.where((x <= 0.0) & (y >= -2.0) & (y <= x - 1.0), ((2.0 + y) **3) / 3.0, # P^1_12
           np.where((x >= 0.0) & (y <= -1.0) & (y >= x - 2.0), (((2.0 - x + y) ** 2) *(2.0+2.0*x+y)) / 3.0,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis Function type 2
def B2_Spline13(x,y):
    return np.where( (x>=0.0) & (y<=0.0) & (y>= x-1.0), ((1.0-x+y)**3)/3.0,   # P^1_1
           np.where( (x<=0.0) & (y>=-1.0) & (y<=x), ((1.0+y)**2) * (1.0 -3.0*x+y)/3.0,  #P^1_2
           np.where( (x>=-1.0) & (y<=0.0) & (y>=x), (1.0+y)*(1.0-3.0*x+2.0*y +3.0*x*y - 3.0*x*x - 2.0*y*y) /3.0,  #P^1_3
           np.where((x <= -1.0) & (y >= -1.0) & (y <= x + 1.0), (1.0 + y) * (1.0 + y) * (4.0 + 3.0 * x - 2.0 * y) /3.0, #P^1_4
           np.where((x >= -2.0) & (y <= 0.0) & (y >= x + 1.0), ((2.0 + x) ** 3)  / 3.0,  #P^1_5
           np.where((x <= 1.0) & (y >= 0.0) & (y <= x), ((1.0 - x) *(1.0 - x) *(1.0 - x + 3.0*y)) / 3.0,  #P^1_6
           np.where((x >= 0.0) & (y <= 1.0) & (y >= x), ((1.0 - x) * (1.0 - 2.0*x + 3.0 * y +3.0*x*y - 2.0*x*x - 3.0*y*y)) / 3.0, #P^1_7
           np.where((x <= 0.0) & (y >= 0.0) & (y <= x + 1.0), 1.0/3.0 - x + y - x*x - y*y - x*x*y + x*y*y,  #P^1_8
           np.where((x >= -1.0) & (y <= 1.0) & (y >= x + 1.0), ((2.0 + x - y) * (1.0 - 2.0*x + 2.0 * y -2.0*x*x +x*y - 2.0*y*y)) / 3.0,  #P^1_9
           np.where((x <= -1.0) & (y >= 0.0) & (y <= x + 2.0), ((2.0 + x - y) * (2.0 + x - y) * (2.0 + x + 2.0 * y)) / 3.0, #P^1_10         #
           np.where((x <= 1.0) & (y >= 1.0) & (y <= x + 1.0), ((1.0 - x) * (1.0 - x) * (4.0 + 2.0*x - 3.0 * y)) / 3.0, # P^1_11
           np.where((x >= 0.0) & (y <= 2.0) & (y >= x + 1.0), ((2.0 - y) **3) / 3.0, # P^1_12
           np.where((x <= 0.0) & (y >= 1.0) & (y <= x + 2.0), (((2.0 + x - y) ** 2) *(2.0-2.0*x-y)) / 3.0,  # P^1_13
           0.0)))))))))))))
## B-Spline Function values at given (x,y): type 1
def B1_Spline13_Scatter(xy_values, x_min,x_max,y_min,y_max, m, n):
    # xy_values: a 2D array of (x,y) scatter points
    # [x_min, x_max, y_min, y_max] gives the domain of B-spline function
    # delta_x, delta_y: the grid sizes
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]  #拆分散点坐标

    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)

    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)

    # Now apply the function B directlyprint(X)
    Z = B1_Spline13(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return Z
## B-Spline Function values at given (x,y): type 2
def B2_Spline13_Scatter(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize

    X, Y =  xy_values[:,0], xy_values[:,1]

    # Generate shift indices
    i_values = np.arange(-1, m + 2)[:, None, None]  # Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Shape: (1, n+2, 1, 1)

    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)

    # Now apply the function B directlyprint(X)
    Z = B2_Spline13(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return Z #Z 是 B（B‑样条基）在散点上的取值矩阵
## B-Spline Basis dB/dx Derivative at given (x,y): type 1
def B1_Spline13_dx(x,y):
    return np.where( (x<=0.0) & (y>=0.0) & (y<= x+1.0), (1.0+x-y)**2,   # P^1_1
           np.where( (x>=0.0) & (y<=1.0) & (y>=x), (1.0-y)**2,  #P^1_2
           np.where( (x<=1.0) & (y>=0.0) & (y<=x), (1.0-y)*(1.0 + y - 2.0*x),  #P^1_3
           np.where((x >= 1.0) & (y <= 1.0) & (y >= x - 1.0), -(1.0 - y)**2, #P^1_4
           np.where((x <= 2.0) & (y >= 0.0) & (y <= x - 1.0), -(2.0 - x) ** 2,  #P^1_5
           np.where((x >= -1.0) & (y <= 0.0) & (y >= x), (1.0 + x) *(1.0 + x - 2.0*y),  #P^1_6
           np.where((x <= 0.0) & (y >= -1.0) & (y <= x), 1.0 + 2.0*x*y - 2.0*x*x - y*y, #P^1_7
           np.where((x >= 0.0) & (y <= 0.0) & (y >= x - 1.0), 1.0 - 2.0*x +2.0*x*y - y*y,  #P^1_8
           np.where((x <= 1.0) & (y >= -1.0) & (y <= x - 1.0), 1.0 - 4.0*x + 2.0 * y + 2.0*x*x - 2.0*x*y + y*y,  #P^1_9
           np.where((x >= 1.0) & (y <= 0.0) & (y >= x - 2.0), -(2.0 - x + y) *(2.0 - x - y), #P^1_10         #
           np.where((x >= -1.0) & (y <= -1.0) & (y >= x - 1.0), 2.0*(1.0 + x) * (1.0 - x + y), # P^1_11
           np.where((x <= 0.0) & (y >= -2.0) & (y <= x - 1.0), 0.0, # P^1_12
           np.where((x >= 0.0) & (y <= -1.0) & (y >= x - 2.0), -2.0*(2.0 - x + y)*x,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis dB/dy Derivative at given (x,y): type 1
def B1_Spline13_dy(x,y):
    return np.where( (x<=0.0) & (y>=0.0) & (y<= x+1.0), -(1.0+x-y)**2,   # P^1_1
           np.where( (x>=0.0) & (y<=1.0) & (y>=x), -(1.0-y)*(1.0 + 2.0*x - y),  #P^1_2
           np.where( (x<=1.0) & (y>=0.0) & (y<=x), -1.0 + x*x -2.0*x*y + 2.0*y*y,  #P^1_3
           np.where((x >= 1.0) & (y <= 1.0) & (y >= x - 1.0), 2.0*( -1.0 + x -x*y + y*y), #P^1_4
           np.where((x <= 2.0) & (y >= 0.0) & (y <= x - 1.0), 0.0,  #P^1_5
           np.where((x >= -1.0) & (y <= 0.0) & (y >= x), -(1.0 + x) *(1.0 + x),  #P^1_6
           np.where((x <= 0.0) & (y >= -1.0) & (y <= x), (1.0 + x)*(-1.0 + x - 2.0*y), #P^1_7
           np.where((x >= 0.0) & (y <= 0.0) & (y >= x - 1.0), -1.0 - 2.0*y + x*x - 2.0*x*y,  #P^1_8
           np.where((x <= 1.0) & (y >= -1.0) & (y <= x - 1.0), -1.0 + 2.0*x - 4.0 * y - x*x + 2.0*x*y -2.0*y*y,  #P^1_9
           np.where((x >= 1.0) & (y <= 0.0) & (y >= x - 2.0), -2.0*(2.0 - x + y)*y, #P^1_10         #
           np.where((x >= -1.0) & (y <= -1.0) & (y >= x - 1.0), (1.0 + x) ** 2, # P^1_11
           np.where((x <= 0.0) & (y >= -2.0) & (y <= x - 1.0), (2.0+y)**2, # P^1_12
           np.where((x >= 0.0) & (y <= -1.0) & (y >= x - 2.0), (2.0 -x +y)*(2.0 + x + y),  # P^1_13
           0.0)))))))))))))
## B-Spline Basis d^2B/dx^2 Derivative at given (x,y): type 1
def B1_Spline13_dxx(x,y):
    return np.where( (x<=0.0) & (y>=0.0) & (y<= x+1.0), 2.0*(1.0+x-y),   # P^1_1
           np.where( (x>=0.0) & (y<=1.0) & (y>=x), 0.0,  #P^1_2
           np.where( (x<=1.0) & (y>=0.0) & (y<=x), -2.0* (1.0 - y),  #P^1_3
           np.where((x >= 1.0) & (y <= 1.0) & (y >= x - 1.0), 0.0, #P^1_4
           np.where((x <= 2.0) & (y >= 0.0) & (y <= x - 1.0), 2.0*(2.0-x),  #P^1_5
           np.where((x >= -1.0) & (y <= 0.0) & (y >= x), 2.0*(1.0 + x - y),  #P^1_6
           np.where((x <= 0.0) & (y >= -1.0) & (y <= x), -4.0* x + 2.0*y, #P^1_7
           np.where((x >= 0.0) & (y <= 0.0) & (y >= x - 1.0), - 2.0 + 2.0*y,  #P^1_8
           np.where((x <= 1.0) & (y >= -1.0) & (y <= x - 1.0), -4.0 + 4.0*x - 2.0 * y,  #P^1_9
           np.where((x >= 1.0) & (y <= 0.0) & (y >= x - 2.0), 4.0 - 2.0*x, #P^1_10         #
           np.where((x >= -1.0) & (y <= -1.0) & (y >= x - 1.0), -4.0*x + 2.0*y, # P^1_11
           np.where((x <= 0.0) & (y >= -2.0) & (y <= x - 1.0), 0.0, # P^1_12
           np.where((x >= 0.0) & (y <= -1.0) & (y >= x - 2.0), -4.0 + 4.0*x - 2.0*y,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis d^2B/dy^2 Derivative at given (x,y): type 1
def B1_Spline13_dyy(x,y):
    return np.where( (x<=0.0) & (y>=0.0) & (y<= x+1.0), 2.0*(1.0+x-y),   # P^1_1
           np.where( (x>=0.0) & (y<=1.0) & (y>=x), 2.0 + 2.0*x - 2.0*y,  #P^1_2
           np.where( (x<=1.0) & (y>=0.0) & (y<=x), -2.0*x + 4.0*y,  #P^1_3
           np.where((x >= 1.0) & (y <= 1.0) & (y >= x - 1.0), -2.0*x + 4.0*y, #P^1_4
           np.where((x <= 2.0) & (y >= 0.0) & (y <= x - 1.0), 0.0,  #P^1_5
           np.where((x >= -1.0) & (y <= 0.0) & (y >= x), 0.0,  #P^1_6
           np.where((x <= 0.0) & (y >= -1.0) & (y <= x), -2.0*(1.0+x), #P^1_7
           np.where((x >= 0.0) & (y <= 0.0) & (y >= x - 1.0), -2.0*(1.0+x),  #P^1_8
           np.where((x <= 1.0) & (y >= -1.0) & (y <= x - 1.0), -4.0 + 2.0*x - 4.0 * y,  #P^1_9
           np.where((x >= 1.0) & (y <= 0.0) & (y >= x - 2.0), -4.0 + 2.0*x - 4.0 * y, #P^1_10         #
           np.where((x >= -1.0) & (y <= -1.0) & (y >= x - 1.0), 0.0, # P^1_11
           np.where((x <= 0.0) & (y >= -2.0) & (y <= x - 1.0), 2.0*(2.0+y), # P^1_12
           np.where((x >= 0.0) & (y <= -1.0) & (y >= x - 2.0), 4.0 + 2.0*y,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis d^2B/dxy Derivative at given (x,y): type 1
def B1_Spline13_dxy(x,y):
    return np.where( (x<=0.0) & (y>=0.0) & (y<= x+1.0), -2.0*(1.0+x-y),   # P^1_1
           np.where( (x>=0.0) & (y<=1.0) & (y>=x), -2.0*(1.0-y),  #P^1_2
           np.where( (x<=1.0) & (y>=0.0) & (y<=x), 2.0*(x - y),  #P^1_3
           np.where((x >= 1.0) & (y <= 1.0) & (y >= x - 1.0), 2.0*(1.0 - y), #P^1_4
           np.where((x <= 2.0) & (y >= 0.0) & (y <= x - 1.0), 0.0,  #P^1_5
           np.where((x >= -1.0) & (y <= 0.0) & (y >= x), -2.0*(1.0+x),  #P^1_6
           np.where((x <= 0.0) & (y >= -1.0) & (y <= x), 2.0*(x - y), #P^1_7
           np.where((x >= 0.0) & (y <= 0.0) & (y >= x - 1.0), 2.0*(x - y),  #P^1_8
           np.where((x <= 1.0) & (y >= -1.0) & (y <= x - 1.0), 2.0*(1.0 - x + y),  #P^1_9
           np.where((x >= 1.0) & (y <= 0.0) & (y >= x - 2.0), 2.0 * y, #P^1_10         #
           np.where((x >= -1.0) & (y <= -1.0) & (y >= x - 1.0), 2.0*(1.0+x), # P^1_11
           np.where((x <= 0.0) & (y >= -2.0) & (y <= x - 1.0), 0.0, # P^1_12
           np.where((x >= 0.0) & (y <= -1.0) & (y >= x - 2.0), -2.0*x,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis dB/dx Derivative at given (x,y): type 2
def B2_Spline13_dx(x,y):
    return np.where( (x>=0.0) & (y<=0.0) & (y>= x-1.0), -(1.0-x+y)**2,   # P^1_1
           np.where( (x<=0.0) & (y>=-1.0) & (y<=x), -(1.0+y)**2,  #P^1_2
           np.where( (x>=-1.0) & (y<=0.0) & (y>=x), -(1.0+y)*(1.0+2.0*x-y),  #P^1_3
           np.where((x <= -1.0) & (y >= -1.0) & (y <= x + 1.0), (1.0 + y) * (1.0 + y), #P^1_4
           np.where((x >= -2.0) & (y <= 0.0) & (y >= x + 1.0), (2.0 + x) *(2.0 + x),  #P^1_5
           np.where((x <= 1.0) & (y >= 0.0) & (y <= x), -(1.0 - x) *(1.0 - x + 2.0*y),  #P^1_6
           np.where((x >= 0.0) & (y <= 1.0) & (y >= x), -1.0 - 2.0*x*y + 2.0*x*x + y*y, #P^1_7
           np.where((x <= 0.0) & (y >= 0.0) & (y <= x + 1.0), -1.0 - 2.0*x - 2.0*x*y + y*y,  #P^1_8
           np.where((x >= -1.0) & (y <= 1.0) & (y >= x + 1.0), -1.0 - 4.0*x + 2.0 * y -2.0*x*x +2.0*x*y - y*y,  #P^1_9
           np.where((x <= -1.0) & (y >= 0.0) & (y <= x + 2.0), (2.0 + x - y) * (2.0 + x + y), #P^1_10         #
           np.where((x <= 1.0) & (y >= 1.0) & (y <= x + 1.0), -2.0*(1.0 - x) * (1.0 + x - y), # P^1_11
           np.where((x >= 0.0) & (y <= 2.0) & (y >= x + 1.0), 0.0, # P^1_12
           np.where((x <= 0.0) & (y >= 1.0) & (y <= x + 2.0), -2.0*(2.0 + x - y)*x,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis dB/dy Derivative at given (x,y): type 2
def B2_Spline13_dy(x,y):
    return np.where( (x>=0.0) & (y<=0.0) & (y>= x-1.0), (1.0-x+y)**2,   # P^1_1
           np.where( (x<=0.0) & (y>=-1.0) & (y<=x), (1.0+y)*(1.0 - 2.0*x +y),  #P^1_2
           np.where( (x>=-1.0) & (y<=0.0) & (y>=x), -1.0 - x*x + 2.0*x*y-2.0*y*y,  #P^1_3
           np.where((x <= -1.0) & (y >= -1.0) & (y <= x + 1.0), 2.0*(1.0 + x + x*y - y*y), #P^1_4
           np.where((x >= -2.0) & (y <= 0.0) & (y >= x + 1.0), 0.0,  #P^1_5
           np.where((x <= 1.0) & (y >= 0.0) & (y <= x), (1.0 - x) *(1.0 - x),  #P^1_6
           np.where((x >= 0.0) & (y <= 1.0) & (y >= x), -(1.0 - x)*(-1.0 - x + 2.0*y), #P^1_7
           np.where((x <= 0.0) & (y >= 0.0) & (y <= x + 1.0), 1.0 - 2.0*y - x*x + 2.0*x*y,  #P^1_8
           np.where((x >= -1.0) & (y <= 1.0) & (y >= x + 1.0), 1.0 + 2.0*x - 4.0 * y + x*x - 2.0*x*y + 2.0*y*y,  #P^1_9
           np.where((x <= -1.0) & (y >= 0.0) & (y <= x + 2.0), -2.0*(2.0 + x - y) * y, #P^1_10         #
           np.where((x <= 1.0) & (y >= 1.0) & (y <= x + 1.0), -(1.0 - x) * (1.0 - x), # P^1_11
           np.where((x >= 0.0) & (y <= 2.0) & (y >= x + 1.0), -(2.0 - y)**2, # P^1_12
           np.where((x <= 0.0) & (y >= 1.0) & (y <= x + 2.0), -(2.0 + x - y)*(2.0-x-y),  # P^1_13
           0.0)))))))))))))
## B-Spline Basis d^2B/dx^2 Derivative at given (x,y): type 2
def B2_Spline13_dxx(x,y):
    return np.where( (x>=0.0) & (y<=0.0) & (y>= x-1.0), 2.0*(1.0-x+y),   # P^1_1
           np.where( (x<=0.0) & (y>=-1.0) & (y<=x), 0.0,  #P^1_2
           np.where( (x>=-1.0) & (y<=0.0) & (y>=x), -2.0*(1.0 + y),  #P^1_3
           np.where((x <= -1.0) & (y >= -1.0) & (y <= x + 1.0), 0.0, #P^1_4
           np.where((x >= -2.0) & (y <= 0.0) & (y >= x + 1.0), 2.0*(2.0+x),  #P^1_5
           np.where((x <= 1.0) & (y >= 0.0) & (y <= x), 2.0*(1.0 - x + y),  #P^1_6
           np.where((x >= 0.0) & (y <= 1.0) & (y >= x),  4.0*x - 2.0*y, #P^1_7
           np.where((x <= 0.0) & (y >= 0.0) & (y <= x + 1.0), -2.0 - 2.0*y,  #P^1_8
           np.where((x >= -1.0) & (y <= 1.0) & (y >= x + 1.0), -4.0 - 4.0 * x + 2.0*y,  #P^1_9
           np.where((x <= -1.0) & (y >= 0.0) & (y <= x + 2.0), 4.0 + 2.0*x, #P^1_10         #
           np.where((x <= 1.0) & (y >= 1.0) & (y <= x + 1.0),4.0*x -2.0*y, # P^1_11
           np.where((x >= 0.0) & (y <= 2.0) & (y >= x + 1.0), 0.0, # P^1_12
           np.where((x <= 0.0) & (y >= 1.0) & (y <= x + 2.0), -4.0 - 4.0* x + 2.0*y,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis d^2B/dy^2 Derivative at given (x,y): type 2
def B2_Spline13_dyy(x,y):
    return np.where( (x>=0.0) & (y<=0.0) & (y>= x-1.0), 2.0*(1.0-x+y),   # P^1_1
           np.where( (x<=0.0) & (y>=-1.0) & (y<=x), 2.0 - 2.0*x + 2.0*y,  #P^1_2
           np.where( (x>=-1.0) & (y<=0.0) & (y>=x), 2.0*x - 4.0*y,  #P^1_3
           np.where((x <= -1.0) & (y >= -1.0) & (y <= x + 1.0), 2.0*x - 4.0*y, #P^1_4
           np.where((x >= -2.0) & (y <= 0.0) & (y >= x + 1.0), 0.0,  #P^1_5
           np.where((x <= 1.0) & (y >= 0.0) & (y <= x), 0.0,  #P^1_6
           np.where((x >= 0.0) & (y <= 1.0) & (y >= x),  -2.0*( 1.0 - x), #P^1_7
           np.where((x <= 0.0) & (y >= 0.0) & (y <= x + 1.0), -2.0*( 1.0 - x),  #P^1_8
           np.where((x >= -1.0) & (y <= 1.0) & (y >= x + 1.0), -4.0 - 2.0 * x + 4.0*y,  #P^1_9
           np.where((x <= -1.0) & (y >= 0.0) & (y <= x + 2.0), -4.0 - 2.0 * x + 4.0*y, #P^1_10         #
           np.where((x <= 1.0) & (y >= 1.0) & (y <= x + 1.0), 0.0, # P^1_11
           np.where((x >= 0.0) & (y <= 2.0) & (y >= x + 1.0), 2.0*(2.0 - y), # P^1_12
           np.where((x <= 0.0) & (y >= 1.0) & (y <= x + 2.0), 4.0 - 2.0*y,  # P^1_13
           0.0)))))))))))))
## B-Spline Basis d^2B/dxy Derivative at given (x,y): type 2
def B2_Spline13_dxy(x,y):
    return np.where( (x>=0.0) & (y<=0.0) & (y>= x-1.0), -2.0*(1.0-x+y),   # P^1_1
           np.where( (x<=0.0) & (y>=-1.0) & (y<=x), - 2.0*(1.0 + y),  #P^1_2
           np.where( (x>=-1.0) & (y<=0.0) & (y>=x), 2.0*(-x +y),  #P^1_3
           np.where((x <= -1.0) & (y >= -1.0) & (y <= x + 1.0), 2.0*(1.0 + y), #P^1_4
           np.where((x >= -2.0) & (y <= 0.0) & (y >= x + 1.0), 0.0,  #P^1_5
           np.where((x <= 1.0) & (y >= 0.0) & (y <= x), -2.0*(1.0 - x),  #P^1_6
           np.where((x >= 0.0) & (y <= 1.0) & (y >= x),  2.0*(-x + y), #P^1_7
           np.where((x <= 0.0) & (y >= 0.0) & (y <= x + 1.0), 2.0*(-x + y),  #P^1_8
           np.where((x >= -1.0) & (y <= 1.0) & (y >= x + 1.0), 2.0*(1.0 + x - y),  #P^1_9
           np.where((x <= -1.0) & (y >= 0.0) & (y <= x + 2.0),   - 2.0 * y, #P^1_10         #
           np.where((x <= 1.0) & (y >= 1.0) & (y <= x + 1.0), 2.0*(1.0 - x), # P^1_11
           np.where((x >= 0.0) & (y <= 2.0) & (y >= x + 1.0), 0.0, # P^1_12
           np.where((x <= 0.0) & (y >= 1.0) & (y <= x + 2.0), 2.0*x,  # P^1_13
           0.0)))))))))))))
## B-Spline Function dB/dx Derivative at given (x,y): type 1
def B1_Spline13_Scatter_dx(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dx = B1_Spline13_dx(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dx
## B-Spline Function dB/dy Derivative at given (x,y): type 1
def B1_Spline13_Scatter_dy(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dy = B1_Spline13_dy(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dy
## B-Spline Function d^2B/dxx Derivative at given (x,y): type 1
def B1_Spline13_Scatter_dxx(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dxx = B1_Spline13_dxx(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dxx
## B-Spline Function d^2B/dxy Derivative at given (x,y): type 1
def B1_Spline13_Scatter_dxy(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dxy = B1_Spline13_dxy(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dxy
## B-Spline Function d^2B/dyy Derivative at given (x,y): type 1
def B1_Spline13_Scatter_dyy(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dyy = B1_Spline13_dyy(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dyy
## B-Spline Function dB/dx Derivative at given (x,y): type 2
def B2_Spline13_Scatter_dx(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dx = B2_Spline13_dx(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dx
## B-Spline Function dB/dy Derivative at given (x,y): type 2
def B2_Spline13_Scatter_dy(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dy = B2_Spline13_dy(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dy
## B-Spline Function d^2B/dx^2 Derivative at given (x,y): type 2
def B2_Spline13_Scatter_dxx(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dxx = B2_Spline13_dxx(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dxx
## B-Spline Function d^2B/dxy Derivative at given (x,y): type 2
def B2_Spline13_Scatter_dxy(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dxy = B2_Spline13_dxy(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dxy
## B-Spline Function d^2B/dy^2 Derivative at given (x,y): type 2
def B2_Spline13_Scatter_dyy(xy_values, x_min,x_max,y_min,y_max, m, n):
    Lx = x_max - x_min
    Ly = y_max - y_min
    delta_x = Lx / m  # stepsize
    delta_y = Ly / n  # stepsize
    X, Y =  xy_values[:,0], xy_values[:,1]
    # Generate shift indices, we add a fake spline
    i_values = np.arange(-1, m + 2)[:, None, None]  # It should be (-1, m+1) Shape: (m+2, 1, 1, 1)print(Z[0,0])
    j_values = np.arange(-1, n + 2)[None, :, None]  # Should be (0, n+2)Shape: (1, n+2, 1, 1)
    # Expand X and Y to 4D arrays
    X_shifted = (X[None, None, :] - x_min) / delta_x - i_values  # Shape: (m+2, 1, x_size, y_size)
    Y_shifted = (Y[None, None, :] - y_min) / delta_y - j_values  # Shape: (1, n+2, x_size, y_size)
    # Now apply the function B directlyprint(X)
    dyy = B2_Spline13_dyy(X_shifted, Y_shifted)  # Shape: (m+2, n+2, xy_size)

    return dyy

# define linear (included in FNNlayer)
class Linear(Module):
    """ Linear layer with optional spatial dimension.
    The spatial dimension determines the number of axes (on the right) along which to operate pointwise.
    """
    def __init__(self, in_channels: int, out_channels: int, spatial_dim: int = 0, bias: bool = True) -> None:
        super().__init__()
        if spatial_dim == 0:
            self._linear = torch.nn.Linear(in_channels, out_channels, bias=bias)
            self._forward = F.linear
        else:
            self._linear = getattr(torch.nn, f"Conv{spatial_dim}d")(in_channels, out_channels, kernel_size=1, bias=bias)
            self._forward = getattr(F, f"conv{spatial_dim}d")

        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        return self._forward(input, self.weight, self.bias)

    @property
    def weight(self) -> Tensor:
        return self._linear.weight

    @property  # property方法，只能读取，不能直接赋值
    def bias(self) -> Optional[Tensor]:  # 让self.bias等价于self._linear.bias
        return self._linear.bias

    def __repr__(self) -> str:
        return f"{self._get_name()}({self.in_channels}, {self.out_channels}, spatial_dim={self.spatial_dim}, bias={self.bias is not None})"

# define FNNlayer
class FNNLayer(Sequential):
    """ Feedforward neural network layer.
    """
    def __init__(self, linear: Linear, batch_norm: Optional[Module] = None,
                 activation: Optional[Module | str] = None, dropout: float = 0.) -> None:
        super().__init__()
        if batch_norm is not None:
            self.add_module('batch_norm', batch_norm)
        else:
            self.register_parameter('batch_norm', None)
        self.add_module('linear', linear)
        if isinstance(activation, str):
            activation = _module_lookup(activation)
        if activation is not None:
            self.add_module('activation', activation)
        else:
            self.register_parameter('activation', None)
        if dropout > 0.:
            self.add_module('dropout', torch.nn.Dropout(dropout))
        else:
            self.register_parameter('dropout', None)

    @classmethod
    def from_config(cls, in_channels: int, out_channels: int, spatial_dim: int = 0,
                    batch_norm: bool = False, **kwargs) -> 'FNNLayer':

        linear = Linear(in_channels, out_channels, spatial_dim=spatial_dim, bias=kwargs.get('bias', True))
        activation = kwargs.get('activation', None)
        dropout = kwargs.get('dropout', 0.)

        if batch_norm:
            bn_kwargs = {key: kwargs[key] for key in ['eps', 'momentum', 'affine', 'track_running_stats'] if
                         key in kwargs}
            batch_norm = getattr(torch.nn, f"BatchNorm{max(spatial_dim, 1)}d")(in_channels, **bn_kwargs)
        else:
            batch_norm = None

        return FNNLayer(linear, batch_norm=batch_norm, activation=activation, dropout=dropout)


# define FNN
class FNN(Sequential):
    """ Feedforward neural network.
    """
    def __init__(self, *layers: FNNLayer) -> None:
        super().__init__(*layers)

    @classmethod
    def from_config(cls, channels: Sequence[int], spatial_dim: int = 0, **kwargs) -> 'FNN':
        """ Create feedforward neural network from channel sizes, spatial dimension, and config kwargs.
        Parameters
        ----------
        channels
            Sequence of channel sizes
        spatial_dim, optional
            Spatial dimension, by default 0

        Returns
        -------
        FNN
            Feedforward neural network
        """
        kwargs = kwargs.copy()

        hidden_activation = kwargs.get('hidden_activation', 'GELU')
        output_activation = kwargs.get('output_activation', None)
        activations = ([hidden_activation] * (len(channels) - 2) + [output_activation])

        layers = []
        for i in range(len(channels) - 1):
            kwargs['activation'] = activations[i]
            layers.append(FNNLayer.from_config(channels[i], channels[i + 1], spatial_dim, **kwargs))

        return FNN(*layers)

# define kernel
class NonlinearKernelTransformWithSkip(Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, spatial_dim: int, hidden_channels: Sequence[int], **kwargs) -> None:
        super().__init__()

        self.fnn = FNN.from_config((2 * spatial_dim + in_channels + skip_channels, *hidden_channels, out_channels * (in_channels + 1)), **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.spatial_dim = spatial_dim

    def forward(self, pos_i: Tensor, pos_j: Tensor, v_j: Tensor, x_j: Tensor, **kwargs) -> Tensor:
        pos_ij = torch.cat((pos_i, pos_j, v_j, x_j), dim=-1)
        kernel = self.fnn(pos_ij).reshape(-1, self.out_channels, self.in_channels + 1)
        kernel, bias = kernel[..., :, :-1], kernel[..., :, -1]
        return torch.einsum(
            '...ij,...j->...i',
            kernel,
            v_j
        ) + bias

# define GNOlayer message passing
class BS_GNOLayer(MessagePassing):
    """Graph Neural Operator (GNO) layer.
    """
    def __init__(self, channels: int, transform: Module,
                 local_linear: bool = False, local_bias: bool = True,
                 activation: Optional[Module] = None,
                 lifting: Optional[Module] = None, projection: Optional[Module] = None) -> None:
        super().__init__(aggr='mean', flow='source_to_target')

        self.transform = transform

        if local_linear:
            self.local = FNN.from_config((channels, channels), batch_norm=False, bias=False)
        else:
            self.register_parameter('local', None)

        if local_bias:
            self.bias = torch.nn.Parameter(Tensor(channels))
        else:
            self.register_parameter('bias', None)

        if activation is not None:
            self.add_module('activation', activation)
        else:
            self.register_parameter('activation', None)

        if lifting is not None:
            self.add_module('lifting', lifting)
        else:
            self.register_parameter('lifting', None)

        if projection is not None:
            self.add_module('projection', projection)
        else:
            self.register_parameter('projection', None)

        self.channels = channels

        self.reset_parameters()

    def __repr__(self) -> str:
        return super(MessagePassing, self).__repr__()

    def extra_repr(self) -> str:
        return 'channels={}, local_linear={}, local_bias={}'.format(
            self.channels, self.local is not None, self.bias is not None
        )

    def update(self, aggr_out: Tensor, pos: Tensor, v: Tensor, x: Tensor) -> Tensor:
        if self.local is not None:
            aggr_out = aggr_out + self.local(v)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def message(self, pos_i, pos_j, v_i, v_j, x_i, x_j) -> Tensor:
        w = self.transform(pos_i=pos_i, pos_j=pos_j, v_i=v_i, v_j=v_j, x_i=x_i, x_j=x_j)
        return w

    def forward(self, edge_index: Tensor, pos: Tensor, v: Tensor, x: Tensor) -> Tensor:
        if self.lifting is not None:
            v = self.lifting(v)
        w = self.propagate(edge_index, pos=pos, v=v, x=x)
        if self.projection is not None:
            w = self.projection(w)
        if self.activation is not None:
            w = self.activation(w)
        return w

# define GNO
class BS_GNO(Module):
    """Graph Neural Operator (GNO) model.
    Given by a sequence of GNO layers.
    """
    def __init__(self, *gno_layers: BS_GNOLayer, lim_r: tuple[float, float] = (0.01, 1.0),
                 lim_z: tuple[float, float] = (-1.5, 0.5),
                 BS_Steps_r: int = 30, BS_Steps_z: int = 50, in_channels: int = 1) -> None:
        super().__init__()
        self.gno_layers = ModuleList(gno_layers)
        self.in_channels = in_channels
        self.BS_Steps_r = BS_Steps_r
        self.BS_Steps_z = BS_Steps_z
        self.lim_r = lim_r
        self.lim_z = lim_z

    def forward(self, x: Tensor, pos_x: Tensor, pos_y: Tensor, edge_index: Tensor, preB_coeffs: Tensor) -> tuple[Tensor, Tensor]:
        """GNO forward propagation.
        Parameters
        ----------
        x
            Input data, shape (n, in_channels)
        pos_x
            Coordinate locations of input data x, shape (n, domain_dim)
        pos_y
            Coordinate locations at B-spline grids, shape (n, domain_dim)
        edge_index
            Graph connectivity, shape (2, e)

        Returns
        -------
        tuple[Tensor, Tensor]
            #Output data at pos_x, shape (n, out_channels)
            Output B-spline coefficients at pos_y, shape (m, out_channels)
        """
        n = pos_x.size(0)
        m = pos_y.size(0)
        x = torch.cat((x, torch.full((m, self.in_channels), fill_value=0., dtype=x.dtype, device=x.device)))
        pos = torch.cat((pos_x, pos_y))

        w = x
        for gno_layer in self.gno_layers:
            w = gno_layer(edge_index, pos=pos, v=w, x=x)

        # B-Spline coefficents
        _, w_y = torch.split(w, (n, m), dim=0)
        w_y = w_y.view(self.BS_Steps_r+3, self.BS_Steps_z+3, -1)  #+ preB_coeffs # I used m+3 and n+3 all for B^1 and B^2

        return w_y

    def __str__(self) -> str:
        extra_lines = []
        child_lines = []

        for key, module in self._modules.items():
            child_lines.append(f"({key}): {str(module)}")
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

class LearnableSoftplus(torch.nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta)
        self.bias = torch.nn.Parameter(torch.tensor(0.6))  # Learnable bias

    def forward(self, x):
        return self.softplus(x) - self.bias

# GNO model instantiate
def create_BS_gno(BS_Steps_r = 20, BS_Steps_z = 30, spatial_dim = 2, num_hidden_layers = 3, in_channels = 1, out_channels = 2, gno_channels = 16, fnn_hidden_channels = 64):
    channels = (in_channels, *(num_hidden_layers * (gno_channels,)), out_channels)  # in total four layers
    gno_layers = []
    for i in range(m := (len(channels) - 1)):
        lifting = FNN.from_config((channels[i], fnn_hidden_channels, gno_channels), hidden_activation='gelu',
                                  batch_norm=False)
        projection = None if i < m - 1 else FNN.from_config((gno_channels, fnn_hidden_channels, channels[i + 1]),
                                                            hidden_activation='gelu', batch_norm=False)
        transform = NonlinearKernelTransformWithSkip(in_channels=gno_channels, out_channels=gno_channels,
                                                     skip_channels=in_channels, spatial_dim=spatial_dim,
                                                     hidden_channels=(fnn_hidden_channels, fnn_hidden_channels),
                                                     hidden_activation='gelu', batch_norm=False)
        if i == 0:
            local_linear = False
        else:
            local_linear = True
        # Shall we keep using LearnableSoftplus()?
        activation = torch.nn.GELU() if i < m - 1 else LearnableSoftplus()   #torch.nn.Softplus(beta=0.5)   #LearnableSoftplus()  # # #torch.nn.Identity()  #

        gno_layer = BS_GNOLayer(gno_channels, transform=transform, local_linear=local_linear, local_bias=True,
                                activation=activation, lifting=lifting, projection=projection)
        gno_layers.append(gno_layer)

    gno = BS_GNO(*gno_layers, in_channels=in_channels, BS_Steps_r=BS_Steps_r, BS_Steps_z=BS_Steps_z)
    return gno

def imply_borrow(x: pd.DataFrame, k: int = 3, atm_bound: Optional[float] = None) -> pd.Series:
    """
    put-call parity
    """
    if atm_bound is None:
        calls = x[x[('mid', 'P')] > x[('mid', 'C')]].head(k)
        puts = x[x[('mid', 'P')] <= x[('mid', 'C')]].head(k)
        atm = pd.concat((calls, puts))
    else:
        moneyness = x.index.get_level_values('strike') / x['underlying_mid', 'C']
        atm = x.loc[(1 - atm_bound <= moneyness) & (moneyness <= 1 + atm_bound)]
    y = atm[('mid', 'C')] - atm[('mid', 'P')]  # 价差
    x_strike = atm.index.get_level_values(level='strike').values  # 用于拟合方程 m⋅A+c=y
    A = np.vstack([x_strike, np.ones(len(x_strike))]).T
    # print(f"A shape: {A.shape}, A values:\n{A}")
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    discount_factor = -m
    underlying_forward = c / discount_factor
    d = {
        'discount_factor': discount_factor,
        'underlying_forward': underlying_forward
    }
    return pd.Series(d, index=['discount_factor', 'underlying_forward'])

# define OptionsDataset
class OptionsDataset(Dataset, ABC):
    def __init__(self, force_reprocess: bool = False) -> None:
        if os.getenv("OPDS_CACHE_DIR") is None:
            raise ValueError("OPDS_CACHE_DIR environment variable not set")
        cache_dir = Path(os.getenv("OPDS_CACHE_DIR")) / self.__class__.__name__

        if not cache_dir.exists() or force_reprocess:
            cache_dir.mkdir(parents=True, exist_ok=True)
            df = (self
                  .load_data()
                  .dropna(subset=['expiry_datetime', 'quote_datetime', 'strike', 'option_type', 'bid', 'ask'])
                  .astype({'quote_datetime': 'datetime64[ns]',
                           'expiry_datetime': 'datetime64[ns]',
                           'strike': float,
                           'option_type': 'str',
                           'bid': float,
                           'ask': float})
                  .pipe(self.add_time_to_maturity)
                  .pipe(self.add_mid)
                  .set_index(['quote_datetime', 'expiry_datetime', 'strike'])
                  .sort_index()
                  .get([
                'option_type',
                'time_to_maturity',
                'mid',
                'bid',
                'ask'
            ])
                  .pipe(self.add_forward, drop_otm=True)
                  .pipe(self.add_implied_volatility)  # 计算隐含波动率
                  .get(self.columns)
                  .dropna())
            self._cache_data(df, cache_dir)

        log.info("Assembling and sorting index of cached files.")
        self.file_paths = sorted(cache_dir.glob(f"{self.__class__.__name__}_*.pt"))
        self.quote_datetimes = pd.DatetimeIndex([self._get_quote_datetime(file) for file in self.file_paths],
                                                name='quote_datetime')
        log.info(f"Created index of {len(self.file_paths)} files.")

    @classmethod
    @abstractmethod
    def load_data(cls, data_dir: str) -> pd.DataFrame:
        """Load the data from the given directory and process
        Return a pandas dataframe with the following columns:

        * ``expiry_datetime``: expiry timestamp of the option
        * ``quote_datetime``: timestamp of the quote (recorded when)
        * ``option_type``: 'C' for call and 'P' for put
        * ``strike``: strike price of the option
        * ``bid``: bid price of the option
        * ``ask``: ask price of the option

        Parameters
        ----------
        data_dir
            The path to the directory containing the data

        Returns
        -------
            The processed data
        """
        pass

    def __len__(self):
        return len(self.quote_datetimes)

    def __getitem__(self, i: int):
        return torch.load(self.file_paths[i], weights_only=True)

    def _get_quote_datetime(self, filepath):
        date_string = str(filepath).split(f'{type(self).__name__}_')[1].split('.')[0]
        quote_datetime = datetime.strptime(date_string, '%Y-%m-%d-%H-%M-%S')
        return quote_datetime

    def _cache_data(self, df: pd.DataFrame, cache_dir: Optional[str] = None) -> None:
        quote_datetimes = df.index.get_level_values('quote_datetime').unique()
        for i, quote_datetime in enumerate(quote_datetimes):
            surface = df.xs(quote_datetime, level='quote_datetime')
            surface = surface.reset_index(drop=True)
            surface = torch.tensor(surface.values.T, dtype=torch.float32)
            datestr = quote_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            filepath = os.path.join(cache_dir, f"{type(self).__name__}_{datestr}.pt")
            torch.save(surface, filepath)  # 将每个时间点的波动率曲面保存为.pt文件，便于后续神经算子的快速读取。

    @property
    def columns(self):
        return ['time_to_maturity', 'log_moneyness', 'implied_volatility', 'bid', 'ask', 'discount_factor',
                'underlying_forward']

    @staticmethod
    def add_time_to_maturity(df: pd.DataFrame) -> pd.DataFrame:
        df['time_to_maturity'] = (df['expiry_datetime'] - df['quote_datetime']).dt.total_seconds() / (
                365 * 24 * 60 * 60)
        df = df.query('time_to_maturity > 0')  # drop time_to_maturity == 0
        return df

    @staticmethod
    def add_mid(df: pd.DataFrame) -> pd.DataFrame:
        df['mid'] = (df['bid'] + df['ask']) / 2
        df = df.query('mid > 0')  # discards zero-ask options
        return df

    @staticmethod
    def add_forward(df: pd.DataFrame, drop_otm=True) -> pd.DataFrame:
        """Adds `discount_factor`, `underlying_forward`, `log_moneyness` to dataframe

        Based on pandas `apply`-method for grouped datarfames, which is slow.
        Might want to hardcode the linear regression...
        """
        df = df.pivot(columns='option_type')
        borrow = df.groupby(['quote_datetime', 'expiry_datetime']).apply(imply_borrow)

        df['discount_factor', 'C'] = borrow['discount_factor']
        df['discount_factor', 'P'] = borrow['discount_factor']
        df['underlying_forward', 'C'] = borrow['underlying_forward']
        df['underlying_forward', 'P'] = borrow['underlying_forward']
        df['log_moneyness', 'C'] = np.log(df.index.get_level_values('strike') / df['underlying_forward', 'C'])
        df['log_moneyness', 'P'] = np.log(df.index.get_level_values('strike') / df['underlying_forward', 'P'])

        calls = df.xs('C', axis=1, level='option_type')
        puts = df.xs('P', axis=1, level='option_type')

        if drop_otm:
            calls = calls.loc[calls['log_moneyness'] > 0]
            puts = puts.loc[puts['log_moneyness'] <= 0]

        df = pd.concat((calls, puts)).sort_index()
        return df

    @staticmethod
    def add_implied_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """Add `implied_volatility_mid`, `implied_volatility_bid`, `implied_volatility_ask` to dataframe

        Based on vectorized implementation of P. Jäckels "Let's be rational" method for computing implied volatility, available as `py_vollib_vectorized` in PyPI.
        Can be a bit tricky to install sometimes...
        """
        S = df['underlying_forward'].values
        K = df.index.get_level_values('strike').values
        t = df['time_to_maturity'].values
        r = np.zeros_like(t)  # (- np.log(df['discount_factor']) / t).values
        flag = np.full(r.shape, fill_value='c')
        flag[df['log_moneyness'] <= 0] = 'p'
        df['implied_volatility'] = vectorized_implied_volatility(df['mid'].values, S, K, t, r, flag, return_as='array')
        return df

# define WRDSOptionsDataset
class WRDSOptionsDataset(OptionsDataset):
    """Dataset for OptionsMetrics data as provided through the WRDS
    When downloading the options chain (for a specified index and a certain date range) from WRDS, be sure to include the following fields:
    * ``date``
    * ``exdate``
    * ``cp_flag``
    * ``strike_price``
    * ``best_bid``
    * ``best_offer``
    * ``am_settlement`` (to be able to discard weekly options)
    Then, place the generated csv-file into an apposite directory, and provide as ``data_dir`` during init.
    """
    @classmethod
    def load_data(cls) -> pd.DataFrame:
        data_dir = Path(os.environ['OPDS_WRDS_DATA_DIR'])
        csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No csv files found in {data_dir}")
        elif len(csv_files) > 1:
            raise ValueError(f"Multiple csv files found in {data_dir}")
        else:
            filepath = os.path.join(data_dir, csv_files[0])

        col_names = {
            'date': 'quote_datetime',
            'exdate': 'expiry_datetime',
            'strike_price': 'strike',
            'cp_flag': 'option_type',
            'best_bid': 'bid',
            'best_offer': 'ask'
        }
        data = (pd.read_csv(filepath, engine='python')
                .query('am_settlement == 1')
                .assign(strike_price=lambda df: df['strike_price'] / 1000)
                .rename(columns=col_names)
                .astype({'quote_datetime': 'datetime64[ns]', 'expiry_datetime': 'datetime64[ns]'})
                .get(col_names.values()))

        return data

# 4.4.1 blacksholes model
def normalizing_transforms(r, z, iv):
    a = - z / iv  # z = log(S/K)/ iv  * r
    total_volatility = iv * r
    d1, d2 = a + total_volatility / 2, a - total_volatility / 2
    return d1, d2

def vega_(r: Tensor, z: Tensor, iv: Tensor) -> Tensor:
    """Compute Black-Scholes vega.
    r：Square root of time-to-expiry
    z：Normalized log-moneyness
    iv：Black-Scholes volatility
    """
    d1, _ = normalizing_transforms(r, z, iv)  # calculate d1
    return normal.log_prob(d1).exp() * r  # vega = S * r * PDF of d1, here calculate log PDF first them exp() to the original PDF


def normalized_option_price(r: Tensor, z: Tensor, iv: Tensor) -> Tensor:
    option_type = torch.ones_like(z)
    option_type[z < 0] = -1  # strike price larger than stock price, indicating a put option
    d1, d2 = normalizing_transforms(r, z, iv)
    strike = torch.exp(r * z)  # normalized strike price
    return F.relu(option_type * (normal.cdf(option_type * d1) - strike * normal.cdf(option_type * d2)))

# 4.4.2 define GNOOptionsDataset
class GNOOptionsDataset(Dataset):
    def __init__(self, options_dataset: OptionsDataset, r_lim: tuple[float, float] = (0.01, 1.),
                 z_lim: tuple[float, float] = (-1.5, .5), subsample=False, mapping: Optional[dict[str, int]] = None):
        """Create dataset for GNO training

        Parameters
        ----------
        options_dataset
            Options dataset
        col_mapping
            dict mapping column names to indices in data tensors
        r_lim
            Limits for (sqrt) time-to-expiry
        z_lim
            Limits for (normalized) log-moneyness
        subsample, optional
            Subsample input on each access, by default False
        """
        self.options_dataset = options_dataset
        self.r_lim = r_lim
        self.z_lim = z_lim
        self.subsample = subsample
        if mapping is None:
            mapping = {
                'time_to_maturity': 0,
                'log_moneyness': 1,
                'implied_volatility': 2,
                'bid': 3,
                'ask': 4,
                'discount_factor': 5,
                'underlying_forward': 6
            }
        self.mapping = mapping

    def __len__(self):
        return len(self.options_dataset)

    def __getitem__(self, i: int) -> Data:
        quote_datetime = self.options_dataset.quote_datetimes[i]
        raw_data = self.options_dataset[i]
        if self.subsample:
            subsample_idx = torch.rand(raw_data.size(1)) <= uniform(0.6, 1.2)
            raw_data = raw_data[:, subsample_idx]

        keys = list(self.mapping.keys())
        t, k = raw_data[keys.index('time_to_maturity')], raw_data[keys.index('log_moneyness')]
        r = t.sqrt()
        z = k / r

        zero_ask_idx = raw_data[keys.index('ask')] > 0
        domain_idx = (self.r_lim[0] <= r) & (r <= self.r_lim[1]) & (self.z_lim[0] <= z) & (z <= self.z_lim[1])
        idx = zero_ask_idx & domain_idx

        r = r[idx, None]
        z = z[idx, None]
        data_dict = dict(zip(self.mapping.keys(), raw_data[:, idx, None]))

        vega = vega_(r, z, data_dict['implied_volatility'])
        weight = torch.maximum(vega / vega.mean(), torch.tensor(1.0))
        return Data(r=r, z=z, **data_dict, vega=vega, weight=weight, num_nodes=r.size(0), quote_datetime=quote_datetime)

# loss function
# 5.1 define Grid (used in loss)
class RectilinearGrid(Module, Mapping):
    """Rectilinear grid with arbitrary number of named axes.
    """
    def __init__(self, **axes: Tensor) -> None:
        super().__init__()
        axes = {k: v.sort()[0] for k, v in axes.items()}
        self._ax_idx = {k: i for i, k in enumerate(axes.keys())}
        self._ax_labels = {i: k for i, k in enumerate(axes.keys())}
        self.register_buffer('_meshgrid', torch.stack(torch.meshgrid(*axes.values(), indexing='ij'), dim=0))

    def __getitem__(self, key) -> Tensor:
        if isinstance(key, str):
            return self._meshgrid[self._ax_idx[key]]
        else:
            return self._meshgrid[key]

    def __iter__(self):
        return iter(self._ax_idx)

    def __len__(self):  # 网格的维度
        return len(self._ax_idx)

    def size(self):
        return self._meshgrid.size()[1:]  # 网格第0维表示坐标轴数量，去掉第0维只返回网格的形状

    def dim(self):
        return self._meshgrid.dim() - 1

    def extra_repr(self) -> str:
        return f"size={tuple(self.size())}"

    def flatten(self, layout: Literal['channel_first', 'channel_last']) -> Tensor:
        flattened = self._meshgrid.flatten(start_dim=1)
        if layout == 'channel_first':
            return flattened
        elif layout == 'channel_last':
            return flattened.transpose(1, 0)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def plot_surface(self, surface: Tensor, x: str | int = 0, y: str | int = 1, title: Optional[str] = None,
                     ax: Optional[plt.Axes] = None, **kwargs) -> list[plt.Artist]:

        surface = surface.squeeze()

        if isinstance(x, str):
            x = self._ax_idx[x]
        if isinstance(y, str):
            y = self._ax_idx[y]

        artists = []
        if ax is None:
            ax = plt.subplot(projection='3d')
        artists.append(ax.plot_surface(self[x].cpu(), self[y].cpu(), surface.cpu(), **kwargs))
        ax.set_xlabel(f'${self._ax_labels[x]}$')
        ax.set_ylabel(f'${self._ax_labels[y]}$')
        if title is not None:
            ax.set_title(title)
        return artists

    def differentiate(self, surface: Tensor, axis: str, order: int = 1) -> list[Tensor]:
        dim = self._ax_idx[axis]  # 网格的维度
        surface_dim = surface.dim() - self.dim() + self._ax_idx[axis]
        h_fw = self[axis].roll(-1, dims=dim) - self[axis]
        h_bw = self[axis] - self[axis].roll(1, dims=dim)

        left_idx = [slice(None)] * len(self)
        left_idx[dim] = 0
        right_idx = [slice(None)] * len(self)
        right_idx[dim] = -1

        diffs = [surface]
        for _ in range(order):
            surface = diffs[-1]
            diff_forward = (surface.roll(-1, dims=surface_dim) - surface) / h_fw
            diff_backward = (surface - surface.roll(1, dims=surface_dim)) / h_bw
            diff = diff_forward / 2 + diff_backward / 2
            diff[..., *left_idx] = diff_forward[..., *left_idx]
            diff[..., *right_idx] = diff_backward[..., *right_idx]
            diffs.append(diff)
        return diffs[1:]

# 5.2 define loss
# 5.2.1 Compare to SVI model
# 5.2.1.1 define Arbitrage, used in SVI and Loss
def butterfly(d1: Tensor, d2: Tensor, iv: Tensor, div_dz: Tensor, div_dzz: Tensor) -> Tensor:
    return (1 + div_dz * d1) * (1 + div_dz * d2) + iv * div_dzz

def calendar(r: Tensor, x: Tensor, iv: Tensor, div_dr: Tensor, div_dz: Tensor) -> Tensor:
    return ((iv - x * div_dz) / r + div_dr) / 2

# 5.2.1.2 define SVI model
@dataclass
class SVI:
    """Implementation of SVI (in its "raw" formulation; see https://arxiv.org/pdf/1204.0646)
    """
    a: float = 0.05
    b: float = 0.1
    rho: float = -.5
    sigma: float = 0.1
    m: float = 0.1

    def __array__(self):
        return np.array(astuple(self))

    def __iter__(self):
        return iter(astuple(self))

    @classmethod
    def implied_variance(cls, z, a: float, b: float, rho: float, sigma: float, m: float, nu=0) -> float:
        root_term = (z - m) ** 2 + sigma ** 2
        if nu == 0:
            return a + b * (rho * (z - m) + np.sqrt(root_term))
        elif nu == 1:
            return b * ((z - m) / np.sqrt(root_term) + rho)
        elif nu == 2:
            return b * sigma ** 2 / root_term ** 1.5
        else:
            raise NotImplementedError(f"Derivative order {nu} not implemented")

    @classmethod
    def implied_volatility(cls, z, *params, nu: int = 0) -> float:
        if nu == 0:
            w = cls.implied_variance(z, nu=0, *params)
            return np.sqrt(np.maximum(w, 0))
        if nu == 1:
            iv = cls.implied_volatility(z, nu=0, *params)
            dw_dz = cls.implied_variance(z, nu=1, *params)
            return dw_dz / (2 * iv)
        if nu == 2:
            iv = cls.implied_volatility(z, nu=0, *params)
            dw_dz = cls.implied_variance(z, nu=1, *params)
            dw_dzz = cls.implied_variance(z, nu=2, *params)
            return dw_dzz / (2 * iv) - (dw_dz ** 2) / (4 * iv ** 3)
        else:
            raise NotImplementedError(f"Derivative order {nu} not implemented")

    def fit(self, data, **kwargs):
        kwargs = kwargs.copy()
        opt_kwargs = self.create_optimization_objective(data)
        kwargs.update(opt_kwargs)
        res = minimize(**kwargs)

        self.a, self.b, self.rho, self.sigma, self.m = res.x

        return self

    def create_optimization_objective(self, data) -> dict:
        r = np.unique(data['r']).item()
        z = data['z']
        iv_target = data['implied_volatility']

        try:
            weight = data['weight']
        except KeyError:
            weight = 1.

        def fun(x):
            iv_predict = SVI.implied_volatility(z, *x)
            error = (iv_target - iv_predict)
            loss = np.sqrt((weight * np.square(error)).mean())
            return loss

        def constraint_fun(x):
            iv = self.implied_volatility(z, *x)
            div_dz = self.implied_volatility(z, *x, nu=1)
            div_dzz = self.implied_volatility(z, *x, nu=2)
            d1, d2 = normalizing_transforms(r, z, iv)
            but = butterfly(d1, d2, iv, div_dz, div_dzz)
            g = np.concatenate((iv, but)) - 1e-4
            return g

        constraints = {'type': 'ineq', 'fun': constraint_fun}

        x0 = np.array(self)
        bounds = list({
                          'a': (None, None),  # ((iv_target * r) ** 2).max()),
                          'b': (0, 1),
                          'rho': (-1, 1),
                          'm': (-1.5, 0.5),
                          'sigma': (1e-8, 2)
                      }.values())

        return {
            'fun': fun,
            'x0': x0,
            'bounds': bounds,
            'constraints': constraints
        }

# define error, used in Loss
def this_spread_error(iv_predict: np.ndarray, data: Mapping[str, np.ndarray]) -> dict[str, float]:
    K = np.exp(np.asarray(data['log_moneyness'])).squeeze()
    S = np.ones_like(K)
    r = np.zeros_like(K)
    t = np.asarray(data['time_to_maturity']).squeeze()
    flag = ['c' if K >= 1 else 'p' for K in K]

    iv = np.asarray(iv_predict).squeeze()
    df = np.asarray(data['discount_factor']).squeeze()
    fw = np.asarray(data['underlying_forward']).squeeze()
    mid_predict = df * fw * vectorized_black_scholes(flag, S, K, t, r, iv, return_as='array')
    mid = np.asarray(((data['bid'] + data['ask']) / 2)).squeeze()
    spread = np.asarray(((data['ask'] - data['bid']) / 2)).squeeze()

    spread_error = np.abs((mid_predict - mid)) / spread  # 误差与市场bid-ask相比较，如果大于1，说明预测iv误差较大
    return spread_error

def this_relative_error(iv_predict: np.ndarray, data: Mapping[str, np.ndarray]) -> dict[str, float]:
    iv_target = data['implied_volatility']
    relative_error = np.abs((iv_predict - iv_target) / iv_target)
    return relative_error

def descriptive_statistics(error: np.ndarray) -> dict[str, float]:
    return {
        'mean': error.mean(),
        'std': error.std(),
        'min': error.min(),
        '%05': np.quantile(error, q=0.05),
        '%25': np.quantile(error, q=0.25),
        '%50': np.quantile(error, q=0.5),
        '%75': np.quantile(error, q=0.75),
        '%95': np.quantile(error, q=0.95),
        'max': error.max(),
    }

# 5.2.3 define error, used in Loss
def pairwise_differences(x: Tensor, y: Tensor) -> Tensor:
    return (x.view(x.size(0), 1, *x.size()[1:]) - y.view(1, y.size(0), *y.size()[1:]))

def BS_generate_edge_index(pos_x: Tensor, pos_y: Tensor, delta_r: float, delta_z: float, subsample_size: int = 50,
                        include_self_loops: bool = False) -> Tensor:
    #pos = torch.cat((pos_x, pos_y), dim=0)
    distances_r = pairwise_differences(pos_x[..., 0], pos_y[..., 0]).abs()
    idx_r = torch.argsort(distances_r, dim=0, stable=True)
    distances_r = torch.gather(distances_r, 0, idx_r)
    idx_bound_r = torch.argmin((distances_r <= 10.0*delta_r).to(dtype=torch.int), dim=0)
    zero_mask = idx_bound_r == 0  # 0 happens if all distances_r <= 10.0*delta_r true or false
    idx_bound_r[zero_mask] = idx_r[subsample_size, zero_mask]

    distances_z = pairwise_differences(pos_x[..., 1], pos_y[..., 1]).abs()
    idx_z = torch.argsort(distances_z, dim=0, stable=True)
    distances_z = torch.gather(distances_z, 0, idx_z)
    idx_bound_z = torch.argmin((distances_z <= 10.0 * delta_z).to(dtype=torch.int), dim=0)
    zero_mask = idx_bound_z == 0
    idx_bound_z[zero_mask] = idx_z[subsample_size, zero_mask]
    #print(idx_bound_r)

    edge_index_list = []
    for i in torch.arange(pos_y.size(0)):

        k_r = idx_bound_r[i]
        step = int(k_r // subsample_size) + 1
        s_r = idx_r[0:k_r:step, i]
        #print(s_r)

        k_z = idx_bound_z[i]
        step = int(k_z // subsample_size) + 1
        s_z = idx_z[0:k_z:step, i]
        #print(s_z)

        s = s_r[torch.isin(s_r, s_z)]    # common regions
        #s = s_r                           # not using common regions
        t = (i+pos_x.size(0)).repeat(s.size())   # pos_x nodes are first indexed

        edge_index_list.append(torch.stack((s, t.to(s.device)), dim=1))
        #if include_self_loops and not (i == s).any() and i < idx_r.size(0):
        #    edge_index_list.append(torch.tensor([[i, i]], device=s.device, dtype=s.dtype))

    edge_index = torch.cat(edge_index_list).transpose(0, 1).contiguous()

    return edge_index

# define slice_data, used in Loss function
def slice_data(axis: Tensor, *features: Tensor):
    """Perform group-by operation on features based on axis

    Parameters
    ----------
    axis
        Axis to group by
    features
        Features to group

    Returns
    -------
    Tuple[Tensor, List[Tensor]]
        Unique values of axis and list of features grouped by axis
    """
    axis = axis.squeeze()
    features = torch.stack([axis] + [d.squeeze() for d in features])
    slices = []
    axis_uniques, inverse = axis.unique(return_inverse=True)
    for i, r in enumerate(axis_uniques):
        slices.append(features[..., inverse == i])

    return axis_uniques, slices

# define detailed Trainer function (i.e. Loss function)
@dataclass
class Loss:
    lim_r: tuple[float, float] = (0.2, 1.5) #(0.01, 1.0)
    lim_z: tuple[float, float] = (-1.3, 0.5)#(-1.5, 0.5)
    B_lim_r: tuple[float, float] = (0.05, 1.55)#(0.0, 1.5)#(0.1, 1.4) #(0.15, 1.55) #(-0.04, 1.05)
    B_lim_z: tuple[float, float] = (-1.5, 0.55)#(-1.5, 0.5)#(-1.4, 0.6)#(-1.35, 0.55)#(-1.55, 0.55)
    # Use for but and cal grids
    step_r: Optional[float] = None  #0.05  # 0.03 is okay
    step_z: Optional[float] = None  #0.01
    # B-Spline grids
    BS_Steps_r: int = 30
    BS_Steps_z: int = 50
    BS_pos: Tensor = None
    useExactDerivative: bool = True    # Use B-Spline true derivatives
    error_weights: dict[str, float] = field(
        default_factory=lambda: {'fit': 1., 'but': 10., 'cal': 10., 'reg_z': 0.1, 'reg_r': 0.1})

    extreme_z_bounds: tuple[float, float] = (-1.0, 0.4)
    extreme_z_weight: float = 5.0  
    
    eps_but: float = 1e-3
    eps_cal: float = 1e-3
    subsample_size: int = 50
    # Training setting
    lr: float = 1e-4
    weight_decay: float = 1e-5

    def load_input(self, data, subsample_size: Optional[int] = None, radius: Optional[float] = None,
                   step_r: Optional[float] = None, step_z: Optional[float] = None):

        if subsample_size is None:
            subsample_size = self.subsample_size

        if step_r is None:
            step_r = self.step_r

        if step_z is None:
            step_z = self.step_z

        pos_x = torch.cat((data['r'], data['z']), dim=1)

        # Generate grid for Butterfly arbitrage:
        #r_axis = torch.arange(*self.lim_r, uniform(0.075, 0.125) if step_r is None else step_r)
        r_axis = torch.arange(*self.lim_r, uniform(0.01, 0.125) if step_r is None else step_r)
        z_axis = torch.arange(*self.lim_z, 0.01 if step_z is None else step_z)
        grid = RectilinearGrid(r=r_axis, z=z_axis).to(pos_x.device)
        #grids.append(grid)
        # Grids for Calendar arbitrage :  We no longer need this as we have a function surface

        # Assemble GNO input and auxiliary data
        BS_delta_r = (self.B_lim_r[1] - self.B_lim_r[0]) / self.BS_Steps_r
        BS_delta_z = (self.B_lim_z[1] - self.B_lim_z[0]) / self.BS_Steps_z
        if self.BS_pos is None:
            BS_r_values = torch.linspace(self.B_lim_r[0], self.B_lim_r[1], self.BS_Steps_r + 1)
            BS_z_values = torch.linspace(self.B_lim_z[0], self.B_lim_z[1], self.BS_Steps_z + 1)
            BS_r_values1 = torch.cat([torch.tensor([self.B_lim_r[0] - BS_delta_r]), BS_r_values,
                                      torch.tensor([self.B_lim_r[1] + BS_delta_r])])
            BS_z_values1 = torch.cat([torch.tensor([self.B_lim_z[0] - BS_delta_z]), BS_z_values,
                                      torch.tensor([self.B_lim_z[1] + BS_delta_z])])
            X, Y = torch.meshgrid(BS_r_values1, BS_z_values1, indexing='ij')
            self.BS_pos = torch.stack([X.flatten(), Y.flatten()],
                                      dim=1)  # in pattern[[x1, y1],[x1,y2],...,[x1,yn],[x2,y1], ...,[x2,yn], ..., [xm,yn]]

        edge_index = BS_generate_edge_index(pos_x, self.BS_pos, delta_r=BS_delta_r, delta_z = BS_delta_z, subsample_size=subsample_size)

        # BS_x is different for each data
        BS_x1 = B1_Spline13_Scatter(pos_x, self.B_lim_r[0], self.B_lim_r[1], self.B_lim_z[0], self.B_lim_z[1],
                                    self.BS_Steps_r, self.BS_Steps_z)
        BS_x2 = B2_Spline13_Scatter(pos_x, self.B_lim_r[0], self.B_lim_r[1], self.B_lim_z[0], self.B_lim_z[1],
                                    self.BS_Steps_r, self.BS_Steps_z)

        BS_x = torch.stack([torch.from_numpy(BS_x1), torch.from_numpy(BS_x2)], dim=-1)

        BS_x1 = np.transpose(BS_x1, (2, 0, 1)).reshape(BS_x1.shape[2], -1)
        BS_x2 = np.transpose(BS_x2, (2, 0, 1)).reshape(BS_x2.shape[2], -1)
        # Initial B-Spline Coefficients
        #ridge = Ridge(alpha=1.0, fit_intercept=False)  # Set fit_intercept=False to match manual implementation
        #ridge.fit(np.hstack([BS_x1, BS_x2]), data['implied_volatility'])
        #BS_coeffs = ridge.coef_.T
        # ======================================================
        # 1) 普通数据点的设计矩阵 X_data, y_data
        X_data = np.hstack([BS_x1, BS_x2])           # shape (N, M)
        y_data = data['implied_volatility'].squeeze().cpu().numpy()         # shape (N,)
        # 2) 端点二阶导≈0 的“伪观测”矩阵 A_eq, b_eq
        #    （我们在网格上拿出 r = r_min 和 r = r_max 两条边）
        #    grid.flatten('channel_last') 得到 (n_r * n_z, 2) 的 (r,z) 对儿
        grid_pts = grid.flatten('channel_last').cpu().numpy()  # shape (n_r*n_z, 2)
        #    计算这 n_r*n_z 个点上的 d²B/dr² 设计矩阵
        B1_drr = B1_Spline13_Scatter_dxx(grid_pts, *self.B_lim_r, *self.B_lim_z, self.BS_Steps_r, self.BS_Steps_z)# shape (n_r*n_z, BS_Steps_r+3, BS_Steps_z+3)
        B2_drr = B2_Spline13_Scatter_dxx(grid_pts, *self.B_lim_r, *self.B_lim_z, self.BS_Steps_r, self.BS_Steps_z)
        # flatten 成 (n_pts, M))
        n_pts = B1_drr.shape[2]
        B1_mat = B1_drr.transpose(2, 0, 1).reshape(n_pts, -1) # (n_pts, (m+2)*(n+2))
        B2_mat = B2_drr.transpose(2, 0, 1).reshape(n_pts, -1)
        A_drr  = np.hstack([B1_mat, B2_mat]) 
        # 找出 r = r_min 和 r = r_max 的行
        r_vals = grid['r'].flatten().cpu().numpy()
        mask_low  = np.isclose(r_vals, self.B_lim_r[0])
        mask_high = np.isclose(r_vals, self.B_lim_r[1])
        A_low  = A_drr[mask_low]
        A_high = A_drr[mask_high]
        A_eq = np.vstack([A_low, A_high])     # shape (2*n_z, M)
        b_eq = np.zeros(A_eq.shape[0])
        # 3) 拼接成带“大权重”伪观测的增强矩阵
        w_bc = 1e3   # 约束权重大一点
        X_aug = np.vstack([ X_data, w_bc * A_eq ])       # (N_data+2n_z, M)
        y_aug = np.concatenate([ y_data,   w_bc * b_eq ]) # (N_data+2n_z,)
 
        # 4) Ridge 拟合
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(X_aug, y_aug)
        BS_coeffs = ridge.coef_.T  # shape (M,)
       # ======================================================
        A1, A2 = np.split(BS_coeffs, 2)  # Each of size (504,)
        A1 = A1.reshape(self.BS_Steps_r + 3, self.BS_Steps_z + 3)
        A2 = A2.reshape(self.BS_Steps_r + 3, self.BS_Steps_z + 3)
        BS_coeffs = torch.from_numpy(np.stack([A1, A2], axis=-1)).to(torch.float32)

        # Calculate for each training data as they are different
        B_grids_data = torch.stack([torch.from_numpy(B1_Spline13_Scatter(grid.flatten('channel_last'),
                                                                          self.B_lim_r[0], self.B_lim_r[1],
                                                                          self.B_lim_z[0], self.B_lim_z[1],
                                                                          self.BS_Steps_r,
                                                                          self.BS_Steps_z)),
                                     torch.from_numpy(B2_Spline13_Scatter(grid.flatten('channel_last'),
                                                                          self.B_lim_r[0], self.B_lim_r[1],
                                                                          self.B_lim_z[0], self.B_lim_z[1],
                                                                          self.BS_Steps_r,
                                                                          self.BS_Steps_z))
                                     ], dim=-1)

        B_grids_dr = None
        B_grids_drr = None
        B_grids_dz = None
        B_grids_dzz = None
        if self.useExactDerivative:
             B_grids_dr = torch.stack([torch.from_numpy(B1_Spline13_Scatter_dx(grid.flatten('channel_last'),
                                                                                 self.B_lim_r[0], self.B_lim_r[1],
                                                                                 self.B_lim_z[0], self.B_lim_z[1],
                                                                                 self.BS_Steps_r,
                                                                                 self.BS_Steps_z)),
                                        torch.from_numpy(B2_Spline13_Scatter_dx(grid.flatten('channel_last'),
                                                                                 self.B_lim_r[0], self.B_lim_r[1],
                                                                                 self.B_lim_z[0], self.B_lim_z[1],
                                                                                 self.BS_Steps_r,
                                                                                 self.BS_Steps_z))
                                        ], dim=-1)
             B_grids_drr = torch.stack([torch.from_numpy(B1_Spline13_Scatter_dxx(grid.flatten('channel_last'),
                                                                          self.B_lim_r[0], self.B_lim_r[1],
                                                                          self.B_lim_z[0], self.B_lim_z[1],
                                                                          self.BS_Steps_r,
                                                                          self.BS_Steps_z)),
                                     torch.from_numpy(B2_Spline13_Scatter_dxx(grid.flatten('channel_last'),
                                                                          self.B_lim_r[0], self.B_lim_r[1],
                                                                          self.B_lim_z[0], self.B_lim_z[1],
                                                                          self.BS_Steps_r,
                                                                          self.BS_Steps_z))
                                     ], dim=-1)

             B_grids_dz = torch.stack([torch.from_numpy(B1_Spline13_Scatter_dy(grid.flatten('channel_last'),
                                                                             self.B_lim_r[0], self.B_lim_r[1],
                                                                             self.B_lim_z[0], self.B_lim_z[1],
                                                                             self.BS_Steps_r,
                                                                             self.BS_Steps_z)),
                                    torch.from_numpy(B2_Spline13_Scatter_dy(grid.flatten('channel_last'),
                                                                             self.B_lim_r[0], self.B_lim_r[1],
                                                                             self.B_lim_z[0], self.B_lim_z[1],
                                                                             self.BS_Steps_r,
                                                                             self.BS_Steps_z))
                                    ], dim=-1)
             B_grids_dzz = torch.stack([torch.from_numpy(B1_Spline13_Scatter_dyy(grid.flatten('channel_last'),
                                                                             self.B_lim_r[0], self.B_lim_r[1],
                                                                             self.B_lim_z[0], self.B_lim_z[1],
                                                                             self.BS_Steps_r,
                                                                             self.BS_Steps_z)),
                                    torch.from_numpy(B2_Spline13_Scatter_dyy(grid.flatten('channel_last'),
                                                                             self.B_lim_r[0], self.B_lim_r[1],
                                                                             self.B_lim_z[0], self.B_lim_z[1],
                                                                             self.BS_Steps_r,
                                                                             self.BS_Steps_z))
                                    ], dim=-1)

        input = TensorDict(**{
            'x': data['implied_volatility'],
            'pos_x': pos_x,
            'pos_y': self.BS_pos,
            'edge_index': edge_index,
            'preB_coeffs': BS_coeffs,
        })

        aux = {
            'grids': grid,
        }

        return input, aux, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz

    @classmethod
    def read_output(cls, output, aux):
        iv_x, iv_y = output
        grids = aux['grids']
        sections = [np.prod(grid.size()) for grid in grids]
        return (iv_x, *(iv.view(grids[i].size()) for i, iv in enumerate(torch.split(iv_y, sections, dim=0))))

    @classmethod
    def replication_error(cls, iv_target: Tensor, iv_predict: Tensor) -> Tensor:
        error = (iv_predict - iv_target) / iv_target
        return error

    @classmethod
    def BS_butterfly_term(cls, grid: RectilinearGrid, iv_surface: Tensor, div_drr: Tensor, div_dz: Tensor, div_dzz: Tensor,
                       include_second_derivatives: bool = False) -> Tensor:
        iv_surface = iv_surface.view(grid.size())
        if div_drr is None:
            div_dz, div_dzz = grid.differentiate(iv_surface, 'z', order=2)
            _, div_drr = grid.differentiate(iv_surface, 'r', order=2)
        d1, d2 = normalizing_transforms(**grid, iv=iv_surface.clamp(min=0.005))
        but = butterfly(d1, d2, iv_surface, div_dz, div_dzz)

        if not include_second_derivatives:
            return but
        else:
            return but, div_dzz, div_drr

    @classmethod
    def BS_calendar_term(cls, grid: RectilinearGrid, iv_grid: Tensor, iv_grid_dr: Tensor, iv_grid_dz: Tensor) -> Tensor:
        iv_grid_dr = iv_grid_dr.view(grid.size())
        iv_grid_dz = iv_grid_dz.view(grid.size())
        iv_grid = iv_grid.view(grid.size())
        #((iv - x * div_dz) / r + div_dr) / 2
        return ((iv_grid - grid['z'] * iv_grid_dz) / grid['r'] + iv_grid_dr) /2.0
        #return grid['r']*iv_grid_dr - grid['z']*iv_grid_dz + iv_grid # grid['r'][:-1] / grid['r'][1:]

    #@classmethod
    def errors(self, data, BS_x, BS_grids_data, BS_grids_dr, BS_grids_drr, BS_grids_dz, BS_grids_dzz, output, aux):
        grid = aux['grids']

        if BS_grids_drr is not None:
            BS_grid_dr = torch.einsum('ijk,ijmk->m', output, BS_grids_dr.to(torch.float32)).reshape(grid.size())
            BS_grid_drr = torch.einsum('ijk,ijmk->m', output, BS_grids_drr.to(torch.float32)).reshape(grid.size())
            BS_grid_dz = torch.einsum('ijk,ijmk->m', output, BS_grids_dz.to(torch.float32)).reshape(grid.size())
            BS_grid_dzz = torch.einsum('ijk,ijmk->m', output, BS_grids_dzz.to(torch.float32)).reshape(grid.size())
        else:
            BS_grid_dr = None
            BS_grid_drr = None
            BS_grid_dz = None
            BS_grid_dzz = None
        iv_predict = torch.einsum('ijk,ijmk->m', output, BS_x.to(torch.float32)).view(-1, 1)
        replication_error = self.replication_error(data['implied_volatility'], iv_predict)
        iv_but = torch.einsum('ijk,ijmk->m', output, BS_grids_data.to(torch.float32)).view(-1, 1)

        butterfly_error, div_dzz, div_drr = self.BS_butterfly_term(grid, iv_but, BS_grid_drr, BS_grid_dz, BS_grid_dzz, include_second_derivatives=True)

        calendar_error = self.BS_calendar_term(grid, iv_but, BS_grid_dr, BS_grid_dz)

        return replication_error, butterfly_error, calendar_error, div_dzz, div_drr

    def loss(self, data, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz, output, aux, dev=True):
        replication_error, butterfly_error, calendar_error, div_dzz, div_drr = self.errors(data, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz, output, aux)

        z = data['z'].squeeze()
        low_z, high_z = self.extreme_z_bounds
        mask_z_extreme = (z < low_z) | (z > high_z)
        w_z = torch.ones_like(z)
        w_z[mask_z_extreme] = self.extreme_z_weight
        w_vega = data['weight'].squeeze()
        w_total = w_vega * w_z

        losses = {
            'fit': (w_total * replication_error.squeeze().square()).mean().sqrt(),
            'but': F.relu(-butterfly_error - self.eps_but).mean(),
            'cal': F.relu(-calendar_error - self.eps_cal).mean(),
            'reg_z': div_dzz.square().mean().sqrt(),
            'reg_r': div_drr.square().mean().sqrt(),
        }
        losses_scalar = {k: v.item() for k, v in losses.items()}
        l = sum([weight * losses[key] for key, weight in self.error_weights.items()])

        if not dev:
            return l
        else:
            with torch.no_grad():
                mape = replication_error.abs().mean()
                weighted_mape = (data['weight'] * replication_error.abs()).mean()

            return l, {'loss': l, 'mape': mape, 'wmape': weighted_mape} | losses


    def compute_batch_loss(self, model: Module, batch: list[tuple[Data, TensorDict, TensorDict]], callback: callable,
                           device: torch.device = None):

        batch_loss = 0
        loss_infos = []

        batch_size = len(batch)
        for data, input, aux, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz in batch:

            data = data.to(device)
            B_grids_data = B_grids_data.to(device)
            if B_grids_drr is not None:
                B_grids_dr = B_grids_dr.to(device)
                B_grids_drr = B_grids_drr.to(device)
                B_grids_dz = B_grids_dz.to(device)
                B_grids_dzz = B_grids_dzz.to(device)
            input = input.to(device)
            BS_x = BS_x.to(device)
            aux['grids'] = aux['grids'].to(device)

            output = model(**input)

            sample_loss, sample_loss_info = self.loss(data, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz, output, aux)

            sample_loss = sample_loss / batch_size
            callback(sample_loss)

            batch_loss = batch_loss + sample_loss
            loss_infos.append(sample_loss_info)

        return batch_loss, loss_infos

    def evaluate(self, model: Module, dataset: GNOOptionsDataset, device: torch.device = None, return_data=False,
                 **kwargs):

        kwargs = kwargs.copy()
        storedir = kwargs.pop('storedir', None)
        logger = kwargs.pop('logger', None)

        data_storedir = None
        if storedir is not None:
            data_storedir = f"{storedir}/data"
            Path(data_storedir).mkdir(exist_ok=True)

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=self.collate_fn, shuffle=True, pin_memory=False, worker_init_fn=worker_init_fn,
                                **kwargs)

        model = model.eval()

        rows_val = []
        rows_rel = []
        rows_fit = []
        data_list = []
        for data, input, aux, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz in dataloader:
            data = data.to(device)
            B_grids_data = B_grids_data.to(device)
            if B_grids_drr is not None:
                B_grids_dr = B_grids_dr.to(device)
                B_grids_drr = B_grids_drr.to(device)
                B_grids_dz = B_grids_dz.to(device)
                B_grids_dzz = B_grids_dzz.to(device)

            input = input.to(device)
            BS_x = BS_x.to(device)
            aux['grids'] = aux['grids'].to(device)

            with torch.no_grad():
                output = model(**input)   #, preB_coeffs=BS_coeffs)

            l, losses = self.loss(data, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz, output, aux)
            rows_val.append(
                {'quote_datetime': data.quote_datetime} | {'loss': l.item()} | {key: loss.item() for key, loss in
                                                                                losses.items()})

            iv_predict = torch.einsum('ijk,ijmk->m', output, BS_x.to(torch.float32)).view(-1, 1)
            iv_surface = torch.einsum('ijk,ijmk->m', output, B_grids_data.to(torch.float32)).view(-1, 1)

            data_dict = {key: val.to('cpu', dtype=torch.float64).numpy() for key, val in data.items() if
                         torch.is_tensor(val)}
            relative_error = this_relative_error(iv_predict.to('cpu', dtype=torch.float64).numpy(), data_dict)
            spread_error = this_spread_error(iv_predict.to('cpu', dtype=torch.float64).numpy(), data_dict)
            rows_rel.append({'quote_datetime': data.quote_datetime} | descriptive_statistics(relative_error))
            rows_fit.append({'quote_datetime': data.quote_datetime} | descriptive_statistics(spread_error))

            grid: RectilinearGrid = aux['grids']
            iv_surface = iv_surface.view(grid.size())
            if self.useExactDerivative:
                div_dz = torch.einsum('ijk,ijmk->m', output, B_grids_dz.to(torch.float32)).reshape(
                    grid.size())
                div_dzz = torch.einsum('ijk,ijmk->m', output, B_grids_dzz.to(torch.float32)).reshape(
                    grid.size())
            else:
                div_dz, div_dzz = grid.differentiate(iv_surface, 'z', order=2)
            d1, d2 = normalizing_transforms(**grid, iv=iv_surface.clamp(min=0.001))
            g = butterfly(d1, d2, iv_surface, div_dz, div_dzz)

            data.iv_predict = iv_predict
            data.iv_surface = iv_surface
            data.normalized_spread = (data['ask'] - data['bid']) / (
                        data['underlying_forward'] * data['discount_factor'])
            data.implied_density = normal.log_prob(-d2).exp() * g / (iv_surface * grid['r'])
            data.replication_error, data.butterfly_error, data.calendar_error, data.div_dzz, data.div_drr = self.errors(
                data, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz, output, aux)
            data.grid = grid
            data_list.append(data)

            if storedir is not None:
                filepath = f"{data_storedir}/data_{data.quote_datetime.strftime('%Y-%m-%d-%H-%M-%S')}.pt"
                torch.save(data.cpu(), filepath)

            if logger is not None:
                logger.info(f"Evaluated quote datetime {data.quote_datetime}")

        df_val = pd.DataFrame(rows_val).set_index('quote_datetime').sort_index()
        df_rel = pd.DataFrame(rows_rel).set_index('quote_datetime').sort_index()
        df_fit = pd.DataFrame(rows_fit).set_index('quote_datetime').sort_index()

        if storedir is not None:
            start, end = df_val.index[0].strftime('%Y-%m-%d'), df_val.index[-1].strftime('%Y-%m-%d')
            df_val.to_csv(f"{storedir}/val_{start}-{end}.csv")
            df_rel.to_csv(f"{storedir}/rel_{start}-{end}.csv")
            df_fit.to_csv(f"{storedir}/fit_{start}-{end}.csv")

        if not return_data:
            return df_val, df_rel, df_fit
        else:
            return (df_val, df_rel, df_fit), data_list

    def collate_fn(self, data_list):
        data = data_list[0]
        return (data, *self.load_input(data))

    @staticmethod
    def format_loss_str(loss_infos):
        batch_size = len(loss_infos)
        loss_details = {k: [info[k] for info in loss_infos] for k in loss_infos[0]}
        loss_str = [
            f"loss: {sum(loss_details['loss']) / batch_size : .8f}",
            f"(mape: {sum(loss_details['mape']) / batch_size :> 8.3g}",
            f"wmape: {sum(loss_details['wmape']) / batch_size :> 8.3g}",
            f"fit: {sum(loss_details['fit']) / batch_size :> 8.3g}",
            f"cal: {sum(loss_details['cal']) / batch_size :> 8.3g}",
            f"but: {sum(loss_details['but']) / batch_size :> 8.3g}",
            f"reg_z: {sum(loss_details['reg_z']) / batch_size :> 8.3g}",
            f"reg_r: {sum(loss_details['reg_r']) / batch_size :> 8.3g})"
        ]
        return ', '.join(loss_str)

    def plot_example(self, data: Data, output: Tensor, BS_x: Tensor, step: int = 3, **kwargs):

        figsize = kwargs.get('figsize', (9, 14))

        grid = RectilinearGrid(r=data.r.unique(), z=torch.arange(-1.5, .5, 0.01))
        iv_predict = torch.einsum('ijk,ijmk->m', output, BS_x.to(torch.float32)).view(-1, 1)
        Grid_data = torch.stack([torch.from_numpy(B1_Spline13_Scatter(grid.flatten('channel_last'),
                                                                          self.B_lim_r[0], self.B_lim_r[1],
                                                                          self.B_lim_z[0], self.B_lim_z[1],
                                                                          self.BS_Steps_r,
                                                                          self.BS_Steps_z)),
                                     torch.from_numpy(B2_Spline13_Scatter(grid.flatten('channel_last'),
                                                                          self.B_lim_r[0], self.B_lim_r[1],
                                                                          self.B_lim_z[0], self.B_lim_z[1],
                                                                          self.BS_Steps_r,
                                                                          self.BS_Steps_z))
                                     ], dim=-1) #.to(data.r.device)
        iv_gno = torch.einsum('ijk,ijmk->m', output, Grid_data.to(torch.float32)).view(grid.size())
        expiries, slices = slice_data(data['r'], data['z'], data['implied_volatility'], iv_predict, data['vega'])
        ncols = floor(len(expiries) ** .5)
        nrows = ceil(len(expiries) / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        for i, ax in zip_longest(range(len(expiries)), chain(*axs)):
            if i is None:
                fig.delaxes(ax)
            else:
                r, z, iv_target, iv_predict, weight = slices[i]
                svi = SVI().fit(
                    {'r': r.numpy(), 'z': z.numpy(), 'implied_volatility': iv_target.numpy(), 'weight': weight.numpy()})
                #iv_svi = SVI.implied_volatility(z, *svi)
                z_plot = np.arange(-1.5, .5, 0.01)
                iv_svi = SVI.implied_volatility(z_plot, *svi)

                ax.scatter(z[::step], iv_target[::step], c='b', alpha=.5, s=8, marker='+', label='Mkt')
                ax.plot(z_plot, iv_svi, c='orange', alpha=.5, label='SVI')
                ax.plot(z_plot, iv_gno[i], c='g', alpha=.5, label='OpDS')
                ax.set_title(rf"$\tau={r[0] ** 2:.3f}$")
                ax.set_xlabel(r"$z = k / \sqrt{\tau}$")
                ax.legend()
                ax.grid()
                ax.set_aspect('auto')

        for col in range(ncols):
            last_ax = None
            for row in range(nrows):
                # check if matplotlib ax has been deleted:
                if not repr(axs[row, col]) == '<Axes: >':
                    last_ax = axs[row, col]
                else:
                    last_ax.xaxis.set_tick_params(labelbottom=True)
                    break

        return fig, axs


# Train and test
def split_dataset(dataset: OptionsDataset):
    """Splits dataset in <2021 training portion, and 2020 test portion (sub-split into months)"""
    train_indices = []
    val_indices = [[] for _ in range(12)]

    for idx, file_path in enumerate(dataset.file_paths):
        date_str = str(file_path).split('_')[-1].replace('.pt', '')
        date = datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
        if date.year < 2021:
            train_indices.append(idx)
        elif date.year == 2021:
            month = date.month - 1
            val_indices[month].append(idx)
    train_dataset, dev_dataset = random_split(Subset(GNOOptionsDataset(dataset), train_indices), [0.982, 0.018])
    print("length of train_dataset and dev_dataset", len(train_dataset), len(dev_dataset))
    test_datasets = [Subset(GNOOptionsDataset(dataset), indices) for indices in val_indices]
    return train_dataset, dev_dataset, test_datasets


def BS_load_checkpoint(model:BS_GNO, path: str, device=None):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def chunked(sequence: Sequence, chunk_size: int) -> list:
    """Chunk a sequence into chunks of size `chunk_size`.
    Parameters
    ----------
    sequence：Sequence to chunk
    chunk_size：Size of chunks
    Returns
    -------
    List：List of chunks
    """
    return [sequence[pos:pos + chunk_size] for pos in range(0, len(sequence), chunk_size)]

def train_model(model, optimizer, train_dataset: GNOOptionsDataset, dev_dataset: GNOOptionsDataset, ** kwargs):
    logger = logging.getLogger('job')
    logger.info("Starting training phase...")
    train_losses = []
    val_losses = []
    val_epochs   = []
    
    kwargs = kwargs.copy()
    num_workers = kwargs.pop('num_workers', 4)
    epochs = kwargs.pop('epochs', 200)
    start_epoch = kwargs.pop('start_epoch', 0)
    batch_size = kwargs.pop('batch_size', 64)
    device = kwargs.pop('device', next(model.parameters()).device)
    BS_Steps_r = kwargs.pop('BS_Steps_r', 20)
    BS_Steps_z = kwargs.pop('BS_Steps_z', 30)
    step_r = kwargs.pop('step_r', 0.05)
    step_z = kwargs.pop('step_z', 0.01)

    storedir = kwargs.pop('storedir', None)
    checkpoint_storedir = kwargs.pop('checkpoint_storedir', None)

    loss = kwargs.pop("train_loss", Loss(BS_Steps_r=BS_Steps_r, BS_Steps_z=BS_Steps_z))
    dev_loss = kwargs.pop('dev_loss', Loss(BS_Steps_r=BS_Steps_r, BS_Steps_z=BS_Steps_z, step_r=step_r, step_z=step_z))
    callback = kwargs.pop('callback', lambda sample_loss: sample_loss.backward())

    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=loss.collate_fn, shuffle=True,
                                      num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)

    best_val_loss = float('inf')
    best_model_state = None

    epochs_no_improve = 0
    patience = 4 

    for epoch in range(start_epoch, epochs):
        model.train()
        train_iterator = iter(train_loader)
        running_train_loss = 0.0
        for batch_idx in (iterations := tqdm(chunked(list(range(len(train_loader))), batch_size))):
            batch = [next(train_iterator) for _ in batch_idx]
            optimizer.zero_grad()
            batch_loss, loss_infos = loss.compute_batch_loss(model, batch, callback, device)
            loss_str = loss.format_loss_str(loss_infos)
            iterations.write('\n'+loss_str)
            optimizer.step()
            running_train_loss += batch_loss.item()
            if (iterations.n % 10 == 0):  # and (storedir is not None):
                logger.info(f"\nEpoch {epoch}; {iterations.n}/{len(iterations)} -- {loss_str}")
        train_losses.append(running_train_loss / len(train_loader))
        
        # Dev loss
        if ((epoch + 1) % 3 == 0):
            df_val, df_rel, df_fit = dev_loss.evaluate(model, dev_dataset, device=device, num_workers=num_workers, storedir=storedir, logger = logger)
            val_loss = df_val['loss'].mean()
            val_losses.append(val_loss)
            val_epochs.append(epoch)

            logger.info(f"Epoch {epoch} Dev: {df_val.describe()}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
                logger.info(f"New best model found at epoch {epoch} with val_loss {val_loss:.6f}")

                if checkpoint_storedir:
                    torch.save({
                        'epoch':epoch,
                        'model': best_model_state,
                        'optimizer': optimizer.state_dict()
                    }, f"{checkpoint_storedir}/fullB_best_model_new.pt")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
                break

    # Load best model for testing
    if best_model_state:
        model.load_state_dict(best_model_state)
    logger.info("Training complete.")
    return model, train_losses, val_losses, val_epochs

def evaluate_model_on_testsets(model, test_datasets, **kwargs):
    logger = kwargs.pop('logger', logging.getLogger('job'))
    device = kwargs.pop('device')
    dev_loss = kwargs.pop('dev_loss')
    num_workers = kwargs.pop('num_workers', 4)
    storedir = kwargs.pop('storedir', None)

    logger.info("Starting evaluation phase on test sets...")

    all_dfs = []
    for k, test_dataset in enumerate(test_datasets):
        logger.info(f"Evaluating on test dataset #{k}")
        if len(test_dataset) > 0:
            (df_val, df_rel, df_fit), data_list = dev_loss.evaluate(model, test_dataset,
                                                              device=device,
                                                              num_workers=num_workers,
                                                              storedir=storedir,
                                                              logger=logger,
                                                              return_data=True)

            for data in data_list:
                df = pd.DataFrame({
                    'quote_datetime': [data.quote_datetime] * len(data['r']),
                    'r': data['r'].squeeze().cpu().numpy(),
                    'z': data['z'].squeeze().cpu().numpy(),
                    'iv_target': data['implied_volatility'].squeeze().cpu().numpy(),
                    'iv_predict': data.iv_predict.squeeze().cpu().numpy(),
                })
                all_dfs.append(df)

    df_all = pd.concat(all_dfs)
    if storedir is not None:
        df_all.to_csv(Path(storedir) / "test_iv_predictions.csv", index=False)

    logger.info("Evaluation on all test sets complete.")



if __name__ == "__main__":
    storedir = "./CHECKPOINTS_NOpreB/"  # Set this to persist evaluation results/checkpoints
    if storedir is not None:
        checkpoint_storedir = f"{storedir}/checkpoints"
        Path(checkpoint_storedir).mkdir(exist_ok=True)

        data_storedir = f"{storedir}/data"
        Path(data_storedir).mkdir(exist_ok=True)
    else:
        checkpoint_storedir = None
        data_storedir = None

    try:
        job_id = os.environ['PBS_JOBID'].split('.pbs')[0]
    except KeyError:
        job_id = 'local'

    logging.basicConfig()
    logger = logging.getLogger('job')
    logger.setLevel(logging.INFO)

    logger.info(f"Defining device (torch.cuda.is_available()={torch.cuda.is_available()})")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f'Running using device `{device}`')

    if device.type == 'cuda':
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        formatted_result = str(result.stdout).replace('\\n', '\n').replace('\\t', '\t')  ##

        logger.info(formatted_result)
        logger.info(f'Device count: {torch.cuda.device_count()}')
        logger.info(f'Visible devices count: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    os.environ['OPDS_CACHE_DIR'] = os.path.expanduser('~/.cache/opds')  # directory where to place the processed files
    os.environ['OPDS_WRDS_DATA_DIR'] = os.path.abspath("volatility_smoothing/data/wrds/spx")  # <- .csv file from WRDS should be place inside this directory

    dataset = WRDSOptionsDataset(force_reprocess=False)    # If you data is not in ~/.cache/opds  You may set to True
    train_dataset, dev_dataset, test_datasets = split_dataset(dataset)
    # dev_dataset = train_dataset
    lr = 1e-4
    weight_decay = 1e-5
    num_workers = 8  #0

    epochs = 300  # Finetune epochs, set to 0 to skip and just evaluate
    batch_size = 64  # Finetune batch size, will be augmented by same amount of training data

    num_hidden_layers = 2#3
    gno_channels = 16
    fnn_hidden_channels = 64
    
    
    # mesh sizes on which to evaluate arbitrage metrics
    step_r = 0.05     # This is only used for testing
    step_z = 0.01     # This is only used for testing
    BS_Steps_r = 8   # You may try 15   or 30,20
    BS_Steps_z = 15   # You may try 15   or 40,30

    train_loss = Loss(BS_Steps_r=BS_Steps_r, BS_Steps_z=BS_Steps_z)
    dev_loss = Loss(BS_Steps_r=BS_Steps_r, BS_Steps_z=BS_Steps_z, step_r=step_r, step_z=step_z)

    torch.cuda.empty_cache()
    logger.info(50 * "=")
    logger.info(f"Evaluation start (Retraining epochs: {epochs}).")
    logger.info(50 * "=")

    #finetune_dataset = Subset(train_dataset, [])
    try:
        model = create_BS_gno(BS_Steps_r = BS_Steps_r, BS_Steps_z = BS_Steps_z,
                                  num_hidden_layers = num_hidden_layers,
                                  gno_channels=gno_channels,fnn_hidden_channels = fnn_hidden_channels).to(device)
        if not SKIP_TRAINING:     
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            start_epoch = 0
#            checkpoint_path = f"{checkpoint_storedir}/fullB_best_model.pt"
#            if os.path.exists(checkpoint_path):
#                 cp = torch.load(checkpoint_path, map_location=device)
#                 model.load_state_dict(cp['model'])
#                 optimizer.load_state_dict(cp['optimizer'])
#                 start_epoch = cp.get('epoch', 0) + 1
#                 logger.info(f"Resuming training from epoch {start_epoch}")
            model, train_losses, val_losses, val_epochs = train_model(model, optimizer,
                            train_dataset=train_dataset,
                            dev_dataset=dev_dataset,
                            train_loss=train_loss,
                            dev_loss=dev_loss,
                            start_epoch=start_epoch,
                            epochs=epochs,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            checkpoint_storedir=checkpoint_storedir,
                            step_r=step_r, step_z=step_z,
                            BS_Steps_r=BS_Steps_r, BS_Steps_z = BS_Steps_z)
            plt.figure(figsize=(8,5))
            train_epochs = list(range(start_epoch, start_epoch + len(train_losses)))
            plt.plot(train_epochs, train_losses,      label='Train Loss')
            plt.plot(val_epochs,   val_losses, 'o--', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training & Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{checkpoint_storedir}/loss_curve.png", bbox_inches='tight')
            plt.close()
        else:
            # if skip training，就从 checkpoint 加载模型
            checkpoint_path = f"{checkpoint_storedir}/fullB_best_model_new.pt"
            model, _ = BS_load_checkpoint(model, checkpoint_path, device=device)
            model.eval()
            

    except KeyboardInterrupt:
        logging.info("Training aborted")
    else:
        logging.info("Training complete")
        evaluate_model_on_testsets(model, test_datasets,
                                   dev_loss=dev_loss,
                                   device=device,
                                   num_workers=num_workers,
                                   storedir=storedir,
                                   logger=logger)


    finally:
        os.environ['OPDS_CACHE_DIR'] = os.path.expanduser('~/.cache/opds_eval')
        os.environ['OPDS_WRDS_DATA_DIR'] = os.path.abspath(
            "volatility_smoothing/data/wrds/spx")  # <- .csv file from WRDS should be place inside this directory

        spx_dataset = WRDSOptionsDataset(force_reprocess=False)
        spx_gno_dataset = GNOOptionsDataset(spx_dataset, subsample=False)

        dataloader = DataLoader(spx_gno_dataset, batch_size=1, shuffle=True,
                                  collate_fn=dev_loss.collate_fn, num_workers=0)
        os.makedirs(f"{storedir}/plots", exist_ok=True)
        for i, (data, input, aux, BS_x, B_grids_data, B_grids_dr, B_grids_drr, B_grids_dz, B_grids_dzz) in enumerate(dataloader):
            model = model.to('cpu')
            with torch.no_grad():
                output = model(**input)

            grid = aux['grids']  # RectilinearGrid
            iv_flat = torch.einsum('ijk,ijmk->m', output, B_grids_data.to(torch.float32))
            iv_surface = iv_flat.view(grid.size())  # (n_r, n_z)

            r_mat = grid['r'].cpu().numpy() 
            z_mat = grid['z'].cpu().numpy() 
            #r_vals = r_mat[:, 0].cpu().numpy()    # (n_r,)
            #z_vals = z_mat[0, :].cpu().numpy()    # (n_z,)
            #R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')
            #iv_vals = iv_surface.cpu().numpy()  # (n_r, n_z)
            tau_mat = r_mat ** 2                 # τ = r^2
            ln_k_mat = r_mat * z_mat              # k = r * z
            k_mat = np.exp(ln_k_mat)
            iv_vals = iv_surface.cpu().numpy()

            fig_surf = plt.figure(figsize=(10, 6))
            ax = fig_surf.add_subplot(111, projection='3d')
            ax.plot_surface(k_mat,
                            tau_mat,
                            iv_vals,
                            rstride=1, cstride=1,
                            cmap='viridis',
                            edgecolor='none')
            ax.set_xlabel('$Moneyness$')
            ax.set_ylabel(r'$time-to-maturity$')
            ax.view_init(elev=20)
            fig_surf.tight_layout() 
            date_str = data.quote_datetime.date().isoformat()
            fig_surf.savefig(f"{storedir}/plots/iv_surface_{date_str}_{i+1}.png", bbox_inches='tight')
            plt.close(fig_surf)

            fig_map, ax_map = plt.subplots(figsize=(8, 6))
            r_axis = grid['r'].cpu().numpy()          # (n_r, n_z)
            z_axis = grid['z'].cpu().numpy()
            iv_vals = iv_surface.cpu().numpy()        # (n_r, n_z)

            pcm = ax_map.pcolormesh(r_axis, z_axis, iv_vals, shading='auto', cmap='viridis')
            ax_map.set_xlabel('r')
            ax_map.set_ylabel('z')
            ax_map.set_title(f'IV Heatmap ({date_str})')

            fig_map.colorbar(pcm, ax=ax_map, label='Implied Volatility')
            plt.tight_layout()
            fig_map.savefig(f"{storedir}/plots/iv_heatmap_{date_str}_{i+1}.png", bbox_inches='tight')
            plt.close(fig_map)



            fig_slice, axs = dev_loss.plot_example(data, output, BS_x, figsize=(14, 20))
            fig_slice.suptitle(f'Example: quote_datetime={date_str}', fontsize=16)
            if i < 5:
                fig_slice.savefig(f"{storedir}/plots/example_{date_str}_{i+1}.png", bbox_inches='tight')
            plt.close(fig_slice)

            if i >= 4:
                break



