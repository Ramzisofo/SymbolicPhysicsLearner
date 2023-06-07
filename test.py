import numpy as np
from numpy import *
from sympy import simplify, expand, symbols, Matrix, lambdify, MatrixSymbol
from scipy.optimize import minimize
from contextlib import contextmanager
import threading
import _thread
import time
import ot
import torch
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial


def path_func(t, param=2):
    return 1 / ((1 / param) + 2 * t)

TimeIdx = np.array([0, 2, 3, 5, 8, 10, 13, 20, 21, 25, 27, 35, 41, 47, 50])
idx = 0
n = 100
init_dist = np.random.randn(n)

data = np.empty((n, len(TimeIdx)))
estim = np.zeros((n, len(TimeIdx)))
for samp in range(n):
    for ti in range(len(TimeIdx)):
        data[samp, ti] = path_func(TimeIdx[ti], init_dist[samp])


M = ot.dist(data[:, idx].reshape((n, 1)), estim[:, idx].reshape((n, 1)))
M /= M.max()

hist1, _ = np.histogram(data[:, idx], bins=n)
hist2, _ = np.histogram(estim[:, idx], bins=n)

# r += ot.sinkhorn2(hist1/n, hist2/n, M, reg=1)
r[2] += 20 * ot.emd2(hist1 / n, hist2 / n, M)