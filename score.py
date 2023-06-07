import numpy as np
from numpy import *
from sympy import simplify, expand, symbols, Matrix, lambdify, MatrixSymbol, sympify, parse_expr
from scipy.optimize import minimize
from contextlib import contextmanager
import threading
import _thread
import time
import ot
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**' + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq: eq = eq.replace(c, 'C')
    return simplify_eq(eq)

dim_sys = 5
def path_func(t, param=2*np.ones(dim_sys)):

    return np.array([param[0]*np.exp(t), param[1]*np.exp(2*t)]) # -np.log(np.exp(-param) - t)

# path_func = np.vectorize(path_func)

TimeIdx = 10*np.array([0, 2, 3, 5, 8, 10, 13, 20])

np.random.seed(7)
n = 100
# init_dist = np.random.randn(n*dim).reshape((n, dim))
# init_dist = np.array([x for x in init_dist if np.exp(-x) > 100])
# print("len init_dist = ", len(init_dist))
# n = len(init_dist)

data_set = np.empty((dim_sys, n, len(TimeIdx)))
df = pd.read_table("output.txt", delimiter=",", header=None)
nb_time = 1001
for dim in range(dim_sys):
    for idx, time_idx in enumerate(TimeIdx):
        data_set[dim, :, idx] = df.iloc[:, dim*nb_time+time_idx]

def reward(eq_loc, pr=False, data=data_set):

    # Find numerical values in the equation
    eq_ = []
    num_values = []
    nb_plus = []
    for d in range(dim_sys):
        eq_.append(simplify_eq(eq_loc[d]))
        num_values.append(re.findall(r'\b\d+(?:\.\d+)?\b', eq_[d]))
        nb_plus.append(eq_[d].count("+"))
        num_values[d] = np.array([float(num) for num in num_values[d]])
        if ((np.all(np.abs(num_values[d])< 0.1)) and (len(num_values[d]) > nb_plus[d]) ) or eq_[d] == '0':
            return 1000

    def eval_eq(t, var):
        res = []
        for i in range(1, dim_sys+1):
            globals()["x"+str(i)] = var[i-1]
        for i in range(dim_sys):
            res.append(eval(eq_loc[i]))
        return np.array(res)

    def eval_vec(t, var):
        eq_loc_ = simplify_eq(eq_loc[0])
        res = []
        for i in range(n):
            globals()["x"] = var[i]
        for i in range(n):
            res.append(eval(eq_loc_))
        return np.array(res)

    execution_time = 0
    # Generate the data
    estim = np.empty((dim_sys, n, len(TimeIdx)))
    # data = np.empty((dim, n, len(TimeIdx)))
    # for samp in range(n):
    #     for ti in range(len(TimeIdx)):
    #         data[:, samp, ti] = path_func(TimeIdx[ti], init_dist[samp])

    r = np.zeros(4)
    estim[:, :, 0] = data[:, :, 0]
    for idx in range(len(TimeIdx)-1):
        # if pr:
        #     print("idx = ", idx)
        t = TimeIdx[idx + 1]
        for sample in range(n):
            # if pr:
            #     print("sample = ", sample)
            nxt_samp = integrate.RK45(eval_eq, TimeIdx[idx], data[:, sample, idx], TimeIdx[-1]+ 1, max_step=0.1)
            space_trj = {}
            while nxt_samp.t < t and nxt_samp.status != "failed":
                nxt_samp.step()
                space_trj[nxt_samp.t] = nxt_samp.y
            estim[:, sample, idx+1] = nxt_samp.y

        for i in range(dim_sys):
            M = ot.dist(data[i, :, idx].reshape((n, 1)), estim[i, :, idx].reshape((n, 1)))
            M /= M.max()

            hist1, _ = np.histogram(data[i, :, idx], bins=n)
            hist2, _ = np.histogram(estim[i, :, idx], bins=n)

            # r[2] += ot.sinkhorn2(hist1/n, hist2/n, M, reg=1)
            r[2] += 20*ot.emd2(hist1/n , hist2/n , M)

    X = MatrixSymbol('X', dim_sys, 1)
    eq_sym = []
    for i in range( dim_sys):
        eq_sym.append(sympify(eq_loc[i]))
        eq_sym[i] = eq_sym[i].xreplace({sympify('x'+str(var+1)): X[var, 0] for var in range(dim_sys)})
    z = Matrix(eq_sym)
    t = z.jacobian(X)
    jac = lambdify(X, t, "numpy")
    # jac(np.ones((dim, 1)))
    r[0] += np.linalg.norm(eval_eq(TimeIdx[-1], data[:, -1, 0]))
    jaco = np.linalg.norm(jac(data[:, -1, 0].reshape((dim_sys, 1))))
    r[1] += jaco

    # Manifold loss
    def density_loss(source, target, top_k=5, hinge_value=0.01):
        source = torch.from_numpy(source)
        target = torch.from_numpy(target)
        c_dist = torch.stack([
            torch.cdist(source, target)
        ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= hinge_value
        values[values < 0] = 0
        loss = torch.mean(values)
        return loss

    # r[3] += 10*density_loss(data, estim)
    print("eq = ", eq_)
    print("r = ", r)
    print("euclid dist = ", np.linalg.norm(data[:, 0, :] - estim[:, 0, :]))
    print("R = ", r[2], "sum = ", r[3], "True reward ", 1/(1+ np.sum(r)))
    return np.sum(r)

def score_with_est(eq, tree_size, data, t_limit=200.0, eta=0.999, pr=False):
    """
    Calculate reward score for a complete parse tree
    If placeholder C is in the equation, also excute estimation for C
    Reward = 1 / (1 + MSE) * Penalty ** num_term

    Parameters
    ----------
    eq : Str object.
        the discovered equation (with placeholders for coefficients).
    tree_size : Int object.
        number of production rules in the complete parse tree.
    data : 2-d numpy array.
        measurement data, including independent and dependent variables (last row).
    t_limit : Float object.
        time limit (seconds) for ssingle evaluation, default 1 second.

    Returns
    -------
    score: Float
        discovered equations.
    eq: Str
        discovered equations with estimated numerical values.
    """

    ## count number of numerical values in eq
    dim = len(eq)
    c_count = []
    for i in range(dim):
        c_count.append(eq[i].count('C'))
    # start_time = time.time()
    if True: # with time_limit(t_limit, 'sleep'):
        try:
            if c_count == []:  ## no numerical values
                f_pred = reward(eq, pr)
                # print("c_count ", f_pred)
            elif np.sum(c_count) >= 10:  ## discourage over complicated numerical estimations
                print("sup à 10 ")
                return 0, eq
            else:  ## with numerical values: coefficient estimation with Powell method

                # eq = prune_poly_c(eq)
                c_lst = ['c' + str(i) for i in range(np.sum(c_count))]
                for d in range(dim):
                    for c in range(c_count[d]):
                        eq[d] = eq[d].replace('C', c_lst[int(c + np.sum(c_count[:d]))], 1)

                def eq_test(c):
                    nonlocal eq
                    eq_loc = eq.copy()
                    for d in range(dim):
                        for i in range(c_count[d]):
                            idx = int(i + np.sum(c_count[:d]))
                            globals()['c' + str(idx)] = c[idx]
                            if pr:
                                print(" i = ", idx, "je suis là ")
                            eq_loc[d] = eq_loc[d].replace('c' + str(idx), str(c[idx]))
                    return reward(eq_loc)

                x0 = [1.0] * len(c_lst)
                opt = {'maxiter':100, 'disp':False}
                # print("avant minimize ")
                c_lst = minimize(eq_test, x0, method='Powell', tol=1e-2, options=opt).x.tolist()
                # print("après minimize ")
                # c_lst = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in c_lst]
                c_lst = [np.round(x, 4) for x in c_lst]
                eq_est = eq.copy()
                for d in range(dim):
                    for i in range(c_count[d]):
                        idx = int(i + np.sum(c_count[:d]))
                        eq_est[d] = eq_est[d].replace('c' + str(idx), str(c_lst[idx]), 1)
                    eq[d] = eq_est[d].replace('+-', '-')
                f_pred = reward(eq, pr)
                # print("simp eq = ", simplify_eq(eq))

        except:
            return 0, eq

    # r = float(eta ** tree_size / (1.0 + np.linalg.norm(f_pred - f_true, 2) ** 2 / f_true.shape[0]))
    r = eta ** (np.sum(tree_size)) / (1 + f_pred )
    # run_time = np.round(time.time() - start_time, 3)
    # print('runtime :', run_time,  eq,  np.round(r, 3))

    return r, eq