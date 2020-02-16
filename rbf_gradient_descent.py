import json
import itertools
from tqdm import tqdm
import os
import random
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split


def rbf_function(data_x, omega, hyperparams):
    """
    Implements RBF network with a hidden layer
    Given data_x (samples) and the vector omega,
    computes y (predictions of the RBF function)

    omega contains:
        - v: [N x 1] array: parameters from the hidden to the output layer
        - c: [N x n] Matrix: N centroids in the first hidden layer
    """
    # Starting with unpack hyperparams and omega
    N, sigma, rho, n = hyperparams
    v = omega[0:N].reshape(N, 1)
    c = omega[N:].reshape(N, n)

    # Transform c, x into array for computations
    c_array = np.tile(c.reshape(-1), data_x.shape[0])
    X_array = np.tile(data_x, N).reshape(-1)

    # create ||X-c|| ^ 2 matrix
    # summation between 1st and 2nd components of x with c
    mat = ((c_array - X_array).reshape(data_x.shape[0], N, 2)) ** 2
    col = mat[:, :, 0] + mat[:, :, 1]
    col = np.exp(-col / (sigma ** 2))
    # Output of hidden layer (intermediate_output) multiply by v
    return np.dot(col, v).reshape(1, -1)


def gradient_rbf_function(omega, data_x, y_true, hyperparams):
    """
    Computes the gradient for loss fn. wrt v & c vectors

    X: samples
    omega:
        - v: [N x 1] array: parameters from the hidden to the output layer
        - c: [N x n] Matrix: N centroids in the first hidden layer
    y_true: true values of the function
    """
    N, sigma, rho, n = hyperparams
    v = omega[0:N].reshape(N, 1)
    c = omega[N:].reshape(N, n)
    # create ||X-c|| ^ 2 matrix
    c_array = np.tile(c.reshape(-1), data_x.shape[0])
    X_array = np.tile(data_x, N).reshape(-1)
    mat = ((c_array - X_array).reshape(data_x.shape[0], N, 2)) ** 2
    col = mat[:, :, 0] + mat[:, :, 1]

    # Apply Activation fn.
    col = np.exp(-col / (sigma ** 2))

    dE_dv = np.dot((rbf_function(data_x, omega, hyperparams) - y_true), col) / \
        data_x.shape[0] + 2 * rho * v.T
    dE_dv = dE_dv.reshape(-1, 1)

    # Matrix 1 & 2 for ops corresponding to 1st, 2nd comp of input
    mat1 = (-(c_array - X_array)).reshape(data_x.shape[0], N, 2)
    mat1 = mat1[:, :, 0]
    mat1 = 2 * (col * v.T * mat1) / (sigma ** 2)
    mat1 = np.dot((rbf_function(data_x, omega, hyperparams) - y_true),
                  mat1) / data_x.shape[0]
    mat2 = (-(c_array - X_array)).reshape(data_x.shape[0], N, 2)
    mat2 = mat2[:, :, 1]
    mat2 = 2 * (col * v.T * mat2) / (sigma ** 2)
    mat2 = np.dot((rbf_function(data_x, omega, hyperparams) - y_true),
                  mat2) / data_x.shape[0]
    # Merge ops results
    fusion = np.append(mat1.T, mat2.T, axis=1)
    dE_dc = fusion + 2 * rho * c
    return np.concatenate((dE_dv.reshape(1, -1), dE_dc.reshape(1, -1)), axis=1).reshape(-1)


def rbf_loss_function(omega, data_x, y_true, hyperparams):
    """
    Regularized training error of the RBF, computes the output
    of forward pass then computes objective function.
    """
    N, sigma, rho, n = hyperparams
    # Network output
    y_pred = np.tanh(rbf_function(data_x, omega, hyperparams))
    # objective function
    return (2 * np.sum((y_pred - y_true)) / (2 * data_x.shape[0]) + rho * np.linalg.norm(omega) ) * (1 - np.tanh(rbf_function(data_x, omega, hyperparams)) ** 2)


def mean_squared_error(y_true, y_pred):
    """ As the name implies """
    return np.mean(np.square(y_true.reshape(-1,) - y_pred.reshape(-1,))) / 2


def rbf_minimization(omega, hyperparams, data):
    """ Run minimization and fetch results"""
    x_train, x_test, y_train, y_test = data

    tok = time.time()
    res = minimize(rbf_loss_function, omega,
                   jac=gradient_rbf_function,
                   args=(x_train, y_train, hyperparams))
    tik = time.time()
    execution_time = tik - tok
    res.execution_time = round(execution_time, 5)

    y_train_pred = rbf_function(x_train, res.x, hyperparams)
    y_test_pred = rbf_function(x_test, res.x, hyperparams)
    res['train_mse'] = round(mean_squared_error(y_train, y_train_pred), 5)
    res['test_mse'] = round(mean_squared_error(y_test, y_test_pred), 5)
    return res


def parse_dataset(path=os.path.join(os.getcwd(), 'Data', 'data_points.csv')):
    dataset = np.genfromtxt(path, delimiter=',')
    x = dataset[1:, :2]
    _y = dataset[1:, 2]
    y = np.expand_dims(_y, -1)  # row -> column vector
    return x, y


def plot_model(hyperparams, res, img_path='rbf_plot.png'):
    X1 = np.linspace(-2, 2, 300)
    X2 = np.linspace(-1, 1, 300)
    X1, X2 = np.meshgrid(X1, X2)
    zs = np.array([rbf_function(np.array([x, y]).reshape(1, 2), res.x, hyperparams)
                   for x, y in zip(np.ravel(X1), np.ravel(X2))])
    Z = zs.reshape(X1.shape)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X1, X2, Z, linewidth=0,
                    cmap=cm.viridis, antialiased=False)
    ax.set_xticks((np.linspace(-2, 2, 10)))
    ax.view_init(elev=15, azim=240)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Z')
    plt.show()


def log_training_info(hyperparams, res):
    N, sigma, rho, n = hyperparams
    print(f'\n{"__________" * 5}')
    print(f'Number of neurons N: {N}, sigma: {sigma}, rho: {rho}')
    print('Optimization solver chosen: BFGS')
    print(f'Number of function evaluations: {res.nfev}')
    print(f'Number of iterations: {res.nit}')
    print(f'Number of gradient evaluations: {res.njev}')
    print(f'exec time: {res.execution_time} seconds')
    print(f'Training MSE: {res.train_mse}')
    print(f'Testing MSE: {res.test_mse}')
    print(f'Norm of gradient: {round(np.linalg.norm(res.jac), 5)}')
    print('__________' * 5)


def rbf_model(hyperparams, data, log=False, plot=False):
    N, sigma, rho, n = hyperparams
    v = np.random.normal(0, 1, N)
    c = np.append(np.random.uniform(-2, 2, N).reshape(-1, 1),
                  np.random.uniform(-1, 1, N).reshape(-1, 1),
                  axis=1).reshape(-1)

    omega = np.concatenate((v, c))
    res = rbf_minimization(omega, hyperparams, data)

    if log:
        log_training_info(hyperparams, res)
    if plot:
        plot_model(hyperparams, res)

    return res


def grid_search_kfolds(log=False, plot=False):
    N = [x for x in range(5, 55, 5)]  # Hidden Units
    rho = [1e-4]  # regularization param, fixed per the project description
    # Spread of Gaussian function (RBF)
    sigma = [x for x in np.arange(0.01, 1.11, 0.01)]
    n = 2
    res_dict = dict()
    x, y = parse_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.7,
                                                        random_state=SEED)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for params in tqdm(list(itertools.product(*(N, sigma, rho))), desc='performing grid search with kfolds'):
        comp_time = []
        nfev, nit, njev = [], [], []
        train_acc, val_acc = [], []
        # KFOLDS
        for train_index, val_index in kf.split(x_train):
            x_train_, x_val = x_train[train_index], x_train[val_index]
            y_train_, y_val = y_train[train_index], y_train[val_index]
            N, sigma, rho = params
            hyperparams = N, sigma, rho, n
            y_train_ = y_train_.reshape(1, -1)
            y_val = y_val.reshape(1, -1)
            data = x_train_, x_val, y_train_, y_val

            start_time = time.time()
            res = rbf_model(hyperparams, data, log=log, plot=plot)
            end_time = time.time()
            execution_time = end_time - start_time

            train_acc.append(res.train_mse * 100)
            val_acc.append(res.test_mse * 100)
            nfev.append(res.nfev)
            nit.append(res.nit)
            njev.append(res.njev)
            comp_time.append(execution_time)
        res_dict.update({f'{params}': [res.success, train_acc[-1],
                                       val_acc[-1], comp_time[-1],
                                       nfev[-1], nit[-1], njev[-1]
                                       ]
                         })
    with open('RBF_grid_search_kfolds_results.json', mode='w+') as file:
        json.dump(res_dict, file)

    return res_dict


if __name__ == '__main__':
    SEED = 1873337
    random.seed(SEED)
    np.random.seed(SEED)

    # For testing a single run
    x, y = parse_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.7,
                                                        random_state=SEED)
    data = x_train, x_test, y_train, y_test 
    hyperparams = N, sigma, rho, n = 30, 0.8, 1e-5, 2
    rbf_model(hyperparams, data)
    

    # _ = grid_search_kfolds()
