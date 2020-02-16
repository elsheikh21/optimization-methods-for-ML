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


def rbf_function(X, omega, hyperparams):
    N, sigma, rho, n = hyperparams
    v = omega[0:N].reshape(N, 1)
    c = omega[N:].reshape(N, n)

    c_array = np.tile(c.reshape(-1), X.shape[0])
    X_array = np.tile(X, N).reshape(-1)

    mat = ((c_array - X_array).reshape(X.shape[0], N, 2)) ** 2

    col = mat[:, :, 0] + mat[:, :, 1]
    col = np.exp(-col / (sigma ** 2))
    return np.dot(col, v).reshape(1, -1)


def gradient_rbf_function(omega, X, y_true, hyperparams):
    N, sigma, rho, n = hyperparams
    v = omega[0:N].reshape(N, 1)
    c = omega[N:].reshape(N, n)

    c_array = np.tile(c.reshape(-1), X.shape[0])
    X_array = np.tile(X, N).reshape(-1)

    mat = ((c_array - X_array).reshape(X.shape[0], N, 2)) ** 2
    col = mat[:, :, 0] + mat[:, :, 1]
    col = np.exp(-col / (sigma ** 2))

    dE_dv = np.dot((rbf_function(X, omega, hyperparams) - y_true), col) / \
        X.shape[0] + 2 * rho * v.T
    dE_dv = dE_dv.reshape(-1, 1)

    mat1 = (-(c_array - X_array)).reshape(X.shape[0], N, 2)
    mat1 = mat1[:, :, 0]
    mat1 = 2 * (col * v.T * mat1) / (sigma ** 2)
    mat1 = np.dot((rbf_function(X, omega, hyperparams) - y_true),
                  mat1) / X.shape[0]
    mat2 = (-(c_array - X_array)).reshape(X.shape[0], N, 2)
    mat2 = mat2[:, :, 1]
    mat2 = 2 * (col * v.T * mat2) / (sigma ** 2)
    mat2 = np.dot((rbf_function(X, omega, hyperparams) - y_true),
                  mat2) / X.shape[0]

    fusion = np.append(mat1.T, mat2.T, axis=1)
    dE_dc = fusion + 2 * rho * c
    return np.concatenate((dE_dv.reshape(1, -1), dE_dc.reshape(1, -1)), axis=1).reshape(-1)


def rbf_loss_function(omega, X, y_true, hyperparams):
    N, sigma, rho, n = hyperparams
    y_pred = rbf_function(X, omega, hyperparams)
    # objective function
    return np.sum((y_pred - y_true)**2) / (2 * X.shape[0]) + rho * np.linalg.norm(omega)**2


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true.reshape(-1,) - y_pred.reshape(-1,))) / 2


def rbf_minimization(omega, hyperparams, data):
    x_train, x_test, y_train, y_test = data

    tok = time.time()
    res = minimize(rbf_loss_function, omega,
                   jac=gradient_rbf_function, args=(x_train, y_train, hyperparams))
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


def grid_search_kfolds():
    N = [5, 10, 25, 50]  # Hidden Units
    rho = [1e-4]  # regularization param, fixed per the project description
    sigma = [0.25, 0.5, 1, 2]  # Spread of Gaussian function (RBF)
    n = 2
    res_dict = dict()
    x, y = parse_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.7,
                                                        random_state=SEED)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for params in tqdm(list(itertools.product(*(N, sigma, rho)))):
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
            res = rbf_model(hyperparams, data, log=False, plot=False)
            end_time = time.time()
            execution_time = end_time - start_time

            train_acc.append(res.train_mse)
            val_acc.append(res.test_mse)
            nfev.append(res.nfev)
            nit.append(res.nit)
            njev.append(res.njev)
            comp_time.append(execution_time)
        res_dict.update({params: [res.success, np.mean(train_acc),
                                  np.mean(val_acc), np.mean(comp_time),
                                  int(np.mean(nfev)), int(np.mean(nit)),
                                  int(np.mean(njev))]})
    return res_dict


if __name__ == '__main__':
    SEED = 1873337
    random.seed(SEED)
    np.random.seed(SEED)

    # data = x_train, x_test, y_train, y_test = parse_dataset()
    # hyperparams = N, sigma, rho, n = 30, 0.8, 1e-5, 2
    # rbf_model(hyperparams, data)

    obj = grid_search_kfolds()
    
