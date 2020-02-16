import os
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split, KFold
import itertools
import logging
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


SEED = 1873337
random.seed(SEED)
np.random.seed(SEED)


class Network:
    def __init__(self, hidden_size, input_size, output_size, _rho):
        # save to use later
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rho = _rho

    def forward(self, *args):
        raise NotImplementedError

    def loss(self, omega, inputs, labels):
        # calculate loss
        outputs = self.forward(inputs, omega)
        error = np.mean(np.square(outputs - labels))
        regularization = self.rho * np.square(np.linalg.norm(omega))
        return error + regularization

    def fit(self, *args):
        raise NotImplementedError

    def extreme_learning(self, *args):
        raise NotImplementedError

    def decomposition(self, *args):
        raise NotImplementedError

    def save(self, *args):
        raise NotImplementedError

    def load(self, *args):
        raise NotImplementedError

    def surface_plot(self, inputs, optimal_params, title=''):
        outputs = self.forward(inputs, optimal_params)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X1, X2, Y = inputs[:, 0], inputs[:, 1], outputs.ravel()

        ax.scatter(X1, X2, Y, color='red', alpha=1)
        ax.plot_trisurf(X1, X2, Y, cmap='viridis', edgecolor='none', antialiased=False)
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        plt.show()


class RBF(Network):
    def __init__(self, hidden_size, input_size=2, output_size=1, _rho=1e-4, _sigma=1.):
        # initialize weights and biases
        self.C = np.random.rand(input_size, hidden_size)
        self.V = np.random.rand(hidden_size, output_size)
        self.sigma = _sigma

        super().__init__(hidden_size, input_size, output_size, _rho)

    def forward(self, inputs, omega):
        self.__unpack_omega(omega)

        # C needs to be in shape (#samples, dim, #centroids)
        c = np.tile(self.C, (inputs.shape[0], 1, 1))

        intermediate_output = np.zeros((inputs.shape[0], self.hidden_size))

        # hidden units = #centroids
        for i in range(self.hidden_size):
            # subtract all points from centroid
            # take norm of each distance vector (axis = 1)
            intermediate_output[:, i] = self.gaussian(
                np.linalg.norm(inputs - c[:, :, i], axis=1))

        return np.dot(intermediate_output, self.V)

    def gaussian(self, z):
        return np.exp(-np.square(z / self.sigma))

    def fit(self, inputs, labels, use_gradient):
        # omega contains all free params of the network
        omega = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])

        return self.__run_minimization(inputs, labels, omega,
                                       gradient=use_gradient)

    def extreme_learning(self, inputs, labels):
        # pick `N` centers from `inputs`
        self.C = np.array(random.choices(inputs, k=self.hidden_size)).T
        # omega contains `V` only
        omega = self.V
        self.__run_minimization(inputs, labels, omega, gradient=False)

    def decomposition(self, inputs, labels):
        tik = time.time()
        early_stopping_cond = 1e-5
        sum_of_gradients, i, max_iters = 1, 0, 50

        clusters = KMeans(n_clusters=self.hidden_size,
                          random_state=SEED).fit(inputs)
        self.C = np.array(clusters.cluster_centers_).T

        omega = self.V

        print(f'Initial training error: {self.test_loss(inputs, labels):.4f}')
        print(
            f'Initial value of objective function: {self.loss(omega, inputs, labels):.4f}')

        while sum_of_gradients > early_stopping_cond and i < max_iters:
            # optimize V
            omega = self.V
            optimizer1 = self.__run_minimization(inputs, labels, omega)
            gradient_1 = np.linalg.norm(optimizer1.jac.T)
            self.V = optimizer1.x.reshape(*self.V.shape)

            # optimize C
            omega = self.C.reshape(self.C.size, 1)
            optimizer2 = self.__run_minimization(inputs, labels, omega)
            gradient_2 = np.linalg.norm(optimizer2.jac.T)
            self.C = optimizer2.x.reshape(*self.C.shape)

            sum_of_gradients = gradient_1 + gradient_2
            i += 1

        tok = time.time()

        self.__print_training_info(inputs, labels, optimizer2, tok - tik)

    def __run_minimization(self, inputs, labels, omega, gradient=True):
        # initial error
        init_training_err = self.test_loss(inputs, labels)
        print(f'Initial training error: {init_training_err:.4f}')
        init_obj_fn = self.loss(omega, inputs, labels)
        print(f'Initial value of objective function: {init_obj_fn:.4f}')
        # back-propagation
        if not gradient:
            tik = time.time()
            optimal = minimize(fun=self.loss, x0=omega, args=(inputs, labels))
            tok = time.time()
        else:
            grad_vector = self.compute_gradients(omega, inputs, labels)
            tik = time.time()
            optimal = minimize(fun=self.loss, x0=omega,
                               jac=grad_vector, args=(inputs, labels))
            tok = time.time()
        # print out required info
        self.__print_training_info(inputs, labels, optimal, tok - tik)
        return optimal

    def helper_gradients(self, inputs, omega):
        # unpack the parameters from omega
        v = omega[0:self.hidden_size].reshape(self.hidden_size, 1)
        c = omega[self.hidden_size:].reshape(self.hidden_size, inputs.shape[1])

        # replicate c and X values as array
        c_array = np.tile(c.reshape(-1), inputs.shape[0])
        X_array = np.tile(inputs, self.hidden_size).reshape(-1)

        # create a tensor representing ||X-c||**2 matrix
        mat = ((X_array - c_array).reshape(inputs.shape[0],
                                           self.hidden_size, 2)) ** 2

        # sum (X[0] - c) with (X[1] - c) for each observation
        col = mat[:, :, 0] + mat[:, :, 1]
        col = np.exp(-col / (self.sigma ** 2))

        # now that we have the output of the hidden layer
        # make the dot product with v vector
        return np.dot(col, v).reshape((1, -1))

    def compute_gradients(self, omega, inputs, labels):
        v = omega[0:self.hidden_size].reshape(self.hidden_size, 1)
        c = omega[self.hidden_size:].reshape(self.hidden_size, self.input_size)

        # de_dv
        # create a tensor representing ||X-c||**2 matrix
        c_array = np.tile(c.reshape(-1), inputs.shape[0])
        X_array = np.tile(inputs, self.hidden_size).reshape(-1)
        mat = ((c_array - X_array).reshape(inputs.shape[0],
                                           self.hidden_size, 2)) ** 2
        col = mat[:, :, 0] + mat[:, :, 1]

        # activation function
        col = np.exp(-col / (self.sigma ** 2))

        # dE_dv
        dE_dv = np.dot((self.helper_gradients(inputs, omega) - labels), col) / inputs.shape[0] + 2 * self.rho * v.T
        dE_dv = dE_dv.reshape(-1, 1)

        # dE_dc
        # mat1 and mat2 are matrices that correspond to calculations
        # performed on the first and second components of X respectively
        mat1 = (-(c_array - X_array)).reshape(inputs.shape[0],
                                              self.hidden_size, 2)
        mat1 = mat1[:, :, 0]
        mat1 = 2 * (col * v.T * mat1) / (self.sigma ** 2)
        mat1 = np.dot((self.helper_gradients(inputs, omega) -
                       labels), mat1) / inputs.shape[0]

        mat2 = (-(c_array - X_array)
                ).reshape(inputs.shape[0], self.hidden_size, 2)
        mat2 = mat2[:, :, 1]
        mat2 = 2 * (col * v.T * mat2) / (self.sigma ** 2)
        mat2 = np.dot((self.helper_gradients(inputs, omega) -
                       labels), mat2) / inputs.shape[0]

        # now merge the results
        fusion = np.append(mat1.T, mat2.T, axis=1)

        # dE_dc
        dE_dc = fusion + 2 * self.rho * c

        return np.concatenate((dE_dv.reshape(1, -1), dE_dc.reshape(1, -1)), axis=1).reshape(-1)

    def __print_training_info(self, inputs, labels, result, elapsed_time):
        print(f'Number of neurons: {self.hidden_size}')
        print(f'Value of sigma: {self.sigma}')
        print(f'Value of rho: {self.rho}')
        print(f'Termination message: {result.message}')
        print(f'Solver: BFGS (Default)')
        print(f'Number of iterations: {result.nit}')
        print(f'Final value of objective function: {result.fun:.4f}')
        print(f'Number of function evaluations: {result.nfev}')
        print(f'Final value of gradient: {np.linalg.norm(result.jac):.4f}')
        print(f'Norm of gradient: {np.linalg.norm(result.jac)}')
        print(f'Number of gradient evaluations: {result.njev}')
        print(f'Time for optimization: {elapsed_time:.4f} seconds')
        print(f'Final Training error: {self.test_loss(inputs, labels):.4f}')

    def test_loss(self, inputs, labels):
        # only for use on val/test data, not during training
        omega = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])
        outputs = self.forward(inputs, omega)
        return np.mean(np.square(outputs - labels))

    def save(self, filename=''):
        omega = np.concatenate([self.V, self.C.reshape(self.C.size, 1)])
        filename = 'rbf_weights' if filename == '' else filename
        np.save(filename, omega)

    def load(self, filename=''):
        filename = 'rbf_weights.npy' if filename == '' else filename
        omega = np.load(filename)
        self.__unpack_omega(omega)

    def surface_plot(self, inputs, title='', *args):
        optimal_parameters = np.concatenate(
            [self.V, self.C.reshape(self.C.size, 1)])
        super().surface_plot(inputs, optimal_parameters, 'RBF' if title == '' else title)

    def __unpack_omega(self, omega):
        # check omega size
        if omega.size == self.V.size:
            self.V = omega[:self.V.size].reshape(*self.V.shape)
        elif omega.size == self.C.size:
            self.C = omega[:self.C.size].reshape(*self.C.shape)
        else:
            self.V = omega[:self.V.size].reshape(*self.V.shape)
            self.C = omega[self.V.size:].reshape(*self.C.shape)


def parse_dataset(path=os.path.join(os.getcwd(), 'Data', 'data_points.csv')):
    dataset = np.genfromtxt(path, delimiter=',')
    x = dataset[1:, :2]
    _y = dataset[1:, 2]
    y = np.expand_dims(_y, -1)  # row -> column vector
    return x, y


def grid_search_kfolds(x, y, save_res=True):
    N = [5, 10, 25, 50]  # Hidden Units
    rho = [1e-4]  # regularization param, fixed per the project description
    sigma = [0.25, 0.5, 1, 2]  # Spread of Gaussian function (RBF)

    # GRID SEARCH
    res_dict = dict()
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, train_size=0.7, random_state=SEED)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for params in tqdm(list(itertools.product(*(N, rho, sigma)))):
        n, r, s = params
        rbf = RBF(hidden_size=n, _rho=r, _sigma=s)
        comp_time = []
        nfev, nit, njev = [], [], []
        train_acc, val_acc = [], []
        # KFOLDS
        for train_index, val_index in kf.split(x_train):
            x_train_, x_val = x_train[train_index], x_train[val_index]
            y_train_, y_val = y[train_index], y[val_index]
            start = time.time()
            res = rbf.fit(x_train_, y_train_, use_gradient=True)
            computational_time = time.time() - start
            train_acc.append(rbf.test_loss(x_train_, y_train_))
            val_acc.append(rbf.test_loss(x_val, y_val))
            nfev.append(res['nfev'])
            nit.append(res['nit'])
            njev.append(res['njev'])
            comp_time.append(computational_time)
        res_dict.update({params: [res.success, np.mean(train_acc),
                                  np.mean(val_acc), np.mean(
                                      comp_time), int(np.mean(nfev)),
                                  int(np.mean(nit)), int(np.mean(njev))]})

    if save_res:
        with open('RBF_GridSearch_KFolds.json', encoding='utf-8', mode='w+') as f:
            f.write(json.dump(res_dict))

    return x_train, x_rest, y_train, y_rest


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'Data', 'data_points.csv')
    x, y = parse_dataset(path)
    logging.info('Dataset is parsed')

    # x_train, x_test, y_train, y_test = grid_search_kfolds(x, y, save_res=True)

    # Best results are chosen w.r.t validation error
    # Best Results are Hidden_Size: {} Sigma: {} Rho: {}
    # TODO: replace with best parameters
    N, r, s = 25, 1e-4, 1.0

    x, y = parse_dataset()
    x_train, x_rest, y_train, y_rest = train_test_split(x, y.reshape(-1, 1),
                                                        train_size=0.7,
                                                        random_state=SEED)

    # Question 1.1, Gradient Descent
    # rbf = RBF(hidden_size=N, _rho=r, _sigma=s)
    # rbf.fit(x_train, y_train, use_gradient=True)
    # print(f'Test error: {rbf.test_loss(x_rest, y_rest):.4f}')
    # rbf.surface_plot(x_rest)

    # Question 1.2, two-block decomposition
    rbf = RBF(hidden_size=N, _rho=r, _sigma=s)
    rbf.extreme_learning(x_train, y_train)
    print(f'Test error: {rbf.test_loss(x_rest, y_rest):.4f}')
    rbf.surface_plot(x_rest)
