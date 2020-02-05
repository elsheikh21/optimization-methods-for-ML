import itertools
import random
import time
from gzip import open as g_open
from os import getcwd
from os.path import join
import json

import numpy as np
from cvxopt import solvers, matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

SEED = 1873337
random.seed(SEED)
np.random.seed(SEED)


class SVM(object):
    def __init__(self, c, gamma, train_x, train_y, test_x, test_y):
        self._c = c
        self._gamma = gamma
        self.b = 0
        self.num_iterations = 0
        self.execution_time = 0
        self.dual_objective = 0
        self.alpha = 0
        self.kkt_violation = None
        self.q = self.q_mat(train_x, train_y)
        self.primal_objective = None

        solvers.options['show_progress'] = False
        solvers.options['abstol'] = 1e-12
        solvers.options['feastol'] = 1e-12

        Q, p, G, h, A, b = self._prepare_solver(train_x, train_y)
        tik = time.time()
        sol = solvers.qp(Q, p, G, h, A, b, initvals=matrix(self.alpha))
        tok = time.time()
        self.alpha = np.array(sol['x'])

        # Support Vectors
        indices = np.where(np.any(self.alpha > 1e-5, axis=1))
        support_vectors_x = train_x[indices]
        support_vectors_y = ((train_y[indices]).T).reshape(-1, 1)
        alpha_star = self.alpha[indices]

        self.num_iterations = sol['iterations']
        self.dual_objective = sol['dual objective']
        self.primal_objective = sol['primal objective']
        self.b = np.mean((1 - support_vectors_y * sum(np.multiply(self.rbf(support_vectors_x, support_vectors_x),
                                                                  np.multiply(alpha_star, support_vectors_y))))/support_vectors_y)
        self.execution_time = tok - tik
        self.kkt_violation = self.compute_kkt_violation(train_x, train_y)

        train_threshold = (
            np.repeat(self.b, len(train_x), axis=0)).reshape((-1, 1))
        y_train_pred = np.sign(((np.multiply(self.rbf(support_vectors_x, train_x), np.multiply(
            alpha_star, support_vectors_y))).sum(axis=0)).reshape((-1, 1)) + train_threshold)

        test_threshold = (
            np.repeat(self.b, len(test_x), axis=0)).reshape((-1, 1))
        y_test_pred = np.sign(((np.multiply(self.rbf(support_vectors_x, test_x), np.multiply(
            alpha_star, support_vectors_y))).sum(axis=0)).reshape((-1, 1)) + test_threshold)

        self.__print_training_info(y_train_pred, train_y, y_test_pred, test_y)

    def rbf(self, data_x, data_y):
        return rbf_kernel(data_x, data_y, self._gamma)

    def q_mat(self, data_x, data_y):
        k = self.rbf(data_x, data_x)
        np.reshape(data_y, (len(data_y), -1))
        q1 = np.multiply(data_y, k)
        q = matrix(np.multiply(data_y, q1.T).T, tc='d')
        return q

    def _prepare_solver(self, train_x, train_y):
        Q = self.q_mat(train_x, train_y)
        p = matrix(np.repeat(-1, len(train_x)).reshape(len(train_x), 1), tc='d')
        A = matrix(train_y.copy().reshape(1, -1), tc='d')
        b = matrix(0, tc='d')

        # create constraints (- alpha <= 0) & (alpha <= C) and corresponding limits
        first_constraint = np.diag([-1] * len(train_y))
        first_limit = np.array([0] * len(train_y))
        second_constraint = np.diag([1] * len(train_y))
        second_limit = np.array([self._c] * len(train_y))

        G = matrix(np.concatenate(
            (first_constraint, second_constraint)), tc='d')
        h = matrix(np.concatenate((first_limit, second_limit)))

        return Q, p, G, h, A, b

    def dual_grad(self):
        return np.dot(self.q, self.alpha) - 1

    def compute_acc(self, y_true, y_pred):
        return (accuracy_score(y_true, y_pred) * 100)

    def compute_kkt_violation(self, train_x, train_y, tollerance=1e-7):
        t_lower, t_upper = tollerance, self._c - tollerance
        lower_bound_idx = np.where(self.alpha <= t_lower)[0]
        # FIXME: RETURNS EMPTY ARRAY, line 109
        self.alpha[lower_bound_idx] = 0
        upper_bound_idx = np.where(self.alpha >= t_upper)[0]
        self.alpha[upper_bound_idx] = self._c
        li = set(np.where(self.alpha == 0)[0])
        ui = set(np.where(self.alpha == self._c)[0])

        pos_l = li.intersection(set(np.where(train_y > 0)[0]))
        neg_l = li.intersection(set(np.where(train_y < 0)[0]))

        pos_u = ui.intersection(set(np.where(train_y > 0)[0]))
        neg_u = ui.intersection(set(np.where(train_y < 0)[0]))

        _rest = set(np.where(self.alpha < self._c)[0])
        rest = _rest.intersection(set(np.where(self.alpha > 0)[0]))

        r_alpha = list((pos_l.union(neg_u)).union(rest))
        s_alpha = list((neg_l.union(pos_u)).union(rest))

        grad = -np.multiply(self.dual_grad(), train_y.reshape(-1, 1))
        m, M = max(np.take(grad, r_alpha)), min(np.take(grad, s_alpha))
        return m - M

    def __print_training_info(self, y_train_pred, train_y, y_test_pred, test_y):
        print(f"Hyper-params: [Gamma: {self._gamma}, C: {self._c}]")
        print(f"Time: {self.execution_time:6f} seconds")
        print(f"Number of iterations: {self.num_iterations}")
        print(f"Final objective function: {self.primal_objective:4f}")
        print(f"Dual objective function: {self.dual_objective:4f}")
        print(f"KKT violation: {self.kkt_violation:6f}")
        print(f'Train accuracy: {self.compute_acc(train_y, y_train_pred):5f} %')
        print(f'Test accuracy: {self.compute_acc(test_y, y_test_pred):5f} %')


def load_mnist(path=join(getcwd(), 'Data'), kind='train'):
    labels_path = join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = join(path, f'{kind}-images-idx3-ubyte.gz')

    with g_open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),
                               dtype=np.uint8,
                               offset=8)

    with g_open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),
                               dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    index_label2 = np.where((labels == 2))
    x_label2 = images[index_label2][:1000, :].astype('float64')
    y_label2 = labels[index_label2][:1000].astype('float64')

    index_label4 = np.where((labels == 4))
    x_label4 = images[index_label4][:1000, :].astype('float64')
    y_label4 = labels[index_label4][:1000].astype('float64')

    # converting labels of classes 2 and 4 into +1 and -1, respectively
    y_label2 = y_label2 / 2.0
    y_label4 = y_label4 / -4.0

    x_label_24 = np.vstack((x_label2, x_label4))
    y_label_24 = np.concatenate((y_label2, y_label4))

    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(x_label_24)
    x_label_24 = scaler.transform(x_label_24)

    data = train_test_split(x_label_24, y_label_24,
                            test_size=0.3, random_state=SEED)

    x_train24, x_test24, y_train24, y_test24 = data

    return x_train24, y_train24, x_test24, y_test24


def grid_search_kfolds(x, y, save_res=True):
    c_params = [0.01, 0.1, 1, 2, 2.5, 3, 6, 10, 100]
    gamma_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    x_train, y_train, x_test, y_test = load_mnist()

    # GRID SEARCH
    results_dict = dict()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for params in tqdm(list(itertools.product(*(c_params, gamma_params)))):
        c, gamma = params
        kf_train_acc, kf_val_acc = [], []
        objs, num_iterations, time_exec = [], [], []
        for train_index, val_index in kf.split(x_train):
            x_train_, x_val = x_train[train_index], x_train[val_index]
            y_train_, y_val = y[train_index], y[val_index]
            svm = SVM(c, gamma, x_train_, y_train_, x_test, y_test)
            kf_train_acc.append(svm.compute_acc(x_train_, y_train_))
            kf_val_acc.append(svm.compute_acc(x_val, y_val))
            time_exec.append(svm.execution_time)
            num_iterations.append(svm.num_iterations)
            objs.append(svm.dual_objective)
    results_dict.update({params: [np.mean(kf_train_acc), np.mean(kf_val_acc),
                                  np.mean(time_exec), np.mean(num_iterations),
                                  np.mean(objs)]})
    if save_res:
        with open('SVM_GridSearch_KFolds.json', encoding='utf-8', mode='w+') as f:
            f.write(json.dump(results_dict))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    svm = SVM(0.1, 1e-5, x_train, y_train, x_test, y_test)
    print(svm)

    # TODO: TEST GridSearch with KFolds
    # TODO: Implement objective function
