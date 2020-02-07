import random
from random import choices
import time
from gzip import open as g_open
from os import getcwd
from os.path import join

import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


SEED = 1873337
random.seed(SEED)
np.random.seed(SEED)


class SVM(object):
    def __init__(self, _c, _gamma, data_x, data_y, test_x, test_y):
        self._c = _c
        self._gamma = _gamma

        self.decomposition(data_x, data_y, test_x, test_y)

        self.b = None
        self.kkt_violation = None
        self.execution_time = None
        self.final_obj = None
        self.iterations = None

        self.primal_infeasibility = None
        self.dual_infeasibility = None
        self.primal_slack = None
        self.dual_slack = None

        self.alpha = None
        self.alpha_init = None
        self.alpha_gradient = None

    def rbf(self, data_x, data_y):
        return rbf_kernel(data_x, data_y, self._gamma)

    def compute_r_s_vectors(self, train_y, tolerance=1e-7):
        t_lower, t_upper = tolerance, self._c - tolerance
        lower_bound_idx = np.where(self.alpha <= t_lower)[0]
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
        return r_alpha, s_alpha

    def decomposition(self, data_x, data_y, test_x, test_y, q=2, first_iter=True):
        if not first_iter:
            r_alpha, s_alpha = self.compute_r_s_vectors(data_y)
            grad_r = self.alpha_gradient[r_alpha].flatten()
            grad_s = self.alpha_gradient[s_alpha].flatten()
            feasibility_condition, feasible = False, None
            for i in range(1, 10000):
                a = i * q
                sorted_s = np.argsort(grad_s)[a: a + (q // 2)]
                sorted_r = np.argsort(-grad_r)[a: a + (q // 2)]
                qr, qs = [r_alpha[j] for j in sorted_r], [s_alpha[i] for i in sorted_s]
                dir = np.zeros((data_x.shape[0], 1))
                dir[qr] = 1 / data_y[qr, ].reshape((-1, 1))
                dir[qs] = -1 / data_y[qs, ].reshape((-1, 1))
                feasible = self.alpha_gradient.T.dot(dir)
                if feasible[0][0] < 0:
                    feasibility_condition = True
                    break
            if feasibility_condition:
                _alpha_gradient = -1 * self.alpha_gradient / data_y.reshape((-1, 1))
                grad_r = _alpha_gradient[r_alpha].flatten()
                grad_s = _alpha_gradient[s_alpha].flatten()
                sorted_s, sorted_r = np.argsort(grad_s)[:q // 2], np.argsort(-1 * grad_r)[:q // 2]
                qr, qs = [r_alpha[j] for j in sorted_r], [s_alpha[i] for i in sorted_s]
            qr.extend(qs)
            q_vector = qr
            q_vec = list(set(q_vector))[:q]
        else:
            self.alpha_init = np.zeros((data_x.shape[0], 1))
            self.alpha = self.alpha_init
            r_alpha, s_alpha = self.compute_r_s_vectors(data_y)
            qr, qs = choices(r_alpha, k=q//2), choices(s_alpha, k=q//2)
            qr.extend(qs)
            q_vector = qr
            q_vec = list(set(q_vector))

        idx = set(np.array(range(1, data_x.shape[0])))
        w_hat_index = list(idx.difference(set(q_vec)))
        alpha_w_hat = np.array(self.alpha_init[w_hat_index]).reshape(
            (len(w_hat_index), 1))
        x_hat = data_x[w_hat_index, ]
        y_hat = data_y[w_hat_index, ].reshape((-1, 1))

        x = data_x[q_vec, ]
        y = data_y[q_vec, ].reshape((-1, 1))

        Q, p, G, h, A, b, Q_hat = self._prepare_solver(x, x_hat,
                                                       y, y_hat,
                                                       data_y, alpha_w_hat)

        tik = time.time()
        sol = solvers.qp(Q, p, G, h, A, b)
        tok = time.time()

        self.alpha = np.array(sol["x"])
        self.alpha_init[q_vec] = self.alpha
        self.alpha_init = self.alpha_init
        index = np.where(np.any(self.alpha > 1e-5, axis=1))
        sv_x, sv_y, alpha_star = self.compute_support_vectors(x, y,
                                                              self.alpha,
                                                              index)

        self.execution_time = tok - tik
        self.iterations = sol["iterations"]
        self.final_obj = sol["primal objective"]
        self.dual_obj = sol["dual objective"]
        self.primal_infeasibility = sol["primal infeasibility"]
        self.dual_infeasibility = sol["dual infeasibility"]
        self.primal_slack = sol["primal slack"]
        self.dual_slack = sol["dual slack"]

        self.b = self.compute_bstar(sv_x, sv_y, alpha_star)

        y_train_pred, y_test_pred = self.compute_predictions(data_x, test_x,
                                                             data_y)

        args = data_x, self.alpha, q_vec, Q, Q_hat, alpha_w_hat, w_hat_index
        self.alpha_gradient = self.calculate_gradient(args)

        self.kkt_violation = self.compute_kkt_violation(r_alpha, s_alpha)

        self.init_obj_val = self.compute_init_obj_val(data_x, data_y)

        while True:
            r_alpha, s_alpha = self.compute_r_s_vectors(data_y)
            _alpha_gradient = -1 * self.alpha_gradient / data_y.reshape((-1, 1))
            m, M = np.max(_alpha_gradient[r_alpha]), np.min(_alpha_gradient[s_alpha])
            if m - M >= 1e-5:
                self.decomposition(data_x, data_y, test_x, test_y, q=2, first_iter=False)
            else:
                break

        train_acc = self.compute_acc(y_train, y_train_pred)
        test_acc = self.compute_acc(y_test, y_test_pred)

        self.__print_training_info(train_acc, test_acc)

    def q_mat(self, x, y, data_y):
        k_mat = self.rbf(x, x)
        np.reshape(data_y, (len(data_y), -1))
        q_ = np.multiply(y, k_mat)
        _q = np.multiply(y, q_.T)
        Q = matrix(0.5 * _q.T)
        return Q

    def configure_solver(self):
        solvers.options['show_progress'] = False
        solvers.options['abstol'] = 1e-12
        solvers.options['feastol'] = 1e-12

    def _prepare_solver(self, x, x_hat, y, y_hat, data_y, alpha_w_hat):
        self.configure_solver()
        Q = self.q_mat(x, y, data_y)

        k_hat = self.rbf(x_hat, x)
        np.reshape(y_hat, (len(y_hat), -1))
        _q = np.multiply(y_hat, k_hat)
        q_ = np.multiply(y, _q.T)
        Q_hat = 0.5 * q_

        p = matrix(1*(Q_hat.dot(alpha_w_hat) - 1))

        A = matrix(y, (1, len(y)))
        # vector b is a scalar
        b = matrix(-y_hat.T.dot(alpha_w_hat), tc='d')

        # create the first constraint (- alpha <= 0)
        first_constraint, first_limit = np.diag(
            [-1]*len(y)), np.array([0]*len(y))
        # create the second constraint (alpha <= C)
        second_constraint, second_limit = np.diag(
            [1]*len(y)), np.array([self._c]*len(y))

        G = matrix(np.concatenate(
            (first_constraint, second_constraint)), tc='d')
        h = matrix(np.concatenate((first_limit, second_limit)))
        return Q, p, G, h, A, b, Q_hat

    def compute_support_vectors(self, x, y, alpha, index):
        if len(list(index)) > 1:
            sv_x = x[index]
            sv_y_ = y[index]
            sv_y = (sv_y_.T).reshape((-1, 1))
            alpha_star = alpha[index]
        else:
            sv_x = x
            sv_y = (y.T).reshape((-1, 1))
            alpha_star = alpha
        return sv_x, sv_y, alpha_star

    def compute_bstar(self, sv_x, sv_y, alpha_star):
        return np.mean((1-sv_y*sum(np.multiply(self.rbf(sv_x, sv_x),
                                               np.multiply(alpha_star, sv_y))))/sv_y)

    def compute_predictions(self, train_x, test_x, train_y):
        y_train_pred = np.sign(((np.multiply(self.rbf(train_x, train_x),
                                             np.multiply(self.alpha_init, train_y.reshape((-1, 1))))).sum(axis=0)).reshape((-1, 1)) + self.b)
        y_test_pred = np.sign(((np.multiply(self.rbf(train_x, test_x),
                                            np.multiply(self.alpha_init, train_y.reshape((-1, 1))))).sum(axis=0)).reshape((-1, 1)) + self.b)
        return y_train_pred, y_test_pred

    def calculate_gradient(self, args):
        data_x, alpha, q_vec, Q, Q_hat, alpha_w_hat, w_hat_index = args
        alpha_gradient_ = np.zeros((data_x.shape[0], 1))
        _alpha_grad = (np.array(Q).dot(alpha) +
                       np.array(Q_hat).dot(alpha_w_hat) - 1)
        __alpha_grad = np.array(Q_hat).T.dot(alpha)
        alpha_gradient_[q_vec] = _alpha_grad
        alpha_gradient_[w_hat_index] = __alpha_grad
        return alpha_gradient_

    def compute_acc(self, y_true, y_pred):
        return (accuracy_score(y_true, y_pred) * 100)

    def compute_init_obj_val(self, data_x, data_y):
        alpha_init = np.zeros((data_x.shape[0], 1))
        kernel_x = self.rbf(data_x, data_x)
        _kernel = (kernel_x * data_y) * data_y.T
        __kernel = 0.5 * _kernel * alpha_init * alpha_init.T
        return np.sum(__kernel) + np.sum(alpha_init)

    def compute_kkt_violation(self, r_alpha, s_alpha):
        m = max(np.take(self.alpha_gradient, r_alpha))
        M = min(np.take(self.alpha_gradient, s_alpha))
        return m - M

    def __print_training_info(self, train_acc, test_acc):
        print(f"Hyper-params: [Gamma: {self._gamma}, C: {self._c}, q: 2]")
        print(f"Time: {self.execution_time:6f} seconds")
        print(f"Number of iterations: {self.iterations}")
        print(f"Final objective function: {self.final_obj:4f}")
        print(f"Dual objective function: {self.dual_obj:4f}")
        print(f'Train acc: {train_acc:5f} %')
        print(f'Test acc: {test_acc:5f} %')
        print(f"initial objective fun val :{self.init_obj_val}")
        print(f"Primal infeasibility: {self.primal_infeasibility}")
        print(f"Dual infeasibility: {self.dual_infeasibility}")
        print(f"Primal slack: {self.primal_slack}")
        print(f"Dual slack: {self.dual_slack}")
        print(f"KKT Violation (m - M): {self.kkt_violation}")


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


if __name__ == "__main__":
    # Question 2.2
    # SVM Decomposition problem (q = 2)
    x_train, y_train, x_test, y_test = load_mnist()
    svm = SVM(0.1, 1e-5, x_train, y_train, x_test, y_test)
    print(svm)
