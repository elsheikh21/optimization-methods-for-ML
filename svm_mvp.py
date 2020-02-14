import time
import numpy as np
from os.path import join
from os import getcwd
from gzip import open as g_open
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
from cvxopt import matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel


SEED = 1873337
random.seed(SEED)
np.random.seed(SEED)


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

    labeled2data = np.append(x_label2, y_label2.reshape(1000, 1), axis=1)
    labeled4data = np.append(x_label4, y_label4.reshape(1000, 1), axis=1)

    data = np.append(labeled2data, labeled4data, axis=0)
    data_x = data[:, :-1]
    data_y = data[:, -1]

    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(data_x)
    data_x = scaler.transform(data_x)

    data = train_test_split(data_x, data_y,
                            test_size=0.3, random_state=SEED)

    x_train, x_test, y_train, y_test = data

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    return x_train, x_test, y_train, y_test


def rbf(data_x, data_y, gamma):
    return rbf_kernel(data_x, data_y, gamma)


def build_q_mat(x, y, gamma):
    k_mat = rbf(x, x, gamma)
    np.reshape(y, (len(y), -1))
    q_ = np.multiply(y, k_mat)
    _q = np.multiply(y, q_.T).T
    return _q


def compute_r_s_vectors(alpha_init, y, c, gamma, tolerance):
    t_lower, t_upper = tolerance, c - tolerance
    indices = np.where(alpha_init <= t_lower)[0]
    alpha_init[indices] = 0
    _indices = np.where(alpha_init >= t_upper)[0]
    alpha_init[_indices] = c
    li = set(np.where(alpha_init == 0)[0])
    pos_l = li.intersection(set(np.where(y > 0)[0]))
    neg_l = li.intersection(set(np.where(y < 0)[0]))

    ui = set(np.where(alpha_init == c)[0])
    pos_u = ui.intersection(set(np.where(y > 0)[0]))
    neg_u = ui.intersection(set(np.where(y < 0)[0]))

    _rest = set(np.where(alpha_init < c)[0])
    rest = _rest.intersection(set(np.where(alpha_init > 0)[0]))

    r_alpha = (pos_l.union(neg_u)).union(rest)
    s_alpha = (neg_l.union(pos_u)).union(rest)
    return list(r_alpha), list(s_alpha)


def compute_t_d_star(a, dij, c, k_grad, qk, w_k):
    a1, a2 = a
    d1, d2 = dij[0, 0], dij[1, 0]

    # Compute t_feasible, then t_star
    t_feasible = None
    if d1 > 0:
        if d2 > 0:
            t_feasible = min(c - a1, c - a2)
        else:
            t_feasible = min(c - a1, a2)
    else:
        if d2 > 0:
            t_feasible = min(a1, c - a2)
        else:
            t_feasible = min(a1, a2)

    if np.dot(k_grad[w_k].T, dij) == 0:
        t_star = 0
    else:
        if np.dot(k_grad[w_k].T, dij) < 0:
            d_star = dij
        else:
            d_star = -dij

        if t_feasible == 0:
            t_star = 0
        elif np.dot(np.dot(d_star.T, qk), d_star) == 0:
            t_star = t_feasible
        else:
            if np.dot(np.dot(d_star.T, qk), d_star) > 0:
                t_max = ((np.dot(-k_grad[w_k].T,
                                 d_star)) / (np.dot(np.dot(d_star.T,
                                                           qk),
                                                    d_star)))[0, 0]
                t_star = min(t_feasible, t_max)
    return t_star, d_star


def compute_acc(data_x, data_y, test_x, test_y, gamma, alpha, epsilon=1e-5):
    # Support vectors
    indices = np.where(np.any(alpha > epsilon, axis=1))
    sv_x, sv_y = data_x[indices], (data_y[indices].T).reshape((-1, 1))
    alpha_star = alpha[indices]

    b_star = np.mean((1 - sv_y * sum(np.multiply(rbf(sv_x, sv_x, gamma),
                                                 np.multiply(alpha_star, sv_y)))) / sv_y)
    y_pred = np.sign(((np.multiply(rbf(sv_x, data_x, gamma),
                                 np.multiply(alpha_star, sv_y))).sum(axis=0)).reshape((-1, 1)) + (np.repeat(b_star,
                                                                                                            len(data_x), axis=0)).reshape((-1, 1)))
    test_pred = np.sign(((np.multiply(rbf(sv_x, test_x, gamma),
                                    np.multiply(alpha_star, sv_y))).sum(axis=0)).reshape((-1, 1)) + (np.repeat(b_star,
                                                                                                               len(test_x), axis=0)).reshape((-1, 1)))

    train_acc = accuracy_score(data_y, y_pred)
    test_acc = accuracy_score(test_y, test_pred)

    return train_acc, test_acc


def print_training_info(hyperparams, results):
    print(
        f"Hyper-params: [Gamma: {hyperparams[0]}, C: {hyperparams[1]}, q: 2]")
    print(f"Time: {results.get('execution_time'):6f} seconds")
    print(f"Number of iterations: {results.get('iterations')}")
    print(f"Final objective function: {results.get('final_obj'):4f}")
    print(f"Train acc: {results.get('train_acc'):5f} %")
    print(f"Test acc: {results.get('test_acc'): 5f} %")
    print(f"KKT Violation (m - M): {results.get('kkt_violation')}")


def svm_mvp(c, gamma, tolerance, data_x, data_y, test_x, test_y):
    tik = time.time()

    p = data_x.shape[0]
    alpha_k = np.zeros((p, 1))
    q_mat = build_q_mat(data_x, data_y, gamma)
    k, opt = 0, False

    while not opt:
        # Create working set, then reduce its dimensionality
        r_alpha, s_alpha = compute_r_s_vectors(alpha_k, data_y,
                                               c, gamma, tolerance)
        k_grad = np.dot(q_mat, alpha_k) - 1
        y_grad = -np.multiply(k_grad, data_y)
        m, M = max(np.take(y_grad, r_alpha)), min(np.take(y_grad, s_alpha))
        m_idx = np.where(y_grad == m)[0][0]
        M_idx = np.where(y_grad == M)[0][0]
        w_k = [m_idx, M_idx]

        data_xk, data_yk = data_x[w_k], data_y[w_k]
        qk = q_mat[w_k, :][:, w_k]
        alpha_kn = alpha_k[w_k]
        dij = np.array([data_yk[0], -data_yk[1]])
        a1, a2 = alpha_kn[0, 0], alpha_kn[1, 0]

        a = a1, a2
        t_star, d_star = compute_t_d_star(a, dij, c, k_grad, qk, w_k)

        alpha_star = alpha_kn + np.dot(t_star, d_star)
        alpha_k[w_k] = alpha_star
        n_grad = np.dot(q_mat, alpha_k) - 1
        r_alpha, s_alpha = compute_r_s_vectors(alpha_k, data_y,
                                               c, gamma, tolerance)
        y_grad = -np.multiply(n_grad, data_y)
        m, M = max(np.take(y_grad, r_alpha)), min(np.take(y_grad, s_alpha))
        k += 1

        if m - M < tolerance or k == max_iterations:
            opt = True
            tok = time.time()
            computational_time = tok - tik
            final_obj = np.dot(np.dot(alpha_k.T, q),
                               alpha_k) * 0.5 - np.sum(alpha_k)
            train_acc, test_acc = compute_acc(
                data_x, data_y, test_x, test_y, gamma, alpha_k)
            hyperparams = [gamma, c]
            results = {
                'iterations': k,
                'final_obj': final_obj[0][0],
                'execution_time': computational_time,
                'kkt_violation': m - M,
                'train_acc': train_acc,
                'test_acc': test_acc
            }
            print_training_info(hyperparams, results)
    return {'Hyper-parameters': hyperparams, 'Results': results}


if __name__ == "__main__":
    # Loading data
    x_train, x_test, y_train, y_test = load_mnist()

    q = 2
    # TODO: REPLACE WITH BEST HYPERPARAMs
    c, gamma = 0.1, 1e-5
    max_iterations, tolerance = 100_000, 1e-5
    train_history = svm_mvp(c, gamma, tolerance, x_train, y_train, x_test, y_test)
    print(train_history)
