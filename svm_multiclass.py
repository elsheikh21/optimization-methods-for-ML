import random
from gzip import open as g_open
from os import getcwd
from os.path import join
import numpy as np
from cvxopt import solvers, matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
import time
import pandas as pd


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

    size = 1000
    index_label2 = np.where((labels == 2))
    x_label2 = images[index_label2][:size, :].astype('float64')

    index_label4 = np.where((labels == 4))
    x_label4 = images[index_label4][:size, :].astype('float64')

    index_label6 = np.where((labels == 6))
    x_label6 = images[index_label6][:size, :].astype('float64')

    y_label2 = np.array([2] * 1000)
    y_label4 = np.array([4] * 1000)
    y_label6 = np.array([6] * 1000)

    label2_data = np.append(x_label2, y_label2.reshape(1000, 1), axis=1)
    label4_data = np.append(x_label4, y_label4.reshape(1000, 1), axis=1)
    label6_data = np.append(x_label6, y_label6.reshape(1000, 1), axis=1)

    all_data = np.append(np.append(label2_data,
                                   label4_data, axis=0),
                         label6_data, axis=0)

    data_x = all_data[:, :-1]
    data_y = all_data[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                        test_size=0.2,
                                                        random_state=SEED)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_x, test_x, train_y, test_y


def rbf(data_x, data_y, gamma):
    return rbf_kernel(data_x, data_y, gamma)


def build_q_mat(x, y, gamma):
    k_mat = rbf(x, x, gamma)
    np.reshape(y, (len(y), -1))
    q_ = np.multiply(y, k_mat)
    _q = matrix(np.multiply(y, q_.T).T)
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


def dual_grad(Q, alpha):
    return np.dot(Q, alpha)-1


def compute_acc(y, y_pred):
    return accuracy_score(y, y_pred)


def configure_solver():
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-12
    solvers.options['feastol'] = 1e-12


def prepare_solver(train_x, train_y, c, gamma):
    configure_solver()
    K = rbf(train_x, train_x, gamma)
    np.reshape(train_y, (len(train_y), -1))
    Q1 = np.multiply(train_y, K)
    Q2 = np.multiply(train_y, Q1.T)
    Q = matrix(Q2.T)

    p = matrix(np.repeat(-1, len(train_x)).reshape(len(train_x), 1), tc='d')

    A = matrix(train_y, (1, len(train_y)), tc='d')
    b = matrix(0, tc='d')

    first_constraint = np.diag([-1] * len(train_y))
    first_limit = np.array([0] * len(train_y))

    second_constraint = np.diag([1] * len(train_y))
    second_limit = np.array([c] * len(train_y))

    G = matrix(np.concatenate((first_constraint, second_constraint)), tc='d')
    h = matrix(np.concatenate((first_limit, second_limit)), tc='d')

    return Q, p, G, h, A, b


def compute_support_vectors(train_x, train_y, alpha):
    ind = np.where(np.any(alpha > 1e-5, axis=1))
    sv_x = train_x[ind]
    sv_y = ((train_y[ind]).T).reshape((-1, 1))
    alpha_star = alpha[ind]
    return sv_x, sv_y, alpha_star


def compute_bstar(sv_x, sv_y, gamma, alpha_star):
    return np.mean((1 - sv_y * sum(np.multiply(rbf(sv_x, sv_x, gamma),
                                               np.multiply(alpha_star, sv_y)))) / sv_y)


def compute_predictions(train_x, sv_x, sv_y, alpha_star, gamma, test_x, b_star):
    y_train_pred = np.sign(((np.multiply(rbf(sv_x, train_x, gamma),
                                         np.multiply(alpha_star, sv_y))).sum(axis=0)).reshape((-1, 1)) + (np.repeat(b_star, len(train_x), axis=0)).reshape((-1, 1)))
    y_test_pred = np.sign(((np.multiply(rbf(sv_x, test_x, gamma),
                                        np.multiply(alpha_star, sv_y))).sum(axis=0)).reshape((-1, 1)) + (np.repeat(b_star, len(test_x), axis=0)).reshape((-1, 1)))

    return y_train_pred, y_test_pred


def svm_classifier(train_x, train_y, c, gamma, test_x, test_y):
    Q, p, G, h, A, b = prepare_solver(train_x, train_y, c, gamma)

    # Solve
    tik = time.time()
    sol = solvers.qp(Q, p, G, h, A, b)
    tok = time.time()

    computational_time = tok - tik

    # Take alpha from the solution
    alpha = np.array(sol['x'])

    # Support vectors are ones corresponding to alpha values greater than 1e-5
    sv_x, sv_y, alpha_star = compute_support_vectors(train_x, train_y, alpha)

    funct_eval = sol["iterations"]
    final_obj = sol['primal objective']
    # KKT condition violation
    primal_inf = sol["primal infeasibility"]
    dual_inf = sol["dual infeasibility"]
    primal_slack = sol["primal slack"]
    dual_slack = sol["dual slack"]

    b_star = compute_bstar(sv_x, sv_y, gamma, alpha_star)

    y_train_pred, y_test_pred = compute_predictions(train_x, sv_x, sv_y,
                                                    alpha_star, gamma,
                                                    test_x, b_star)
    acc_train = compute_acc(train_y, y_train_pred)
    acc_test = compute_acc(test_y, y_test_pred)

    hyperparameters = {'c': c, 'gamma': gamma}
    results = {'computational_time': computational_time,
               'final_obj': final_obj, 'funct_eval': funct_eval,
               'primal_inf': primal_inf, 'dual_inf': dual_inf,
               'primal_slack': primal_slack, 'dual_slack': dual_slack,
               'acc_train': acc_train, 'acc_test': acc_test}
    suport_vectors = {'sv_x': sv_x, 'sv_y': sv_y,
                      'b_star': b_star, 'alpha_star': alpha_star}

    return hyperparameters, results, suport_vectors, alpha


def log_training_info(hyperparams, results):
    print("__________" * 5)
    print(f"C: {hyperparams['c']}, Gamma: {hyperparams['gamma']}")
    print(f"Acc Training Set: {round(results['acc_train'], 5) * 100}%")
    print(f"Acc Test Set: {round(results['acc_test'], 5) * 100}%")
    print(f"Number of function evaluations: {results['funct_eval']}")
    print(
        f"Obj. fun final val (dual problem): {round(results['final_obj'], 5)}")
    print(f"KKT violation: {round(results['kkt_violation'], 5)}")
    print(f"Time: {round(results['computational_time'], 5)} seconds")


def train_onevsone_classifier(c, gamma, tolerance=1e-5):
    train_x, test_x, train_y, test_y = load_mnist()
    train_data = np.append(train_x, train_y.reshape(len(train_y), 1), axis=1)
    test_data = np.append(test_x, test_y.reshape(len(test_y), 1), axis=1)

    tik = time.time()
    acc_train_classifier, kkt_classifier = [], []
    fun_evals_classifier, classifiers = [], []
    combos = [(2, 4), (2, 6), (4, 6)]
    for idx, combo in enumerate(combos):
        print("__________" * 5)
        print(f"Classes Combination: {combo}")
        lbl1, lbl2 = combo[0], combo[1]
        classifier_data = train_data[np.where(
            (train_data[:, -1] == lbl1) | (train_data[:, -1] == lbl2))]
        data_x = classifier_data[:, :-1]
        data_y = classifier_data[:, -1].reshape((-1, 1))
        classes = np.unique(data_y)
        # one classes is -1, other is +1
        data_y = np.where(data_y == classes[0], -1, 1)
        train_x_classifier = data_x
        train_y_classifier = data_y

        # testing data for 2 classes (each combination of class pairs)
        test_classifier_data = test_data[np.where(
            (test_data[:, -1] == lbl1) | (test_data[:, -1] == lbl2))]
        _data_x = test_classifier_data[:, :-1]
        _data_y = test_classifier_data[:, -1].reshape((-1, 1))
        # one class is -1, other is +1
        _data_y = np.where(_data_y == classes[0], -1, 1)
        test_x_classifier = _data_x
        test_y_classifier = _data_y

        # train classifier
        hyperparams, res, sv, alpha = svm_classifier(train_x_classifier,
                                                     train_y_classifier,
                                                     c, gamma,
                                                     test_x_classifier,
                                                     test_y_classifier)

        sv_x, sv_y = sv.get('sv_x'), sv.get('sv_y')
        b_star, alpha_star = sv.get('b_star'), sv.get('alpha_star')

        # storing the output of SVM for each classifier
        classifiers.append((sv_x, sv_y, alpha_star, b_star, combos[idx]))

        # Calculate KKT violation
        r_alpha, s_alpha = compute_r_s_vectors(alpha, train_y_classifier,
                                               c, gamma, tolerance)

        gradient = -np.multiply(dual_grad(build_q_mat(train_x_classifier,
                                                      train_y_classifier,
                                                      gamma),
                                          alpha),
                                train_y_classifier.reshape(-1, 1))

        m, M = max(np.take(gradient, r_alpha)), min(np.take(gradient, s_alpha))
        kkt_violation = m-M
        res['kkt_violation'] = kkt_violation

        acc_train_classifier.append(res['acc_train'])
        kkt_classifier.append(res['kkt_violation'])
        fun_evals_classifier.append(res['funct_eval'])

        log_training_info(hyperparams, res)
    '''
    Train Classifiers separately
    '''
    # Number of classes = 3 (class 2, 4, 6)
    res_matrix_train = np.zeros((len(train_y), 3))
    res_matrix_test = np.zeros((len(test_y), 3))
    for idx, classifier in enumerate(classifiers):
        sv_x, sv_y, alpha_star, b_star, classes_combination = classifier
        y_train_pred = np.sign(((np.multiply(rbf(sv_x, train_x, gamma),
                                             np.multiply(alpha_star, sv_y))).sum(axis=0)).reshape((-1, 1)) + (np.repeat(b_star, len(train_x), axis=0)).reshape((-1, 1)))

        y_test_pred = np.sign(((np.multiply(rbf(sv_x, test_x, gamma),
                                            np.multiply(alpha_star, sv_y))).sum(axis=0)).reshape((-1, 1)) + (np.repeat(b_star, len(test_x), axis=0)).reshape((-1, 1)))
        lbl1, lbl2 = classes_combination[0], classes_combination[1]

        res_matrix_train[:, idx] = np.where(y_train_pred == -1,
                                            lbl1,
                                            lbl2).reshape(len(y_train_pred),)
        res_matrix_test[:, idx] = np.where(y_test_pred == -1,
                                           lbl1,
                                           lbl2).reshape(len(y_test_pred),)

    res_df_tr = pd.DataFrame({'Classifier1': res_matrix_train[:, 0],
                              'Classifier2': res_matrix_train[:, 1],
                              'Classifier3': res_matrix_train[:, 2]})
    y_train_pred_final = res_df_tr.mode(axis=1)
    y_train_pred_final = y_train_pred_final.loc[:, 0].values

    res_df = pd.DataFrame({'Classifier1': res_matrix_test[:, 0],
                           'Classifier2': res_matrix_test[:, 1],
                           'Classifier3': res_matrix_test[:, 2]})

    y_test_pred_final = res_df.mode(axis=1)
    y_test_pred_final = y_test_pred_final.loc[:, 0].values

    acc_train = compute_acc(train_y, y_train_pred_final)
    acc_test = compute_acc(test_y, y_test_pred_final)
    tok = time.time()

    print("----------" * 5)
    print("Classifiers combined")
    print(f"C value: {c}, Gamma: {gamma}")
    print(f"Training acc: {round(acc_train, 5) * 100}%")
    print(f"Testing acc: {round(acc_test, 5) * 100} %")
    print(f"Number of function evaluations: {int(np.sum(fun_evals_classifier))}")
    print(f"KKT violation: {round(np.mean(kkt_classifier), 5)}")
    print(f"Time: {np.round(tok - tik, 5)} seconds")
    print("----------" * 5)


if __name__ == "__main__":
    # TODO: REPLACE WITH BEST HYPERPARAMs
    train_onevsone_classifier(c=1, gamma=1e-5, tolerance=1e-5)
