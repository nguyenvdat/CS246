import numpy as np
import os
import matplotlib.pyplot as plt


def get_error(P, Q, ld, data_file):
    error = 0
    with open(data_file) as f:
        line = f.readline()
        while line:
            u, i, r = line.split()
            u = int(u)
            i = int(i)
            r = float(r)
            u -= 1
            i -= 1
            error += (r - np.dot(Q[i], P[u])) ** 2
            line = f.readline()
    error += ld * (np.sum(P * P) + np.sum(Q * Q))
    f.close()
    return error


def training(P, Q, ld, eta, n_iters, data_file):
    errors = []
    for n in range(n_iters):
        with open(data_file) as f:
            line = f.readline()
            while line:
                u, i, r = line.split()
                u = int(u)
                i = int(i)
                r = float(r)
                u -= 1
                i -= 1
                grad_q = -2 * (r - np.dot(Q[i], P[u])) * P[u] + 2 * ld * Q[i]
                grad_p = -2 * (r - np.dot(Q[i], P[u])) * Q[i] + 2 * ld * P[u]
                Q[i] -= eta * grad_q
                P[u] -= eta * grad_p
                line = f.readline()
        error = get_error(P, Q, ld, data_file)
        errors.append(error)
        print('Iteration {}, error: {}'.format(n + 1, error))
        f.close()
    return P, Q, errors


def init_parameters(m, n, k):
    P = np.random.uniform(0, np.sqrt(5/k), (n, k))
    Q = np.random.uniform(0, np.sqrt(5/k), (m, k))
    return P, Q


def plot_error_vs_iteration(errors):
    plt.plot(range(1, len(errors) + 1), errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.xticks(np.arange(5, len(errors) + 1, 5.0))
    plt.title('Error over iteration')
    plt.show()


if __name__ == "__main__":
    # u_s = []
    # i_s = []
    # data_file = 'q3/data/ratings.train.txt'
    # with open(data_file) as f:
    #     line = f.readline()
    #     while line:
    #         u, i, r = line.split()
    #         u = int(u)
    #         i = int(i)
    #         r = float(r)
    #         u_s.append(u)
    #         i_s.append(i)
    #         line = f.readline()
    # f.close()
    # print(max(u_s))
    # print(max(i_s))
    # print(min(u_s))
    # print(min(i_s))

    m = 1682
    n = 943
    k = 20
    ld = 0.1
    eta = 0.009
    n_iters = 40
    data_file = 'q3/data/ratings.train.txt'
    P, Q = init_parameters(m, n, k)
    P, Q, errors = training(P, Q, ld, eta, n_iters, data_file)
    plot_error_vs_iteration(errors)
