import numpy as np


def load_data(user_show_file, shows_file):
    R = np.loadtxt(user_show_file)
    shows_f = open(shows_file, 'r')
    shows = shows_f.readlines()
    return R, shows


def item_item_cf(R, shows, user_id, n_hidden_item, top_k):
    R[user_id, :n_hidden_item] = 0
    item_degree = np.sum(R, axis=0)
    Q = np.diag(item_degree)
    Q_inverse_norm = np.where(Q == 0, Q, Q ** (-1/2))
    gamma = np.dot(R, Q_inverse_norm)
    gamma = np.dot(gamma, R.T)
    gamma = np.dot(gamma, R)
    gamma = np.dot(gamma, Q_inverse_norm)
    values = [(-v, i) for i, v in enumerate(gamma[user_id, :n_hidden_item])]
    values = sorted(values)
    top_k_idx = [i for v, i in values[:top_k]]
    top_k_vals = [-v for v, i in values[:top_k]]
    top_k_shows = [shows[j] for j in top_k_idx]
    return top_k_vals, top_k_shows


def user_user_cf(R, shows, user_id, n_hidden_item, top_k):
    R[user_id, :n_hidden_item] = 0
    user_degree = np.sum(R, axis=1)
    P = np.diag(user_degree)
    P_inverse_norm = np.where(P == 0, P, P ** (-1/2))
    gamma = np.dot(R.T, P_inverse_norm)
    gamma = np.dot(gamma, R)
    gamma = np.dot(gamma, R.T)
    gamma = np.dot(gamma, P_inverse_norm)
    values = [(-v, i) for i, v in enumerate(gamma[:n_hidden_item, user_id])]
    values = sorted(values)
    top_k_idx = [i for v, i in values[:top_k]]
    top_k_vals = [-v for v, i in values[:top_k]]
    top_k_shows = [shows[j] for j in top_k_idx]
    return top_k_vals, top_k_shows


if __name__ == "__main__":
    user_show_file = "q4/data/user-shows.txt"
    shows_file = "q4/data/shows.txt"
    R, shows = load_data(user_show_file, shows_file)
    user_id = 499
    n_hidden_item = 100
    top_k = 5
    top_k_vals, top_k_shows = item_item_cf(
        R, shows, user_id, n_hidden_item, top_k)
    print(top_k_vals)
    print(top_k_shows)
    print()

    top_k_vals, top_k_shows = user_user_cf(
        R, shows, user_id, n_hidden_item, top_k)
    print(top_k_vals)
    print(top_k_shows)
