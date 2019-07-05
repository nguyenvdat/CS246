import numpy as np
import matplotlib.pyplot as plt


def get_grad(w, b, x, y, C):
    sign = y * (np.dot(x, w) + b) < 1  # (bs)
    dw_batch = sign.reshape(-1, 1) * (-y.reshape(-1, 1) * x)  # (bs, d_feature)
    dw = C * np.sum(dw_batch, axis=0) + w  # (d_feature)
    db = C * np.sum(sign * (-y))
    return dw, db


def get_cost(w, b, x, y, C):
    score = 1 - y * (np.dot(x, w) + b)  # (bs)
    hinge_loss = np.sum(np.where(0 > score, 0, score))
    return 1/2*np.dot(w, w) + C * hinge_loss


def load_data():
    x = np.loadtxt('q1/data/features.txt', delimiter=',')
    y = np.loadtxt('q1/data/target.txt')
    return x, y


def train_batch_gd(x, y, C, eta, epsilon):
    w = np.zeros(x.shape[1])
    b = 0
    prev_cost = get_cost(w, b, x, y, C)
    costs = [prev_cost]
    while True:
        dw, db = get_grad(w, b, x, y, C)
        w -= eta * dw
        b -= eta * db
        cost = get_cost(w, b, x, y, C)
        costs.append(cost)
        if np.abs(prev_cost - cost) * 100 / prev_cost < epsilon:
            break
        prev_cost = cost
    return w, b, costs


def train_sgd(x, y, C, eta, epsilon):
    n = len(x)
    random_idx = np.random.permutation(n)
    x = x[random_idx]
    y = y[random_idx]
    w = np.zeros(x.shape[1])
    b = 0
    prev_cost = get_cost(w, b, x, y, C)
    costs = [prev_cost]
    i = 0
    prev_delta = 0
    while True:
        dw, db = get_grad(w, b, x[i], y[i], C)
        w -= eta * dw
        b -= eta * db
        cost = get_cost(w, b, x, y, C)
        costs.append(cost)
        delta_p = np.abs(prev_cost - cost) * 100 / prev_cost
        delta = 0.5 * prev_delta + 0.5 * delta_p
        if delta < epsilon:
            break
        prev_cost = cost
        prev_delta = delta
        i = (i + 1) % n
    return w, b, costs


def train_minibatch_gd(x, y, C, eta, epsilon, batch_size):
    n = len(x)
    random_idx = np.random.permutation(n)
    x = x[random_idx]
    y = y[random_idx]
    w = np.zeros(x.shape[1])
    b = 0
    prev_cost = get_cost(w, b, x, y, C)
    costs = [prev_cost]
    l = 0
    prev_delta = 0
    while True:
        x_batch = x[int(l * batch_size): min(n, int((l + 1) * batch_size))]
        y_batch = y[int(l * batch_size): min(n, int((l + 1) * batch_size))]
        dw, db = get_grad(w, b, x_batch, y_batch, C)
        w -= eta * dw
        b -= eta * db
        cost = get_cost(w, b, x, y, C)
        costs.append(cost)
        delta_p = np.abs(prev_cost - cost) * 100 / prev_cost
        delta = 0.5 * prev_delta + 0.5 * delta_p
        if delta < epsilon:
            break
        prev_cost = cost
        prev_delta = delta
        l = (l + 1) % (np.ceil(n / batch_size))
    return w, b, costs


def plot_cost(batch_cost, sgd_cost, minibatch_cost):
    plt.plot(np.arange(len(batch_cost)) + 1, batch_cost)
    plt.plot(np.arange(len(sgd_cost)) + 1, sgd_cost)
    plt.plot(np.arange(len(minibatch_cost)) + 1, minibatch_cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend(['batch gd', 'sgd', 'minibatch gd'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    x, y = load_data()
    C = 100
    eta = 0.0000003
    epsilon = 0.25
    _, _, batch_cost = train_batch_gd(x, y, C, eta, epsilon)

    eta = 0.0001
    epsilon = 0.001
    _, _, sgd_cost = train_sgd(x, y, C, eta, epsilon)
    # print(len(costs))
    # print(costs[-1])

    eta = 0.00001
    epsilon = 0.01
    batch_size = 20
    _, _, minibatch_cost = train_minibatch_gd(
        x, y, C, eta, epsilon, batch_size)

    plt.plot(batch_cost, sgd_cost, minibatch_cost)
