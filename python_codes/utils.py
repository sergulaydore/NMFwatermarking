import numpy as np

def modified_nmf(X, eta, iterno, k, W):  # TODO: add Hkey

    m, n = X.shape  # change orgimg with any rectang
    Hprev = np.random.uniform(size=(k, n))
    cost_list = []

    for _ in range(iterno):
        Hnext = Hprev + eta * (np.matmul(np.transpose(W), X) - np.matmul(np.matmul(np.transpose(W), W), Hprev))
        cost = np.linalg.norm(X - np.matmul(W, Hnext))
        cost_list.append(cost)
        Hprev = Hnext

    return Hnext, cost_list