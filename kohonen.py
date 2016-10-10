import numpy as np

def kohonen(data, W, T, sigma, n):
    for t in range(T):
        x = data[np.random.choice(range(len(data)), None)]
        i0 = np.argmax(W.dot(x))
        row = i0 // W.shape[1]
        column = i0 % W.shape[1]
        neighbourhood = np.exp(- np.sum((W - W[row, column]) ** 2) / (2 * sigma(t) ** 2))
        dW = n(t) * neighbourhood * (x - W)
        W += dW
    return W

def ordering_phase(data, W, T_order, tau, sigma_0, n_0):
    sigma = lambda t: sigma_0 * np.exp(-t / tau)
    n = lambda t: n_0 * np.exp(-t / tau)
    return kohonen(data, W, T_order, sigma, n)

def convergence_phase(data, W, T_conv, tau, sigma_conv, n_conv):
    sigma = lambda t: sigma_conv
    n = lambda t: n_conv
    return kohonen(data, W, T_conv, sigma, n)

