import numpy as np

# gaussian
def neighbourhood_g(W, i0, sigma):
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    ij = np.array([row, column])
    indices = np.indices((W.shape[0], W.shape[1])).T
    lattice_distance = np.sum((indices - ij) ** 2, 2)
    return np.exp(- lattice_distance / (2 * sigma ** 2))

# step
def neighbourhood_g(W, i0, sigma):
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    ij = np.array([row, column])
    indices = np.indices((W.shape[0], W.shape[1])).T
    lattice_distance = np.sum((indices - ij) ** 2, 2)
    return np.exp(- lattice_distance / (2 * sigma ** 2))

def kohonen(data, W, T, sigma, n):
    for t in range(T):
        for x in data:
            #x = data[np.random.choice(range(len(data)), None)]
            i0 = np.argmax(W.dot(x))
            dW = n(t) * neighbourhood_g(W, i0, sigma(t)) * (x - W).T
            W += dW.T
    return W

def ordering_phase(data, W, T_order, tau, sigma_0, n_0):
    sigma = lambda t: sigma_0 * np.exp(-t / tau)
    n = lambda t: n_0 * np.exp(-t / tau)
    return kohonen(data, W, T_order, sigma, n)

def convergence_phase(data, W, T_conv, tau, sigma_conv, n_conv):
    sigma = lambda t: sigma_conv
    n = lambda t: n_conv
    return kohonen(data, W, T_conv, sigma, n)

