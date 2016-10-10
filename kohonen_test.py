import numpy as np

def kohonen(data, W, T, sigma, n):
    for t in range(T):
        for x in data:
            #x = data[np.random.choice(range(len(data)), None)]
            i0 = np.argmax(W.dot(x))
            row = i0 // W.shape[1]
            column = i0 % W.shape[1]
            distance = np.sum((np.indices((W.shape[0], W.shape[1])).T - np.array([row,column]))**2, 2)
            neighbourhood = np.exp(- distance / (2 * sigma(t) ** 2))
            #print(neighbourhood.shape)
            #print((x-W).shape)
            dW = n(t) * neighbourhood * (x - W).T
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

