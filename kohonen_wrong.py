import numpy as np

def kohonen(data, W, T, sigma, n):
    for t in range(T):
        for x in data:
            #x = data[t % len(data)]
            i0 = np.argmax(W.dot(x))
            #print(W.flatten().shape)
            #print((np.sum((W - x) ** 2, axis=2)).shape)
            #i0 = np.argmax(np.sum((W - x) ** 2, axis=2))
            row = i0 // W.shape[1]
            column = i0 % W.shape[1]
            #print((np.sum((W - W[row, column]) ** 2, axis=2)).shape)
            distance = np.sum((W - W[row, column]) ** 2, axis=2).T
            #print('distance')
            #print(distance.shape)
            neighbourhood = np.exp(- distance / (2 * sigma(t) ** 2))
            #print('neighbourhood')
            #print(neighbourhood.shape)
            #print((x - W).shape)
            dW = n(t) * neighbourhood * (x - W).T
            #print(dW.shape)
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

