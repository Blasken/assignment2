import numpy as np
from numpy.linalg import norm

# gaussian
def neighbourhood_g(W, i0, sigma):
    """
    Gaussian neighbourhood function, see lecture notes.
    Distance between output neurons is the distance between indices in the output matrix.
    The gaussian neighbourhood function is on p. 177 in lecture notes.
    :param W:
    :param i0:
    :param sigma:
    :return:
    """
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    ij = np.array([row, column])
    indices = np.indices((W.shape[0], W.shape[1])).T
    lattice_distance = norm(indices - ij, axis=2) ** 2 #  np.sum((indices - ij) ** 2, 2)
    return np.exp(- lattice_distance.T / (2 * sigma ** 2))

# step
def neighbourhood_s(W, i0, sigma):
    """
    TODO: this is just a test but this code could be written in a smarter/shorter way... LOL
    Neighbourhood is 1.0 for i0 and 0.5 for the closest units to i0 in the output matrix.
    Neighbourhood is 0.0 for all other nodes.

    :param W:
    :param i0:
    :param _:
    :return:
    """
    i = i0 // W.shape[1]
    j = i0 % W.shape[1]

    N = np.zeros((W.shape[0], W.shape[1]))
    N[i,j] = 1.0
    if j - 1 >= 0:
        N[i,j-1] = 0.5
    if j + 1 < W.shape[1]:
        N[i,j+1] = 0.5
    if i - 1 >= 0:
        N[i-1,j] = 0.5
        if j - 1 >= 0:
            N[i-1,j-1] = 0.5
        if j + 1 < W.shape[1]:
            N[i-1,j+1] = 0.5
    if i+1 < W.shape[0]:
        N[i+1,j] = 0.5
        if j-1 >= 0:
            N[i+1,j-1] = 0.5
        if j + 1 < W.shape[1]:
            N[i+1,j+1] = 0.5
    return N.T

def kohonen(data, W, T, sigma, n, normalise):
    """
    For each iteration, take a random input and update
    weights using kohonen's algorithm. Update rule for kohonen's algorithm is on
    p.175 in lecture notes.

    :math:`\\mathbf{\partial W} = \\mathbf{W_v}(s)+\\alpha(s)...`

    :param data: data to apply algorithm on
    :param W: Weights for algorithm
    :param T: T timesteps?
    :param sigma: sigma value for algorithm
    :param n: n-values for algorithm?
    :type data: numpy array
    :type W: numpy array
    :type T: integer
    :type sigma: float
    :type n: 1-d numpy array
    :type normalise: boolean
    :returns: updated weights

    .. note:: this is a bit overkill for a comment, but params should be there
    .. todo:: fix this comment...
    """
    for t in range(T):
        #rand = np.random.choice(range(len(data)), len(data), replace=False)
        #for i in rand:
        for x in data:
            #x = data[i]
            #i0 = np.argmax(W.dot(x))
            i0 = np.argmin(norm(W-x, axis=2))
            dW = n(t) * neighbourhood_g(W, i0, sigma(t)).T * (x - W).T
            W += dW.T
            #if normalise:
            #    W.T.dot(1/norm(W, axis=2))
    return W

def ordering_phase(data, W, T_order, tau, sigma_0, n_0, normalise=True):
    """
    The ordering phase using kohonen's algorithm.
    Domain width, sigma(t), decreases with time.
    Learning rate, n(t), decreases with time.

    :param data: input
    :param W: weights
    :param T_order: number of iterations
    :param tau: int/float
    :param sigma_0: int/float
    :param n_0: int/float
    :type normalise: boolean
    :return: updated W
    """
    sigma = lambda t: sigma_0 * np.exp(-t / tau)
    n = lambda t: n_0 * np.exp(-t / tau)
    return kohonen(data, W, T_order, sigma, n, normalise)

def convergence_phase(data, W, T_conv, tau, sigma_conv, n_conv, normalise=True):
    """
    The convergence phase using kohonen's algorithm.
    :param data: input
    :param W: weights
    :param T_conv: number of iterations
    :param tau: not used.
    :param sigma_conv: width of domain, constant
    :param n_conv: learning rate, constant
    :type normalise: boolean
    :return: updated W
    """
    sigma = lambda t: sigma_conv
    n = lambda t: n_conv
    return kohonen(data, W, T_conv, sigma, n, normalise)

