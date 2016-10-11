import numpy as np

# gaussian
def neighbourhood_g(W, i0, sigma):
    """
    Gaussian neighbourhood function, see lecture notes.
    Distance between output neurons is the distance between indices in the output matrix.
    :param W:
    :param i0:
    :param sigma:
    :return:
    """
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    ij = np.array([row, column])
    indices = np.indices((W.shape[0], W.shape[1])).T
    lattice_distance = np.sum((indices - ij) ** 2, 2)
    return np.exp(- lattice_distance / (2 * sigma ** 2))

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

def kohonen(data, W, T, sigma, n):
    """
    Updating weights using kohonens algorithm,
    :math:`\\mathbf{W_v}(s+1)=\\mathbf{W_v}(s)+\\alpha(s)...`

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
    :returns: updated weights

    .. note:: this is a bit overkill for a comment, but params should be there
    .. todo:: fix this comment...
    """
    for t in range(T):
        for i in np.random.choice(range(len(data)), len(data), replace=False):
            x = data[i]
            i0 = np.argmax(W.dot(x))
            dW = n(t) * neighbourhood_s(W, i0, sigma(t)) * (x - W).T
            W += dW.T
    return W

def ordering_phase(data, W, T_order, tau, sigma_0, n_0):
    """
    The ordering phase using kohonen's algorithm.
    Width, sigma(t), decreases with time.
    Learning rate, n(t), decreases with time.

    :param data: input
    :param W: weights
    :param T_order: number of iterations
    :param tau: int/float
    :param sigma_0: int/float
    :param n_0: int/float
    :return: updated W
    """
    sigma = lambda t: sigma_0 * np.exp(-t / tau)
    n = lambda t: n_0 * np.exp(-t / tau)
    return kohonen(data, W, T_order, sigma, n)

def convergence_phase(data, W, T_conv, tau, sigma_conv, n_conv):
    """
    The convergence phase using kohonen's algorithm.
    :param data: input
    :param W: weights
    :param T_conv: number of iterations
    :param tau: not used.
    :param sigma_conv: width of domain, constant
    :param n_conv: learning rate, constant
    :return: updated W
    """
    sigma = lambda t: sigma_conv
    n = lambda t: n_conv
    return kohonen(data, W, T_conv, sigma, n)

