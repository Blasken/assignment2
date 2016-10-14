import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from data import task3

def g(x, W):
    """
    Function g in assignment
    :param x: input, [x1, x2], type: float
    :param W: weights in gaussian, shape: (k, 2), type: float
    :return:
    """
    normalisation = np.sum(np.exp(-norm(x-W, axis=1)**2/2))
    return np.array(list(map(lambda j: np.exp(-norm(x-W[j])**2/2) / normalisation, range(len(W))))).reshape(len(W), 1)

def RBF():
    min_Cv = np.Inf
    min_Cvs = []

    # init weights
    k = 20
    nr_of_input_units = 2
    for z in range(1):
        W1 = np.random.uniform(-1, 1, size=(k, nr_of_input_units))
        # unsupervised part: find W
        n = 0.02
        nr_of_iterations = 10**4
        for _ in range(nr_of_iterations):
            i = np.random.choice(range(len(task3)), None, replace=False)
            (point, c) = task3[i]
            x = np.array(list(point))
            j0 = np.argmax(g(x, W1))

            dW = np.zeros((k, nr_of_input_units))
            dW[j0] = n * (x - W1[j0])
            W1 += dW

        # supervised part: o = tanh(B*(W2*g(x) - b))
        rands = np.random.choice(range(len(task3)), len(task3), replace=False)
        s = round(0.7*len(task3))
        training_data = rands[:s]
        validation_data = rands[s:]

        n = 0.1
        B = 0.5
        W2 = np.random.uniform(-1, 1, size=(k, 1))
        b = np.random.uniform(-1, 1)
        nr_of_iterations = 3000

        # training
        for _ in range(nr_of_iterations):
            for i in training_data:
                (point, c) = task3[i]
                x = np.array(list(point))

                g_ = g(x, W1)
                #print(g_)
                y = np.tanh(B*(W2.T.dot(g_) - b))[0, 0]  # forward propagation
                #print('c: ' + str(c))
                #print('y: ' + str(y))
                dW = - 0.5 * (c - y) * (1 - y**2) * g_  # backward propagation
                #print('dW: ' + str(dW))
                db = - 0.5 * (c - y) * (1 - y**2) * (-1.0)
                W2 -= n*dW
                b -= n*db

        # validation
        Cvs = []
        for i in validation_data:
            (point, c) = task3[i]
            x = np.array(list(point))
            y = np.tanh(B * (W2.T.dot(g(x, W1)) - b))[0, 0]  # forward propagation
            #print('c: ' + str(c))
            #print('y: ' + str(y))
            #print(0.5*0.5*(c-y)**2)
            Cvs.append(0.5*np.abs((c-y)))
        Cv = np.average(Cvs)
        min_Cvs.append(Cv)
        if Cv < min_Cv:
            # save
            min_Cv = Cv
            W1_ = W1
            W2_ = W2
            b_ = b

    np.save('W1.npy', W1_)
    np.save('W2.npy', W2_)
    np.save('b.npy', b_)
    print(min_Cv)

def test():
    W1_ = np.load('W1.npy')
    W2_ = np.load('W2.npy')
    b_ = np.load('b.npy')

    x1 = np.arange(-25, 25, 0.01)
    x2 = np.arange(-25, 25, 0.01)
    [X1, X2] = np.meshgrid(x1, x2)

    B = 0.5
    y = lambda x1, x2: np.tanh(B * (W2_.T.dot(g(np.array([x1, x2]), W1_)) - b_))[0, 0]

    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(len(X1)):
        for j in range((len(X1))):
            x1 = X1[i, j]
            x2 = X2[i, j]
            Z[i, j] = y(x1, x2)
    np.save('Z.npy', Z)
    tol = 0.1
    print(np.where(np.abs(Z) < tol))

def plotZ():
    Z = np.load('Z.npy')
    temp = np.loadtxt('task3.txt')
    class1 = temp[np.where(temp[:, 0] == 1.0),1:][0]
    class2 = temp[np.where(temp[:, 0] == -1.0),1:][0]

    plt.figure()
    plt.scatter(class1[:,0], class1[:,1], color='blue')
    plt.scatter(class2[:,0], class2[:,1], color='red')
    tol = 0.1
    a,b = np.where(np.abs(Z) < tol)
    #plt.scatter((a-25)*0.1,(b-25)*0.1, color='black')

    x1 = np.arange(-25, 25, 0.01)
    x2 = np.arange(-25, 25, 0.01)
    [X1, X2] = np.meshgrid(x1, x2)
    plt.contourf(X1, X2, Z, 0)
    plt.show()

test()
plotZ()
