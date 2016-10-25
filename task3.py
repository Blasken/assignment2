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

def RBF(k):
    min_Cv = np.Inf
    average_Cvs = []
    for z in range(1):
        nr_of_input_units = 2
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
                y = np.tanh(B*(W2.T.dot(g_) - b))[0, 0]  # forward propagation
                dW = - 0.5 * (c - y) * (1 - y**2) * g_  # backward propagation
                db = - 0.5 * (c - y) * (1 - y**2) * (-1.0)
                W2 -= n*dW
                b -= n*db

        # validation
        Cvs = []
        for i in validation_data:
            (point, c) = task3[i]
            x = np.array(list(point))
            y = np.tanh(B * (W2.T.dot(g(x, W1)) - b))[0, 0]  # forward propagation
            Cvs.append(0.5*np.abs((c-y)))
        average_Cv = np.average(Cvs)
        average_Cvs.append(average_Cv)
        if average_Cv < min_Cv:
            # save
            min_Cv = average_Cv
            W1_ = W1
            W2_ = W2
            b_ = b

    np.save('W1.npy', W1_)
    np.save('W2.npy', W2_)
    np.save('b.npy', b_)
    print(min_Cv)
    return average_Cvs

def decision_boundary():
    W1_ = np.load('W1.npy')
    W2_ = np.load('W2.npy')
    b_ = np.load('b.npy')

    x1 = np.arange(-15, 25, 0.05)
    x2 = np.arange(-15, 25, 0.05)
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

    Z = np.round(Z)
    A = np.where(Z == 0.0)
    B = zip(A[0], A[1])
    points = []
    for x, y in B:
        x_ = X1[x, y]
        y_ = X2[x, y]
        points.append([x_, y_])

    # failing points...
    points2 = [points[0]]
    for i in range(len(points)-1):
        if np.linalg.norm(np.array(points[i+1])-np.array(points[i])) < 0.25:
            points2 += [points[i+1]]

    C = np.array(points2)
    D = C[C[:, 0].argsort()]
    np.save('D.npy', D)


def plot():
    D = np.load('D.npy')

    plt.figure()
    line = plt.plot(D[:, 0].T, D[:, 1].T, color='black', label='decision boundary')

    temp = np.loadtxt('task3.txt')
    class1 = temp[np.where(temp[:, 0] == 1.0), 1:][0]
    class2 = temp[np.where(temp[:, 0] == -1.0), 1:][0]

    scatter1 = plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='+ 1')
    scatter2 = plt.scatter(class2[:, 0], class2[:, 1], color='red', label='- 1')

    W = np.load('W1.npy')
    scatter3 = plt.scatter(W[:, 0], W[:, 1], color='black', marker='^', edgecolor='black', facecolors='yellow', label='weigths') #  linewidths=3.0

    plt.ylabel('$x_2$')
    plt.xlabel('$x_1$')

    plt.legend(handles=[scatter1, scatter2, line[0], scatter3])

    plt.show()


def task3c():
    ks = [1, 2, 3, 4, 5] #, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    result = []
    for k in ks:
        average_cvs = RBF(k)
        result += [average_cvs]
    print(np.array(result))


def task3a():
    print('Training network...')
    RBF(5)
    print('Creating decision boundary...')
    decision_boundary()
    print('Plotting..')
    plot()


def task3b():
    print('Training network...')
    RBF(10)
    print('Creating decision boundary...')
    decision_boundary()
    print('Plotting..')
    plot()

task3b()
