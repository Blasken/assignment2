import numpy as np
import matplotlib.pyplot as plt

def plot_triangle(ax):
    """
    plot a triangle on an axis

    :param ax: pyplot axis to draw triangle on
    :type ax: pyplot axis type
    .. todo:: make it pretty
    """
    xs1 = np.arange(0,0.51,0.01)
    xs2 = np.arange(0.5,1.01,0.01)
    ax.plot(xs1, sqrt3*xs1, color='blue')
    ax.plot(xs2, -sqrt3*xs2+sqrt3, color='blue')

def plot_data(data):
    """
    scatter data on an instantiated figure

    :param data: data to scatter on the figure
    :type data: 2-dimensional array of points
    .. todo:: add the figure as argument, else this function is unnecessary
    """
    plt.scatter(data[:, 0], data[:, 1], color='black', marker='*')

# a)
# The distribution is uniform inside an equilateral
# triangle with unit side length, and zero outside. To generate the input
# data set, draw 1000 points uniformly distributed within this triangle:
points = []
sqrt3 = np.sqrt(3)
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0.5, sqrt3/2])
while len(points) < 1000:
    v_1 = np.random.uniform(low=0.0, high=1.0, size=None)
    v_2 = np.random.uniform(low=0.0, high=1.0, size=None)
    P = B*v_1 + C*v_2 # http://mathworld.wolfram.com/TrianglePointPicking.html
    if sqrt3 * P[0] >= P[1] and sqrt3 * P[0] + P[1] <= sqrt3:
        points.append(P)

triangle_data = np.array(points)

# b)
wine_data = np.loadtxt('wine.data.txt', delimiter=',')
print('Data shape' + str(wine_data.shape))

# classes
class1 = wine_data[np.where(wine_data[:, 0] == 1), 1:14][0]
class2 = wine_data[np.where(wine_data[:, 0] == 2), 1:14][0]
class3 = wine_data[np.where(wine_data[:, 0] == 3), 1:14][0]
print('  class 1 ' + str(class1.shape))
print('  class 2 ' + str(class2.shape))
print('  class 3 ' + str(class3.shape))

data = []
with open('wine.data.txt') as f:
    for line in f:
        clss, *datas = line.split(',')
        data.append((datas, clss))
task2 = np.array(data, dtype=[('data', np.float, (13,)),('class', np.int)])

#print(task2[['data','class']][0:2])
#print(task2[0:2])
#print(task2['data'])

data = []
with open('task3.txt') as f:
    for line in f:
        clss, x1, x2 = line.split()
        data.append(((float(x1),float(x2)), int(clss)))
task3 = np.array(data, dtype=[('point', [('x1', np.float), ('x2', np.float)]),
                              ('class', np.int)])

#print(task3['point'][0], task3['class'][0])
