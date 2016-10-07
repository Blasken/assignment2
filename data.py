import numpy as np
import matplotlib.pyplot as plt

# a)
# The distribution is uniform inside an equilateral
# triangle with unit side length, and zero outside. To generate the input
# data set, draw 1000 points uniformly distributed within this triangle:
# http://mathworld.wolfram.com/TrianglePointPicking.html
# and
# http://math.stackexchange.com/questions/51326/determining-if-an-arbitrary-point-lies-inside-a-triangle-defined-by-three-points
A = np.array([0, 0]).reshape(2, 1)
B = np.array([1, 0]).reshape(2, 1)
C = np.array([0.5, np.sqrt(3)/2]).reshape(2, 1)
v_1 = np.random.uniform(low=0.0, high=1.0, size=1000)
v_2 = np.random.uniform(low=0.0, high=1.0, size=1000)
P = B*v_1 + C*v_2
#AB =
#AC =
#BC =
AP = A-P
BP = B-P
CP = C-P
np.cross()
plt.figure()
plt.scatter(P[0, :], P[1, :], color='black', marker='*')
plt.ylim(ymin=0.0)
plt.xlim(xmin=0.0)
plt.show()

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


