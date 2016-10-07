import numpy as np

# a)
# The distribution is uniform inside an equilateral
# triangle with unit side length, and zero outside. To generate the input
# data set, draw 1000 points uniformly distributed within this triangle:
# http://mathworld.wolfram.com/TrianglePointPicking.html


# b)
wine_data = np.loadtxt('wine.data.txt', delimiter=',')
print('Data shape' + str(data.shape))
print('Attribute 1: class')
print('Attribute 14: unknown and unused')
print('')

# classes
class1 = wine_data[np.where(wine_data[:, 0] == 1), 1:13][0]
class2 = wine_data[np.where(wine_data[:, 0] == 2), 1:13][0]
class3 = wine_data[np.where(wine_data[:, 0] == 3), 1:13][0]
print('  class 1 ' + str(class1.shape))
print('  class 2 ' + str(class2.shape))
print('  class 3 ' + str(class3.shape))


