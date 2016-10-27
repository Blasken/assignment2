import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from data import wine_data_n
from kohonen import ordering_phase, convergence_phase

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data = wine_data_n.copy()[:, 1:14]
np.random.shuffle(data)

nr_of_input_units = 13
nr_of_output_units = (20, 20)
W_shape = (nr_of_output_units[0], nr_of_output_units[1], nr_of_input_units)
W = np.random.normal(loc=0.0, scale=0.05, size=W_shape)

tau = 300.0
sigma_0 = 30
n_0 = 0.1
T_order = 1000
ordering_phase(data, W, T_order, tau, sigma_0, n_0)

# convergence phase
tau = 300.0
sigma_conv = 0.9
n_conv = 0.01
T_conv = 200
convergence_phase(data, W, T_conv, tau, sigma_conv, n_conv)

plt.figure()
colors = {1: 'blue', 2: 'black', 3: 'red'}
markers = {1: 'o', 2: 'x', 3: 's'}
legends = {}
for x in wine_data_n:
    i0 = np.argmin(norm(W - x[1:14], axis=2))
    (row, column) = np.unravel_index(i0, (W.shape[0], W.shape[1]))
    legends[x[0]] = plt.scatter(row+1, column+1, marker=markers[x[0]], color=colors[x[0]], label=str(int(x[0])))

plt.legend(handles=[legends[1], legends[2], legends[3]])
plt.xlim(0, 21)
plt.ylim(0, 21)
plt.ylabel('m')
plt.xlabel('n')
plt.grid()
plt.show()
plt.savefig('task2.png')
