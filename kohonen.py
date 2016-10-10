import numpy as np
import matplotlib.pyplot as plt
from data import triangle_data, plot_triangle

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_xlim((0.0,1.0))
#ax1.set_xticks(np.arange(0.0, 1.0 + 0.1, 0.1))
ax1.set_ylim((0.0,1.0))
#ax1.set_yticks(np.arange(0.0, 1.0 + 0.1, 0.1))
ax2.set_xlim((0.0,1.0))
#ax2.set_xticks(np.arange(0.0, 1.0 + 0.1, 0.1))
ax2.set_ylim((0.0,1.0))
#ax2.set_yticks(np.arange(0.0, 1.0 + 0.1, 0.1))
ax3.set_xlim((0.0,1.0))
#ax3.set_xticks(np.arange(0.0, 1.0 + 0.1, 0.1))
ax3.set_ylim((0.0,1.0))
#ax3.set_yticks(np.arange(0.0, 1.0 + 0.1, 0.1))

# Init weights
nr_of_input_units = 2
nr_of_output_units = 100
W_shape = (nr_of_output_units, nr_of_input_units)
W = np.random.normal(loc=0.5, scale=0.01, size=W_shape)

ax1.scatter(W[:, 0], W[:, 1], color='black', marker='*')

# ordering phase
tau = 200.0

sigma_0 = 100.0
sigma = lambda t: sigma_0 * np.exp(-t/tau)
n_0 = 0.1
n = lambda t: n_0 * np.exp(-t/tau)

T_order = 1000
for t in range(T_order):
    x = triangle_data[np.random.choice(range(len(triangle_data)), None)]
    i0 = np.argmax(W.dot(x))
    neighbourhood = np.exp(- np.sum((W - W[i0])**2) / (2 * sigma(t) ** 2))
    dW = n(t) * neighbourhood * (x-W)
    W += dW

ax2.scatter(W[:, 0], W[:, 1], color='black', marker='*')


# convergence phase
sigma_conv = 0.9
n_conv = 0.01
T_conv = 50000
for _ in range(T_conv):
    x = triangle_data[np.random.choice(range(len(triangle_data)), None)]
    i0 = np.argmax(W.dot(x))
    neighbourhood = np.exp(- np.sum((W - W[i0])**2) / (2 * sigma_conv ** 2))
    dW = n_conv * neighbourhood * (x-W)
    W += dW

ax3.scatter(W[:, 0], W[:, 1], color='black', marker='*')

#plt.show()
plt.savefig('task1a.png')
