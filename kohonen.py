import numpy as np
import matplotlib.pyplot as plt
from data import triangle_data

# Init weights
nr_of_input_units = 2
nr_of_output_units = 100
W_shape = (nr_of_output_units, nr_of_input_units)
W = np.random.normal(loc=0.5, scale=0.01, size=W_shape)

plt.figure()
plt.scatter(W[:, 0], W[:, 1], color='black', marker='*')
plt.ylim(ymin=0.0, ymax=1.0)
plt.xlim(xmin=0.0, xmax=1.0)
plt.show()

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
print(W)
plt.figure()
plt.scatter(W[:, 0], W[:, 1], color='black', marker='*')
plt.ylim(ymin=0.0, ymax=1.0)
plt.xlim(xmin=0.0, xmax=1.0)
plt.show()

# convergence phase
sigma_conv = 0.9
n_conv = 0.01
T_conv = 50000
for _ in range(T_conv):
    break

