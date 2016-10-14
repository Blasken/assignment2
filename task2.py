import numpy as np
import matplotlib.pyplot as plt
from data import class1_n, class2_n, class3_n
from kohonen import ordering_phase, convergence_phase

print(class1_n[0, :])
print(class2_n[0, :])
print(class3_n[0, :])

data = np.append(class1_n, class2_n, axis=0)
data = np.append(data, class3_n, axis=0)
print(data.shape)

nr_of_input_units = 13
nr_of_output_units = (20, 20)
W_shape = (nr_of_output_units[0], nr_of_output_units[1], nr_of_input_units)
W = np.random.normal(loc=0.5, scale=0.01, size=W_shape)

tau = 300.0
sigma_0 = 30
n_0 = 0.1
T_order = 1000
ordering_phase(data, W, T_order, tau, sigma_0, n_0)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(W[:, :, 0], W[:, :, 1], color='black', marker='*')

# convergence phase
tau = 300.0
sigma_conv = 0.9
n_conv = 0.01
T_conv = 2000
convergence_phase(data, W, T_conv, tau, sigma_conv, n_conv)

ax2.scatter(W[:, :, 0], W[:, :, 1], color='black', marker='*')

for x in class1_n:
    i0 = np.argmax(W.dot(x))
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    w = W[row, column]
    ax2.scatter(w[0], w[1], color='blue')

for x in class2_n:
    i0 = np.argmax(W.dot(x))
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    w = W[row, column]
    ax2.scatter(w[0], w[1], color='green')

for x in class3_n:
    i0 = np.argmax(W.dot(x))
    row = i0 // W.shape[1]
    column = i0 % W.shape[1]
    w = W[row, column]
    ax2.scatter(w[0], w[1], color='red')

plt.show()

