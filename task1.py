import numpy as np
import matplotlib.pyplot as plt
from data import triangle_data, plot_triangle
from kohonen import ordering_phase, convergence_phase

sigmas = {'a': 100,'b':5}
for q in ['a','b']:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_xlim((0.0,1.0))
    ax1.set_ylim((0.0,1.0))
    ax2.set_xlim((0.0,1.0))
    ax2.set_ylim((0.0,1.0))
    ax3.set_xlim((0.0,1.0))
    ax3.set_ylim((0.0,1.0))

    # Init weights
    nr_of_input_units = 2
    nr_of_output_units = (1,100)
    W_shape = (nr_of_output_units[0], nr_of_output_units[1], nr_of_input_units)
    W = np.random.normal(loc=0.5, scale=0.01, size=W_shape)
    #W = np.array([[[x,0.5] for x in np.arange(0.25, 0.75, 0.005)]])
    ax1.scatter(W[:,:,0], W[:,:,1], color='black', marker='*')
    plot_triangle(ax1)

    # ordering phase
    tau = 200.0
    sigma_0 = sigmas[q]
    n_0 = 0.1
    T_order = 1000
    ordering_phase(triangle_data, W, T_order, tau, sigma_0, n_0)

    ax2.scatter(W[:, :, 0], W[:, :, 1], color='black', marker='*')
    plot_triangle(ax2)

    # convergence phase
    tau = 200.0
    sigma_conv = 0.9
    n_conv = 0.01
    T_conv = 5000
    convergence_phase(triangle_data, W, T_conv, tau, sigma_conv, n_conv)

    ax3.scatter(W[:, :, 0], W[:, :, 1], color='black', marker='*')
    plot_triangle(ax3)

    #plt.show()
    plt.savefig('task1' + q + '.png')
