import numpy as np
import matplotlib.pyplot as plt
from data import triangle_data

sigma_0 = 100.0
sigma_conv = 0.9

n_0 = 0.1
n_conv = 0.01

tau = 200.0

T_order = 1000
T_conv = 50000

sigma = lambda t: sigma_0 * np.exp(-t/tau)
n = lambda t: n_0 * np.exp(-t/tau)


