import os

os.chdir(os.path.dirname(__file__))  # Set diectory to current location

import pickle

import matplotlib.pyplot as plt
import numpy as np

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]
with open('sol_outer.data', 'rb') as file:
    sol_outer = pickle.load(file)

labels_x = (r'$h_g$ [kft]', r'$V$ [ft/s]', r'$\gamma$ [deg]')
plt_gains_x = (1e3, 1., 180./np.pi)
fig_states, axes_states = plt.subplots(nrows=3, ncols=2)

for idx, axes_x_lam in enumerate(axes_states):
    axes_x_lam[0].grid(zorder=-1)
    axes_x_lam[0].set_xlabel(r'$t$ [s]')
    axes_x_lam[0].set_ylabel(labels_x[idx])
    axes_x_lam[0].plot(sol_outer.t, sol_outer.x[idx, :] * plt_gains_x[idx], '--')
    axes_x_lam[0].plot(sol.t, sol.x[idx, :] * plt_gains_x[idx])

    axes_x_lam[1].grid(zorder=-1)
    axes_x_lam[1].set_xlabel(r'$t$ [s]')
    axes_x_lam[1].set_ylabel(r'$(\lambda)$')
    axes_x_lam[1].plot(sol_outer.t, sol_outer.lam[idx, :], '--')
    axes_x_lam[1].plot(sol.t, sol.lam[idx, :])

fig_states.tight_layout()

plt.show()
