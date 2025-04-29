import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('guess_nlp.data', 'rb') as f:
    guess = pickle.load(f)
with open('sol_nlp.data', 'rb') as f:
    sol = pickle.load(f)

# Plot states / costates
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig_u, axes_u = plt.subplots(sol.u.shape[0])
u_labels = (
    r'$\alpha$',
    r'$\sigma$'
)
t_lab = r'$t$ [s]'

for idx, ax_u in enumerate(axes_u):
    ax_u.set_ylabel(u_labels[idx])
    ax_u.grid()
    ax_u.plot(guess.t, guess.u[idx], '--', color=cols[1], label='Guess')
    ax_u.plot(sol.t, sol.u[idx], color=cols[0], label='Sol')
axes_u[-1].set_xlabel(t_lab)

fig_u.tight_layout()

fig_x, axes_x = plt.subplots(sol.x.shape[0], 2)

x_labels = (
    r'$h$',
    r'$\phi$',
    r'$\theta$',
    r'$V$',
    r'$\gamma$',
    r'$\psi$',
)

for idx, ax_xlam in enumerate(axes_x):
    ax_x, ax_lam = ax_xlam

    ax_x.set_ylabel(x_labels[idx])
    ax_x.grid()
    ax_lam.grid()
    ax_lam.set_ylabel(r'$\lambda$ ' + x_labels[idx])
    ax_x.plot(guess.t, guess.x[idx], '--', color=cols[1], label='Guess')
    ax_x.plot(sol.t, sol.x[idx], color=cols[0], label='Sol')
    ax_lam.plot(sol.t, sol.lam[idx], color=cols[0], label='Sol')

ax_xlam = axes_x[-1]
ax_xlam[0].set_xlabel(t_lab)
ax_xlam[1].set_xlabel(t_lab)

fig_x.tight_layout()

plt.show()
