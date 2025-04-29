import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('guess_nlp.data', 'rb') as f:
    guess = pickle.load(f)
with open('sol_nlp.data', 'rb') as f:
    sol = pickle.load(f)

r2d = 180./np.pi

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

fig_x, axes_x = plt.subplots(sol.x.shape[0]//2, 2)
axes_x_flat = axes_x.ravel()
fig_lam, axes_lam = plt.subplots(sol.x.shape[0]//2, 2)
axes_lam_flat = axes_lam.ravel()

x_labels = (
    r'$h$ [km]',
    r'$\phi$ [deg]',
    r'$\theta$ [deg]',
    r'$V$ [km/s]',
    r'$\gamma$ [deg]',
    r'$\psi$ [deg]',
)
lam_labels = (
    r'$\lambda_h$ [s]',
    r'$\lambda_\phi$ [m/s-rad]',
    r'$\lambda_\theta$ [m/s-rad]',
    r'$\lambda_V$ [-]',
    r'$\lambda_\gamma$ [m/s-rad]',
    r'$\lambda_\psi$ [m/s-rad]',
)
x_scale = (
    1E-3,
    r2d,
    r2d,
    1E-3,
    r2d,
    r2d
)

for idx, ax_x in enumerate(axes_x_flat):
    ax_lam = axes_lam_flat[idx]

    ax_x.set_ylabel(x_labels[idx])
    ax_x.grid()
    ax_lam.grid()
    ax_lam.set_ylabel(lam_labels[idx])
    ax_x.plot(guess.t, guess.x[idx]*x_scale[idx], '--', color=cols[1], label='Guess')
    ax_x.plot(sol.t, sol.x[idx]*x_scale[idx], color=cols[0], label='Sol')
    ax_lam.plot(sol.t, sol.lam[idx], color=cols[0], label='Sol')

ax_x = axes_x_flat[-1]
ax_lam = axes_lam_flat[-1]
ax_x.set_xlabel(t_lab)
ax_lam.set_xlabel(t_lab)

fig_x.tight_layout()
fig_lam.tight_layout()

plt.show()
