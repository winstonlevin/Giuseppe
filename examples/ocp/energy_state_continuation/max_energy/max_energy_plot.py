import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('sol_nlp.data', 'rb') as f:
    sol_dict = pickle.load(f)


# Generate sequence
tau_vals = np.linspace(-1, 1, 1000)

# Guess
t_guess = sol_dict['tf_guess'] * (tau_vals + 1)/2
x_guess = sol_dict['xb'][:, None] + sol_dict['xr'][:, None] * np.vstack([
    _x_poly(tau_vals) for _x_poly in sol_dict['x_guess']
])
u_guess = sol_dict['ub'][:, None] + sol_dict['ur'][:, None] * np.vstack([
    _u_poly(tau_vals) for _u_poly in sol_dict['u_guess']
])

t = sol_dict['tf'] * (tau_vals + 1)/2
x = sol_dict['xb'][:, None] + sol_dict['xr'][:, None] * np.vstack([_x_poly(tau_vals) for _x_poly in sol_dict['x']])
u = sol_dict['ub'][:, None] + sol_dict['ur'][:, None] * np.vstack([_u_poly(tau_vals) for _u_poly in sol_dict['u']])
lam = sol_dict['lamb'][:, None] + sol_dict['lamr'][:, None] * np.vstack([
    _lam_poly(tau_vals) for _lam_poly in sol_dict['lam']
])

# Plot states / costates
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig_u, axes_u = plt.subplots(u.shape[0])
u_labels = (
    r'$\alpha$',
    r'$\sigma$'
)

for idx, ax_u in enumerate(axes_u):
    ax_u.set_ylabel(u_labels[idx])
    ax_u.plot(t_guess, u_guess[idx], '--', color=cols[1], label='Guess')
    ax_u.plot(t, u[idx], color=cols[0], label='Sol')

fig_u.tight_layout()

fig_x, axes_x = plt.subplots(x.shape[0], 2)

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
    ax_lam.set_ylabel(r'$\lambda$ ' + x_labels[idx])
    ax_x.plot(t_guess, x_guess[idx], '--', color=cols[1], label='Guess')
    ax_x.plot(t, x[idx], color=cols[0], label='Sol')
    ax_lam.plot(t, lam[idx], color=cols[0], label='Sol')

fig_x.tight_layout()

plt.show()
