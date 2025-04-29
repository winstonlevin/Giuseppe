import pickle
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

import giuseppe

with open('guess_nlp.data', 'rb') as f:
    guess = pickle.load(f)
with open('sol_nlp.data', 'rb') as f:
    sol = pickle.load(f)


def interpolate_signal(_t, _y, _t_interp):
    # Get tau vals
    t0, tf = _t[((0, -1),)]
    _tb, _tr = 0.5*(tf+t0), 0.5*(tf-t0)

    is_col = np.ones(shape=_t.shape, dtype=bool)
    is_col[np.isnan(_y[0, :])] = False  # Remove non-collocated point
    is_col[-1] = False  # Terminal point is extrapolated
    col_pts = (_t[is_col] - _tb) / _tr
    col_interp = (_t_interp - _tb) / _tr

    interp_mat, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
        col_pts, col_interp, compute_interp_matrix=True, compute_diff_matrix=False
    )
    return _y[:, is_col] @ interp_mat.T


def interpolate_solution(_sol, _t_interp):
    return giuseppe.data_classes.Solution(
        t=_t_interp,
        nu0=_sol.nu0,
        nuf=_sol.nuf,
        x=interpolate_signal(_sol.t, _sol.x, _t_interp),
        u=interpolate_signal(_sol.t, _sol.u, _t_interp),
        lam=interpolate_signal(_sol.t, _sol.lam, _t_interp)
    )


sol_interp = interpolate_solution(sol, np.linspace(sol.t[0], sol.t[-1], 1_000))

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
    ax_u.plot(guess.t, guess.u[idx], '*', color=cols[1], label='Guess')
    ax_u.plot(sol_interp.t, sol_interp.u[idx], '--', color=cols[0], label='Interp')
    ax_u.plot(sol.t, sol.u[idx], '*', color=cols[0], label='Sol')
axes_u[-1].set_xlabel(t_lab)

fig_u.tight_layout()

fig_lam, axes_lam = plt.subplots(sol.x.shape[0]//2, 2)
axes_lam_flat = axes_lam.ravel()
fig_x, axes_x = plt.subplots(sol.x.shape[0]//2, 2)
axes_x_flat = axes_x.ravel()

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
    ax_x.plot(guess.t, guess.x[idx]*x_scale[idx], '*', color=cols[1], label='Guess')
    ax_x.plot(sol_interp.t, sol_interp.x[idx]*x_scale[idx], '--', color=cols[0], label='Interp')
    ax_x.plot(sol.t, sol.x[idx]*x_scale[idx], '*', color=cols[0], label='Sol')
    ax_lam.plot(sol_interp.t, sol_interp.lam[idx], '--', color=cols[0], label='Interp')
    ax_lam.plot(sol.t, sol.lam[idx], '*', color=cols[0], label='Sol')

ax_x = axes_x_flat[-1]
ax_lam = axes_lam_flat[-1]
ax_x.set_xlabel(t_lab)
ax_lam.set_xlabel(t_lab)

fig_x.tight_layout()
fig_lam.tight_layout()

plt.show()
