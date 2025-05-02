import pickle
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import matplotlib as mpl

import giuseppe


mpl.rcParams['axes.formatter.useoffset'] = False

with open('guess_nlp_legendre.data', 'rb') as f:
    guess = pickle.load(f)
with open('sol_nlp.data', 'rb') as f:
    sol_nlp = pickle.load(f)
with open('sol_nlp_legendre.data', 'rb') as f:
    sol_legendre = pickle.load(f)
with open('sol_nonsingular.data', 'rb') as f:
    sol_nonsingular = pickle.load(f)


def interpolate_signal(_t, _y, _t_interp):
    # Get tau vals
    t0, tf = _t[((0, -1),)]
    _tb, _tr = 0.5*(tf+t0), 0.5*(tf-t0)

    col_pts = (_t - _tb) / _tr
    col_interp = (_t_interp - _tb) / _tr

    interp_mat, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
        col_pts, col_interp, compute_interp_matrix=True, compute_diff_matrix=False
    )
    return _y @ interp_mat.T


def interpolate_solution(_sol, n_vals: int = 1000):
    _t_interp = np.linspace(_sol.t[0], _sol.t[-1], n_vals)
    return giuseppe.data_classes.Solution(
        t=_t_interp,
        nu0=_sol.nu0,
        nuf=_sol.nuf,
        x=interpolate_signal(_sol.t[_sol.k > 0], _sol.x[:, _sol.k > 0], _t_interp),
        u=interpolate_signal(_sol.t[_sol.k > 1], _sol.u[:, _sol.k > 1], _t_interp),
        lam=interpolate_signal(_sol.t[_sol.k > 1], _sol.lam[:, _sol.k > 1], _t_interp)
    )


sol_nlp_interp = interpolate_solution(sol_nlp)

r2d = 180./np.pi

# Plot states / costates
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

guess_plot_kwargs = {'marker': '*', 'linestyle': '', 'color': cols[1], 'label': 'Guess'}
nlp_plot_kwargs = {'marker': '*', 'linestyle': '', 'color': cols[0], 'label': 'NLP'}
nlp_interp_plot_kwargs = {'linestyle': '--', 'color': cols[0]}
legendre_plot_kwargs = {'marker': 'o', 'linestyle': '', 'color': cols[2], 'label': 'NLP Leg.', 'markersize': 6}
indirect_plot_kwargs = {'marker': 'o', 'linestyle': '', 'color': cols[3], 'label': 'Indirect', 'markersize': 6}

fig_u, axes_u = plt.subplots(sol_nlp.u.shape[0])
u_labels = (
    r'$\alpha$',
    r'$\sigma$'
)
t_lab = r'$t$ [s]'

for idx, ax_u in enumerate(axes_u):
    ax_u.set_ylabel(u_labels[idx])
    ax_u.grid()
    ax_u.plot(guess.t, guess.u[idx], **guess_plot_kwargs)
    ax_u.plot(sol_nlp_interp.t, sol_nlp_interp.u[idx], **nlp_interp_plot_kwargs)
    ax_u.plot(sol_nlp.t, sol_nlp.u[idx], **nlp_plot_kwargs)
    ax_u.plot(sol_legendre.t, sol_legendre.u[idx], **legendre_plot_kwargs)
    ax_u.plot(sol_nonsingular.t, sol_nonsingular.u[idx], **indirect_plot_kwargs)
axes_u[-1].set_xlabel(t_lab)

fig_u.tight_layout()

fig_lam, axes_lam = plt.subplots(sol_nlp.x.shape[0] // 2, 2)
axes_lam_flat = axes_lam.ravel()
fig_x, axes_x = plt.subplots(sol_nlp.x.shape[0] // 2, 2)
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

    ax_x.plot(guess.t, guess.x[idx]*x_scale[idx], **guess_plot_kwargs)
    ax_x.plot(sol_nlp_interp.t, sol_nlp_interp.x[idx] * x_scale[idx], **nlp_interp_plot_kwargs)
    ax_x.plot(sol_nlp.t, sol_nlp.x[idx] * x_scale[idx], **nlp_plot_kwargs)
    ax_x.plot(sol_legendre.t, sol_legendre.x[idx] * x_scale[idx], **legendre_plot_kwargs)
    ax_x.plot(sol_nonsingular.t, sol_nonsingular.x[idx] * x_scale[idx], **indirect_plot_kwargs)

    ax_lam.plot(sol_nlp_interp.t, sol_nlp_interp.lam[idx], **nlp_interp_plot_kwargs)
    ax_lam.plot(sol_nlp.t, sol_nlp.lam[idx], **nlp_plot_kwargs)
    ax_lam.plot(sol_legendre.t, sol_legendre.lam[idx], **legendre_plot_kwargs)
    ax_lam.plot(sol_nonsingular.t, sol_nonsingular.lam[idx], **indirect_plot_kwargs)

ax_x = axes_x_flat[-1]
ax_lam = axes_lam_flat[-1]
ax_x.set_xlabel(t_lab)
ax_lam.set_xlabel(t_lab)

fig_x.tight_layout()
fig_lam.tight_layout()


# Fig Legendre Polynomial Coefficients to solutions
def fit_legendre(_t, _y):
    # Non-dimensionalize time
    _t0, _tf = _t[0], _t[-1]
    _tb, _tr = 0.5*(_tf + _t0), 0.5*(_tf - _t0)
    _tau = (_t - _tb) / _tr

    # Generate legendre basis
    _nb = _tau.size
    _bases = [np.polynomial.Legendre.basis(_deg) for _deg in range(_nb)]

    # Generate coefficient matrix
    _design_matrix = np.vstack([_phi(_tau) for _phi in _bases])

    # Non-dimensionalize state
    _yl, _yu = _y.min(initial=np.inf, axis=-1), _y.max(initial=-np.inf, axis=-1)
    _yb, _yr = 0.5*(_yu + _yl), 0.5*(_yu - _yl)
    _y_nd = (_y - _yb[:, None]) / _yr[:, None]

    # Generate fit
    return np.linalg.solve(_design_matrix, _y_nd.T).T


X_leg = fit_legendre(sol_nlp.t[sol_nlp.k > 0], sol_nlp.x[:, sol_nlp.k > 0])
U_leg = fit_legendre(sol_nlp.t[sol_nlp.k > 1], sol_nlp.u[:, sol_nlp.k > 1])
Lam_leg = fit_legendre(sol_nlp.t[sol_nlp.k > 1], sol_nlp.lam[:, sol_nlp.k > 1])

print('Largest Legendre coeff magnitude (for n.d. state/control/costate):')
print(np.max((np.abs(X_leg).max(), np.abs(U_leg).max(), np.abs(Lam_leg).max())))

plt.show()
