import pickle

import numpy as np
from scipy.special import ellipk, ellipkinc, ellipe, ellipeinc
from matplotlib import pyplot as plt

with open('sol_set.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]

# -------------------------------------------------------------------------------------------------------------------- #
# ANALYTICAL REFERENCES                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
# Reference control law parameterized in terms of costates ----------------------------------------------------------- #
x = sol.x[0, :]
y = sol.x[1, :]
lam_psi0 = sol.lam[2, 0]
lam_x = np.mean(sol.lam[0, :])
lam_y = np.min(sol.lam[1, :])
lam_psi_ref = lam_psi0 + lam_x * (y - y[0]) - lam_y * (x - x[0])
lam_ref = (None, None, lam_psi_ref)

# Reference solution in terms of elliptic integrals ------------------------------------------------------------------ #
# Adapted from: https://doi.org/10.2514/3.21541
sin_u = np.sin(sol.u[0, :])

k = sol.k[sol.annotations.constants.index('k')]
lam_r = (lam_x**2 + lam_y**2)**0.5  # Polar form of costates
lam_psi = sol.lam[2, :]
theta = np.arctan2(lam_y, lam_x)
c_theta = lam_x / lam_r
s_theta = lam_y / lam_r

# decompose into sat / unsat / sat portions of trajectory
use_u_max = lam_psi < -k
use_u_min = lam_psi > k
use_u_sat = np.logical_or(use_u_min, use_u_max)
use_u_unsat = np.logical_not(use_u_sat)

idx0_unsat = np.argmax(use_u_unsat)
idxf_unsat = use_u_unsat.shape[0] - np.argmax(use_u_unsat[::-1])

t_unsat = sol.t[idx0_unsat:idxf_unsat]

x_unsat = sol.x[0, idx0_unsat:idxf_unsat]
y_unsat = sol.x[1, idx0_unsat:idxf_unsat]
psi_unsat = sol.x[2, idx0_unsat:idxf_unsat]

sin_u_unsat = sin_u[idx0_unsat:idxf_unsat]
gam_unsat = 0.5 * (theta - psi_unsat)

sign_u = np.sign(sin_u_unsat)
non_switch = np.allclose(sign_u[-1], sign_u[0])


def elliptic_fun(_t0, _x0, _y0, _psi, _theta, _c_t, _m):
    _gam = 0.5 * (_theta - _psi)
    _f_elliptic = ellipkinc(_gam, _m)  # Incomplete elliptic of the first kind
    _e_elliptic = ellipeinc(_gam, _m)  # Incomplete elliptic of the second kind
    _a_int = _f_elliptic + 2. / _m * (_e_elliptic - _f_elliptic)
    _b_int = -(2. * _m * np.cos(2. * _gam) - 2. * _m + 4.) ** 0.5 / _m

    _t_elliptic = _t0 + (_f_elliptic - _f_elliptic[0])/_c_t
    _x_elliptic = _x0 + (c_theta * (_a_int - _a_int[0]) + s_theta * (_b_int - _b_int[0]))/_c_t
    _y_elliptic = _y0 + (s_theta * (_a_int - _a_int[0]) - c_theta * (_b_int - _b_int[0]))/_c_t

    _u_elliptic = -2 * _c_t * (1 - _m * np.sin(_gam) ** 2) ** 0.5
    return _t_elliptic, np.vstack((_x_elliptic, _y_elliptic, _psi)), _u_elliptic


def euler_fun(_t, _x0, _y0, _psi0, _u0):
    _u_euler = np.empty_like(_t)
    _u_euler[:] = _u0

    _x_euler = _x0 + np.cos(_psi0) * _t
    _y_euler = _y0 + np.sin(_psi0) * _t
    _psi_euler = _psi0 + (_t - _t[0]) * _u0

    return _t, np.vstack((_x_euler, _y_euler, _psi_euler)), _u_euler


c_t = -((1. + lam_r) / (2 * k)) ** 0.5
m = 2. * lam_r / (1. + lam_r)

t_elliptic = np.nan * np.empty_like(sol.t)
u_elliptic = np.nan * np.empty_like(sol.u)
x_elliptic = np.nan * np.empty_like(sol.x)

if non_switch:
    t_elliptic[idx0_unsat:idxf_unsat], x_elliptic[:, idx0_unsat:idxf_unsat], u_elliptic[:, idx0_unsat:idxf_unsat] = \
        elliptic_fun(t_unsat[0], x_unsat[0], y_unsat[0], psi_unsat, theta, sign_u[0]*c_t, m)
else:
    idx_switch_unsat = np.argmin(np.isclose(sign_u[-1], sign_u))
    idx_switch = idx0_unsat + idx_switch_unsat
    t_elliptic[idx0_unsat:idx_switch], x_elliptic[:, idx0_unsat:idx_switch], u_elliptic[:, idx0_unsat:idx_switch] \
        = elliptic_fun(t_unsat[0], x_unsat[0], y_unsat[0], psi_unsat[:idx_switch_unsat], theta, sign_u[0]*c_t, m)
    t_elliptic[idx_switch:idxf_unsat], x_elliptic[:, idx_switch:idxf_unsat], u_elliptic[:, idx_switch:idxf_unsat] \
        = elliptic_fun(t_unsat[idx_switch_unsat], x_unsat[idx_switch_unsat], y_unsat[idx_switch_unsat], psi_unsat[idx_switch_unsat:], theta, sign_u[-1]*c_t, m)

# Feather in Euler
eps_u = sol.k[sol.annotations.constants.index('eps_u')]
k_euler = (1. - lam_r)/eps_u  # TODO - finish

# Plotting ----------------------------------------------------------------------------------------------------------- #
fig_x_lam, ax_x_lam = plt.subplots(nrows=3, ncols=2, sharex=True)
labs_x = (r'$x$', r'$y$', r'$\psi$')
labs_lam = (r'$\lambda_x$', r'$\lambda_y$', r'$\lambda_\psi$')

for idx, (x, x_e, lam) in enumerate(zip(sol.x, x_elliptic, sol.lam)):
    ax_x_lam[idx][0].grid()
    ax_x_lam[idx][1].grid()
    ax_x_lam[idx][0].plot(sol.t, x, label='Numeric')
    ax_x_lam[idx][0].plot(sol.t, x_e, '--', label='Analytic')
    ax_x_lam[idx][1].plot(sol.t, lam, label='Numeric')
    if lam_ref[idx] is not None:
        ax_x_lam[idx][1].plot(sol.t, lam_ref[idx], '--', label='Analytic')
        ax_x_lam[idx][1].legend()
    ax_x_lam[idx][0].set_ylabel(labs_x[idx])
    ax_x_lam[idx][1].set_ylabel(labs_lam[idx])
    ax_x_lam[idx][0].set_xlabel(r'$t$')
    ax_x_lam[idx][1].set_xlabel(r'$t$')

fig_x_lam.tight_layout()

fig_xy, ax_xy = plt.subplots(nrows=1, ncols=1)
ax_xy.axis('equal')
ax_xy.grid()
ax_xy.set_xlabel(labs_x[0])
ax_xy.set_ylabel(labs_x[1])
ax_xy.plot(sol.x[0, :], sol.x[1, :], label='Numeric')
ax_xy.plot(x_elliptic[0, :], x_elliptic[1, :], '--', label='Analytic')
ax_xy.legend()
fig_xy.tight_layout()

u_labs = (r'$u$', r'sin($u$)')

# fig_u, ax_u = plt.subplots(nrows=2, ncols=1)
# for idx, (u, u_e) in enumerate(zip((sol.u[0, :], sin_u), (None, u_elliptic[0, :]))):
#     ax_u[idx].grid()
#     ax_u[idx].set_xlabel(r'$t$')
#     ax_u[idx].set_ylabel(u_labs[idx])
#     ax_u[idx].plot(sol.t, u, label='Numeric')
#     if u_e is not None:
#         ax_u[idx].plot(sol.t, u_e, '--', label='Analytic')
# fig_u.tight_layout()

fig_u, ax_u = plt.subplots(nrows=1, ncols=1)
ax_u.grid()
ax_u.set_xlabel(r'$t$')
ax_u.set_ylabel(r'$u$')
ax_u.plot(sol.t, sin_u, label='Numeric')
ax_u.plot(sol.t, u_elliptic[0, :], '--', label='Analytic')
fig_u.tight_layout()

plt.show()
