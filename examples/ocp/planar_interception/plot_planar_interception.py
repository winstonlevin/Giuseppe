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
# From: https://doi.org/10.2514/3.21541
# k = sol.k[sol.annotations.constants.index('k')]
# sin_u = np.sin(sol.u[0, :])
# lam_psi = sol.lam[2, :]
# lam_r = (lam_x**2 + lam_y**2)**0.5  # Polar form of costates
# theta = np.arctan2(-lam_y, -lam_x)
#
# # decompose into sat / unsat / sat portions of trajectory
# use_u_max = lam_psi < -k
# use_u_min = lam_psi > k
# use_u_sat = np.logical_or(use_u_min, use_u_max)
# use_u_unsat = np.logical_not(use_u_sat)
#
# idx0_unsat = np.argmax(use_u_unsat)
# idxf_unsat = use_u_unsat.shape[0] - np.argmax(use_u_unsat[::-1])
#
# # time/state/control over unsaturated (Elliptic) portion of trajectory
# t_unsat = sol.t[idx0_unsat:idxf_unsat]
# x_unsat = sol.x[0, idx0_unsat:idxf_unsat]
# y_unsat = sol.x[1, idx0_unsat:idxf_unsat]
# psi_unsat = sol.x[2, idx0_unsat:idxf_unsat]
# sin_u_unsat = sin_u[idx0_unsat:idxf_unsat]
# lam_psi_unsat = -lam_psi[idx0_unsat:idxf_unsat]
#
# # "psi" from the paper is 1/2 (theta - psi) here:
# dpsi_unsat = 0.5*(theta - psi_unsat)
#
# t_f = t_unsat[-1] - t_unsat[0]
# psi_0 = psi_unsat[0]
# psi_f = psi_unsat[-1]
# lam_psi_f = lam_psi_unsat[-1]
#
# # Handle switching vs. nonswitching separately
# non_switch = np.allclose(np.sign(sin_u_unsat[-1]), np.sign(sin_u_unsat[0]))
#
# if non_switch:
#     # Non-switching case
#     if np.isclose(lam_r, 0.):
#         # Optimal control is constant (NSc)
#         sin_u_elliptic = np.ones_like(sin_u_unsat) * lam_psi_f
#         sign_u_elliptic = np.sign(sin_u_elliptic)
#
#         c_u = sign_u_elliptic / lam_psi_f
#         t_elliptic = c_u * (psi_unsat - psi_0)
#         x_elliptic = x_unsat[0] + c_u * (np.sin(psi_unsat) - np.sin(psi_0))
#         y_elliptic = y_unsat[0] + c_u * (np.cos(psi_0) - np.cos(psi_unsat))
#         psi_elliptic = psi_unsat
#     else:
#         c_den = np.sin(dpsi_unsat[-1])**2 - lam_psi_unsat[-1]**2/(4*lam_r)
#         c_u = np.sign(sin_u_unsat[0]) / lam_r**0.5
#
#         if np.allclose(c_den, 0.):
#             # Neutral case (NS0)
#             pass  # TODO
#
#         elif c_den < 0:
#             # Negative case (NS-)
#             m = 1 / (1 - c_den)**0.5
#             alpha = np.real(np.arcsin(np.sin(dpsi_unsat)/(m*(np.sin(dpsi_unsat**2 - c_den))**0.5)))
#             f_elliptic = ellipeinc(alpha, m)  # Incomplete elliptic of the first kind
#             e_elliptic = ellipkinc(alpha, m)  # Incomplete elliptic of the second kind
#
#             t_elliptic = t_unsat[0] + c_u*m*(f_elliptic[0] - f_elliptic)
#             a = (1 - 2*c_den)*t_elliptic + c_u * (
#                     2/m*(e_elliptic - e_elliptic[0])
#                     + np.sin(2*dpsi_unsat[0])/(np.sin(dpsi_unsat[0])**2 - c_den)**0.5
#                     - np.sin(2*dpsi_unsat)/(np.sin(dpsi_unsat)**2 - c_den)**0.5
#             )
#             b = 2*c_u*((np.sin(dpsi_unsat[0])**2 - c_den)**0.5 - (np.sin(dpsi_unsat)**2 - c_den)**0.5)
#         else:
#             # Positive case (NS+)
#             m = (1 - c_den)**0.5
#             alpha = np.real(np.arcsin(np.cos(dpsi_unsat)/m, dtype=complex))
#             f_elliptic = ellipeinc(alpha, m)  # Incomplete elliptic of the first kind
#             e_elliptic = ellipkinc(alpha, m)  # Incomplete elliptic of the second kind
#
#             t_elliptic = t_unsat[0] + c_u * (f_elliptic - f_elliptic[0])
#             a = t_elliptic - 2*c_u * (e_elliptic - e_elliptic[0])
#             b = 2*c_u*((np.sin(dpsi_unsat[0])**2 - c_den)**0.5 - (np.sin(dpsi_unsat)**2 - c_den)**0.5)
#
#         x_elliptic = x_unsat[0] + np.cos(theta)*a + np.sin(theta)*b
#         y_elliptic = y_unsat[0] + np.sin(theta)*a - np.cos(theta)*b
# else:
#     # Switching case
#     pass  # TODO

# Plotting ----------------------------------------------------------------------------------------------------------- #
fig_x_lam, ax_x_lam = plt.subplots(nrows=3, ncols=2, sharex=True)
labs_x = (r'$x$', r'$y$', r'$\psi$')
labs_lam = (r'$\lambda_x$', r'$\lambda_y$', r'$\lambda_\psi$')

for idx, (x, lam) in enumerate(zip(sol.x, sol.lam)):
    ax_x_lam[idx][0].grid()
    ax_x_lam[idx][1].grid()
    ax_x_lam[idx][0].plot(sol.t, x)
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
ax_xy.plot(sol.x[0, :], sol.x[1, :])
fig_xy.tight_layout()

u_labs = (r'$u$', r'sin($u$)')
fig_u, ax_u = plt.subplots(nrows=2, ncols=1)
for idx, u in enumerate((sol.u[0, :], sin_u)):
    ax_u[idx].grid()
    ax_u[idx].set_xlabel(r'$t$')
    ax_u[idx].set_ylabel(u_labs[idx])
    ax_u[idx].plot(sol.t, u)

plt.show()
