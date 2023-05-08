import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, atm

mpl.rcParams['axes.formatter.useoffset'] = False

DATA = 2

if DATA == 0:
    with open('guess_climb.data', 'rb') as f:
        sol = pickle.load(f)
elif DATA == 1:
    with open('seed_sol_climb.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set_climb.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]


# Constants
r2d = 180 / np.pi
g0 = 32.174
s_ref = 500.
eta = 1.0

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$V$ [ft/s]', r'$\gamma$ [deg]', r'$m$ [lbm]')
ymult = np.array((1., 1., r2d, g0))
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

fig_states.suptitle(f'Cost(Wt = {sol.k[-1]}) = {sol.cost}, tf = {sol.t[-1]}, mf = {sol.x[-1, -1]}')
fig_states.tight_layout()


# PLOT U
ylabs = (r'$\alpha$ [deg]',)
ymult = np.array((r2d,))
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(sol.u)):
    axes_u.append(fig_u.add_subplot(1, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl * ymult[idx])

fig_u.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{h}$', r'$\lambda_{V}$', r'$\lambda_{\gamma}$', r'$\lambda_{m}$')
ymult = np.array((1., 1., r2d, g0))
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(sol.lam)):
    axes_costates.append(fig_costates.add_subplot(2, 2, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate * ymult[idx])

fig_costates.suptitle(f'Cost(Wt = {sol.k[-1]}) = {sol.cost}, tf = {sol.t[-1]}, mf = {sol.x[-1, -1]}')
fig_costates.tight_layout()

# PLOT FORCES
lift = np.empty(sol.t.shape)
drag = np.empty(sol.t.shape)
thrust = np.empty(sol.t.shape)

for idx, (x, u) in enumerate(zip(list(sol.x.T), list(sol.u.T))):
    _mach = x[1] / atm.speed_of_sound(x[0])
    _qdyn = 0.5 * x[1] ** 2 * atm.density(x[0])
    lift[idx] = float(_qdyn * s_ref * cl_alpha_table(_mach) * u[0])
    drag[idx] = float(_qdyn * s_ref * (cd0_table(_mach) + eta * cl_alpha_table(_mach) * u[0] ** 2))
    thrust[idx] = float(thrust_table((_mach, x[0])))

ylabs = (r'$L$ [g]', r'$D$ [g]', r'$T$ [g]')
ydata = (lift, drag, thrust)

fig_aero = plt.figure()

for idx, y in enumerate(ydata):
    ax = fig_aero.add_subplot(3, 1, idx + 1)
    ax.plot(sol.t, y / (sol.x[3] * g0))
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])

fig_aero.tight_layout()
#
# # PLOT SERIES
# fig_series = plt.figure()
#
# ax_hv = fig_series.add_subplot(1, 2, 1)
# ax_hv.grid()
# ax_hv.set_xlabel('Velocity [ft/s]')
# ax_hv.set_ylabel('Altitude [ft]')
#
# ax_ctrl = fig_series.add_subplot(1, 2, 2)
# ax_ctrl.grid()
# ax_ctrl.set_xlabel('Time [s]')
# ax_ctrl.set_ylabel(r'$\alpha$ [deg]')
#
# for solution in [sols[idx] for idx in (-100, -1)]:
#     ax_hv.plot(solution.x[1, :], solution.x[0, :])
#     ax_ctrl.plot(solution.t, solution.u[0, :] * r2d)
#
# fig_series.tight_layout()

plt.show()
