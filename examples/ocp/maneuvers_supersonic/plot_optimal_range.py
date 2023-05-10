import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, atm

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

DATA = 2

if DATA == 0:
    with open('guess_range.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan
elif DATA == 1:
    with open('seed_sol_range.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set_range.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

# Create Dicts
k_dict = {}
x_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val

for key, val in zip(sol.annotations.states, list(sol.x)):
    x_dict[key] = val

for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val

# Process Data
r2d = 180 / np.pi
g0 = k_dict['mu'] / k_dict['Re'] ** 2
s_ref = k_dict['s_ref']
eta = k_dict['eta']

g = k_dict['mu'] / (k_dict['Re'] + x_dict['h']) ** 2
weight = x_dict['m'] * g

mach = np.empty(sol.t.shape)
rho = np.empty(sol.t.shape)
cl_alpha = np.empty(sol.t.shape)
cd0 = np.empty(sol.t.shape)

for idx, (h, v, alpha) in enumerate(zip(x_dict['h'], x_dict['v'], u_dict['alpha'])):
    mach[idx] = v / atm.speed_of_sound(h)
    rho[idx] = atm.density(h)
    cl_alpha[idx] = float(cl_alpha_table(mach[idx]))
    cd0[idx] = float(cd0_table(mach[idx]))

qdyn = 0.5 * rho * x_dict['v'] ** 2
lift = qdyn * s_ref * cl_alpha * u_dict['alpha']
drag = qdyn * s_ref * (cd0 + eta * cl_alpha * u_dict['alpha'] ** 2)
alpha_mg = weight / (qdyn * s_ref * cl_alpha)

ke = 0.5 * x_dict['v']**2
qdyn_min_drag = (eta * weight**2. / (s_ref**2 * cd0)) ** 0.5
ke_min_drag = qdyn_min_drag / rho
pe = g * x_dict['h']
e = ke + pe

t_label = 'Time [s]'
title_str = f'Cost = {sol.cost * k_dict["xn_ref"]}, Downrange = {x_dict["xn"][-1]}'

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$x_N$ [ft]', r'$V$ [ft/s]', r'$\gamma$ [deg]', r'$m$ [lbm]')
ymult = np.array((1., 1., 1., r2d, g0))
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 3, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

fig_states.suptitle(title_str)
fig_states.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{h}$', r'$\lambda_{x_N}$', r'$\lambda_{V}$', r'$\lambda_{\gamma}$', r'$\lambda_{m}$')
ymult = np.array((1., 1., 1., 1., 1.))
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(sol.lam)):
    axes_costates.append(fig_costates.add_subplot(2, 3, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate * ymult[idx])

fig_costates.suptitle(title_str)
fig_costates.tight_layout()

# PLOT FORCES
ylabs = (r'$L$ [g]', r'$D$ [g]', r'$L/D$')
ydata = (lift / weight, drag / weight, lift / drag)

fig_aero = plt.figure()

for idx, y in enumerate(ydata):
    ax = fig_aero.add_subplot(3, 1, idx + 1)
    ax.plot(sol.t, y)
    # ax.plot(sol.t, ydataref[idx], 'k--', label='Max L/D')
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

fig_aero.suptitle(title_str)
fig_aero.tight_layout()

# PLOT U
ylabs = (r'$\alpha$ [deg]',)
ymult = np.array((r2d,))
ydataref = (alpha_mg,)
# ydataref - (alpha_ld_max,)
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(sol.u)):
    axes_u.append(fig_u.add_subplot(1, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl * ymult[idx])
    ax.plot(sol.t, ydataref[idx] * ymult[idx], 'k--', label=r'$\alpha$ : L = mg')
    ax.legend()

fig_u.suptitle(title_str)
fig_u.tight_layout()

# PLOT ENERGY STATE
fig_energy_state = plt.figure()

ax_e = fig_energy_state.add_subplot(221)
ax_e.plot(sol.t, ke, color=col[0])
ax_e.plot(sol.t, pe, color=col[1])
ax_e.plot(sol.t, ke_min_drag, '--', color=col[0])
ax_e.plot(sol.t, e, 'k')
ax_e.grid()
ax_e.set_xlabel(t_label)
ax_e.set_ylabel(r'Energy [ft$^2$/s$^2$]')

ax_hv = fig_energy_state.add_subplot(222)
ax_hv.plot(mach, x_dict['h'])
ax_hv.grid()
ax_hv.set_xlabel('M')
ax_hv.set_ylabel('h [ft]')

ax_qdyn = fig_energy_state.add_subplot(223)
ax_qdyn.plot(sol.t, qdyn)
ax_qdyn.plot(sol.t, qdyn_min_drag, 'k--', label=r'$q_{\infty}$ : $D$ = $D_{min}$')
ax_qdyn.grid()
ax_qdyn.set_xlabel(t_label)
ax_qdyn.set_ylabel(r'$q_{\infty}$ [psf]')

fig_energy_state.suptitle(title_str)
fig_energy_state.tight_layout()

plt.show()
