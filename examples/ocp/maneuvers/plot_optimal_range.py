import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, dens_table, temp_table, atm

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOT_COSTATE = True
PLOT_AUXILIARY = True
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
lam_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val

for key, x_val, lam_val in zip(sol.annotations.states, list(sol.x), list(sol.lam)):
    x_dict[key] = x_val
    lam_dict[key] = lam_val

for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val

# Process Data
r2d = 180 / np.pi
g0 = k_dict['mu'] / k_dict['Re'] ** 2
s_ref = k_dict['s_ref']
eta = k_dict['eta']

g = k_dict['mu'] / (k_dict['Re'] + x_dict['h']) ** 2
v = (2 * (x_dict['e'] - g * x_dict['h'])) ** 0.5
weight = x_dict['m'] * g

mach = np.asarray(v / (atm.specific_heat_ratio * atm.gas_constant * temp_table(x_dict['h'])) ** 0.5).flatten()
rho = np.asarray(dens_table(x_dict['h'])).flatten()

cl_alpha = np.asarray(cl_alpha_table(mach), dtype=float).flatten()
cd0 = np.asarray(cd0_table(mach), dtype=float).flatten()

qdyn = 0.5 * rho * v ** 2
lift = qdyn * s_ref * cl_alpha * u_dict['alpha']
drag = qdyn * s_ref * (cd0 + eta * cl_alpha * u_dict['alpha'] ** 2)
alpha_mg = weight / (qdyn * s_ref * cl_alpha) * np.cos(x_dict['gam']) / np.cos(u_dict['phi'])
alpha_ld = (cd0 / (k_dict['eta'] * cl_alpha)) ** 0.5
alpha_opt = lam_dict['gam'] / (2 * k_dict['eta'] * v**2 * lam_dict['e'])
ld_mg = cl_alpha * alpha_mg / (cd0 + eta * cl_alpha * alpha_mg ** 2)
ld_max = cl_alpha * alpha_ld / (cd0 + eta * cl_alpha * alpha_ld ** 2)

# Gam to minimize drag : L = W cos(gam)
ke = 0.5 * v**2
qdyn_min_drag = (eta * weight**2. / (s_ref**2 * cd0)) ** 0.5
ke_min_drag = qdyn_min_drag / rho
pe = g * x_dict['h']

t_label = 'Time [s]'
title_str = f'Cost(Ang. ={k_dict["terminal_angle"]}) = {sol.cost * k_dict["x_ref"]},' \
            f'\nDownrange = {x_dict["xn"][-1]}, Crossrange = {x_dict["xe"][-1]}'

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$x_N$ [ft]', r'$x_E$ [ft]',
         r'$E$ [ft$^2$/s$^2$]', r'$\gamma$ [deg]', r'$\psi$ [deg]', r'$m$ [lbm]', r'$V$ [ft/s]')
ymult = np.array((1., 1., 1., 1., r2d, r2d, g0, 1.))
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(np.vstack((sol.x, v)))):
    axes_states.append(fig_states.add_subplot(2, 4, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

fig_states.suptitle(title_str)
fig_states.tight_layout()

# PLOT U
ylabs = (r'$\alpha$ [deg]', r'$\phi$ [deg]')
ymult = np.array((r2d, r2d))
ydataref1 = (alpha_mg, np.nan * alpha_mg)
ydataref2 = (alpha_ld, np.nan * alpha_ld)
ydataref3 = (alpha_opt, np.nan * alpha_opt)
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(sol.u)):
    axes_u.append(fig_u.add_subplot(1, 2, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl * ymult[idx])
    ax.plot(sol.t, ydataref1[idx] * ymult[idx], 'k--', label='n = 1')
    ax.plot(sol.t, ydataref2[idx] * ymult[idx], 'k:', label='max L/D')
    # ax.plot(sol.t, ydataref3[idx] * ymult[idx], 'k*', label='Analytic')

axes_u[0].legend()

fig_u.suptitle(title_str)
fig_u.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = (
        r'$\lambda_{h}$', r'$\lambda_{x_N}$', r'$\lambda_{x_E}$',
        r'$\lambda_{V}$', r'$\lambda_{\gamma}$', r'$\lambda_{\psi}$',
        r'$\lambda_{m}$'
    )
    ymult = np.array((k_dict['h_ref'], k_dict['x_ref'], k_dict['x_ref'],
                      k_dict['e_ref'], k_dict['gam_ref'], k_dict['psi_ref'], k_dict['m_ref']))
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sol.lam)):
        axes_costates.append(fig_costates.add_subplot(2, 4, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t, costate * ymult[idx])

    fig_costates.suptitle(title_str)
    fig_costates.tight_layout()

if PLOT_AUXILIARY:
    # PLOT FORCES
    ylabs = (r'$L$ [g]', r'$D$ [g]', r'$L/D$')
    ydata = (lift / weight, drag / weight, lift / drag)

    fig_aero = plt.figure()
    axes_aero = []

    for idx, y in enumerate(ydata):
        axes_aero.append(fig_aero.add_subplot(3, 1, idx + 1))
        ax = axes_aero[-1]
        ax.plot(sol.t, y)
        # ax.plot(sol.t, ydataref[idx], 'k--', label='Max L/D')
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])

    axes_aero[-1].plot(sol.t, ld_mg, 'k--', label='n = 1', zorder=0)
    axes_aero[-1].plot(sol.t, ld_max, 'k:', label='L/D max', zorder=0)
    axes_aero[-1].legend()

    fig_aero.suptitle(title_str)
    fig_aero.tight_layout()

    # PLOT ENERGY STATE
    fig_energy_state = plt.figure()

    ax_e = fig_energy_state.add_subplot(221)
    ax_e.plot(sol.t, ke, color=col[0], label='KE')
    ax_e.plot(sol.t, pe, color=col[1], label='PE')
    # ax_e.plot(sol.t, ke_min_drag, '--', color=col[0], label=r'KE($D_{min}$)')
    ax_e.plot(sol.t, x_dict['e'], 'k', label='E = KE + PE')
    ax_e.grid()
    ax_e.set_xlabel(t_label)
    ax_e.set_ylabel(r'Energy [ft$^2$/s$^2$]')
    ax_e.legend()

    ax_hv = fig_energy_state.add_subplot(222)
    ax_hv.plot(mach, x_dict['h'])
    ax_hv.grid()
    ax_hv.set_xlabel('M')
    ax_hv.set_ylabel('h [ft]')

    # qdyn_idces = np.where(np.logical_and(sol.t > 200., sol.t < 800.))
    ax_qdyn = fig_energy_state.add_subplot(223)
    # ax_qdyn.plot(x_dict['gam'][qdyn_idces] * r2d, qdyn[qdyn_idces])
    # ax_qdyn.plot(x_dict['gam'] * r2d, qdyn)
    ax_qdyn.plot(sol.t, qdyn)
    # ax_qdyn.plot(sol.t, qdyn_min_drag, 'k--', label=r'$q_{\infty}$ : $D$ = $D_{min}$')
    ax_qdyn.grid()
    ax_qdyn.set_xlabel(t_label)
    # ax_qdyn.set_xlabel(r'$\gamma$ [deg]')
    ax_qdyn.set_ylabel(r'$q_{\infty}$ [psf]')

    ax_ne = fig_energy_state.add_subplot(224)
    ax_ne.plot(x_dict['xe'], x_dict['xn'])
    ax_ne.grid()
    ax_ne.set_xlabel(r'$x_E$')
    ax_ne.set_ylabel(r'$x_N$')

    fig_energy_state.suptitle(title_str)
    fig_energy_state.tight_layout()

plt.show()
