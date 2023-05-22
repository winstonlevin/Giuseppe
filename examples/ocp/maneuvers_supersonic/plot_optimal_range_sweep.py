import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, dens_table, temp_table, atm

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

gradient = mpl.colormaps['viridis'].colors

PLOT_COSTATE = True
PLOT_AUXILIARY = True
DATA = 'altitude'  # {altitude, velocity, crossrange}

with open('sol_set_range_' + DATA + '.data', 'rb') as f:
    sols = pickle.load(f)
    # sols = [sols[0], sols[25], sols[50], sols[-1]]

# Process Data
grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(sols))))


def cols_gradient(n):
    return gradient[grad_idcs[n]]


auxiliaries = [{} for _ in range(len(sols))]

for idx, sol in enumerate(sols):
    auxiliaries[idx][sol.annotations.independent] = sol.t
    for key, val in zip(sol.annotations.constants, sol.k):
        auxiliaries[idx][key] = val
    for key, val in zip(sol.annotations.parameters, sol.p):
        auxiliaries[idx][key] = val
    for key, val in zip(sol.annotations.states, list(sol.x)):
        auxiliaries[idx][key] = val
    for key, val in zip(sol.annotations.controls, list(sol.u)):
        auxiliaries[idx][key] = val

    auxiliaries[idx]['g0'] = auxiliaries[idx]['mu'] / auxiliaries[idx]['Re'] ** 2
    auxiliaries[idx]['g'] = auxiliaries[idx]['mu'] / (auxiliaries[idx]['Re'] + auxiliaries[idx]['h']) ** 2
    auxiliaries[idx]['weight'] = auxiliaries[idx]['m'] * auxiliaries[idx]['g']

    auxiliaries[idx]['mach'] = np.asarray(
        auxiliaries[idx]['v'] / (atm.specific_heat_ratio
                                 * atm.gas_constant
                                 * temp_table(auxiliaries[idx]['h'])) ** 0.5
    ).flatten()
    auxiliaries[idx]['rho'] = np.asarray(dens_table(auxiliaries[idx]['h'])).flatten()

    auxiliaries[idx]['cl_alpha'] = np.asarray(cl_alpha_table(auxiliaries[idx]['mach']), dtype=float).flatten()
    auxiliaries[idx]['cd0'] = np.asarray(cd0_table(auxiliaries[idx]['mach']), dtype=float).flatten()

    auxiliaries[idx]['ke'] = 0.5 * auxiliaries[idx]['v'] ** 2
    auxiliaries[idx]['qdyn_min_drag'] = (auxiliaries[idx]['eta'] * auxiliaries[idx]['weight'] ** 2.
                                         / (auxiliaries[idx]['s_ref'] ** 2 * auxiliaries[idx]['cd0'])) ** 0.5
    auxiliaries[idx]['ke_min_drag'] = auxiliaries[idx]['qdyn_min_drag'] / auxiliaries[idx]['rho']
    auxiliaries[idx]['pe'] = auxiliaries[idx]['g'] * auxiliaries[idx]['h']
    auxiliaries[idx]['e'] = auxiliaries[idx]['ke'] + auxiliaries[idx]['pe']

    auxiliaries[idx]['qdyn'] = auxiliaries[idx]['ke'] * auxiliaries[idx]['rho']
    _qdyn_s_ref = auxiliaries[idx]['qdyn'] * auxiliaries[idx]['s_ref']
    auxiliaries[idx]['lift'] = _qdyn_s_ref * auxiliaries[idx]['cl_alpha'] * auxiliaries[idx]['alpha']
    auxiliaries[idx]['drag'] = _qdyn_s_ref * (
            auxiliaries[idx]['cd0']
            + auxiliaries[idx]['eta'] * auxiliaries[idx]['cl_alpha'] * auxiliaries[idx]['alpha'] ** 2)
    auxiliaries[idx]['alpha_mg'] = \
        auxiliaries[idx]['weight'] / (_qdyn_s_ref * auxiliaries[idx]['cl_alpha']) \
        * np.cos(auxiliaries[idx]['gam']) / np.cos(auxiliaries[idx]['phi'])
    auxiliaries[idx]['alpha_ld'] = \
        (auxiliaries[idx]['cd0'] / (auxiliaries[idx]['eta'] * auxiliaries[idx]['cl_alpha'])) ** 0.5
    auxiliaries[idx]['ld_max'] = \
        auxiliaries[idx]['cl_alpha'] * auxiliaries[idx]['alpha_ld'] \
        / (auxiliaries[idx]['cd0']
           + auxiliaries[idx]['eta'] * auxiliaries[idx]['cl_alpha'] * auxiliaries[idx]['alpha_ld'] ** 2)
    auxiliaries[idx]['ld_mg'] = \
        auxiliaries[idx]['cl_alpha'] * auxiliaries[idx]['alpha_mg'] \
        / (auxiliaries[idx]['cd0']
           + auxiliaries[idx]['eta'] * auxiliaries[idx]['cl_alpha'] * auxiliaries[idx]['alpha_mg'] ** 2)

    auxiliaries[idx]['t_ld_dist_intersect'] = (np.empty((0,)), np.nan, np.inf)

    # Obtain the closest point to ld = max(ld) = ld : n = 1
    for t, ld, ld_max, ld_mg in zip(auxiliaries[idx]['t'],
                                    auxiliaries[idx]['lift'] / auxiliaries[idx]['drag'],
                                    auxiliaries[idx]['ld_max'],
                                    auxiliaries[idx]['ld_mg']):
        dist = ((ld - ld_max) ** 2 + (ld - ld_mg) ** 2 + (ld_max - ld_mg) ** 2)**0.5
        if dist < auxiliaries[idx]['t_ld_dist_intersect'][2]:
            auxiliaries[idx]['t_ld_dist_intersect'] = (t, ld, dist)


r2d = 180 / np.pi
g0 = auxiliaries[0]['g0']

t_label = 'Time [s]'

# INITIALIZE PLOTS ----------------------------------------------------

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$x_N$ [ft]', r'$x_E$ [ft]', r'$V$ [ft/s]', r'$\gamma$ [deg]', r'$\psi$ [deg]', r'$m$ [lbm]')
ymult = np.array((1., 1., 1., 1., r2d, r2d, g0))
fig_states = plt.figure()
axes_states = []

for idx, lab in enumerate(ylabs):
    axes_states.append(fig_states.add_subplot(2, 4, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    for jdx, sol in enumerate(sols):
        ax.plot(sol.t, sol.x[idx, :] * ymult[idx], color=cols_gradient(jdx))

fig_states.tight_layout()

# PLOT U
ylabs = (r'$\alpha$ [deg]', r'$\phi$ [deg]')
ymult = np.array((r2d, r2d))
fig_u = plt.figure()
axes_u = []

for idx, lab in enumerate(ylabs):
    axes_u.append(fig_u.add_subplot(1, 2, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    for jdx, (sol, aux) in enumerate(zip(sols, auxiliaries)):
        ax.plot(sol.t, sol.u[idx, :] * ymult[idx], color=cols_gradient(jdx))

        if idx == 0:
            if jdx == 0:
                ax.plot(sol.t, aux['alpha_ld'] * ymult[idx], ':', label='Max L/D', color=cols_gradient(jdx))
                ax.plot(sol.t, aux['alpha_mg'] * ymult[idx], '--', label=r'$n = 1$', color=cols_gradient(jdx))
            else:
                ax.plot(sol.t, aux['alpha_ld'] * ymult[idx], ':', color=cols_gradient(jdx))
                ax.plot(sol.t, aux['alpha_mg'] * ymult[idx], '--', color=cols_gradient(jdx))

axes_u[0].legend()

fig_u.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = (
        r'$\lambda_{h}$', r'$\lambda_{x_N}$', r'$\lambda_{x_E}$',
        r'$\lambda_{V}$', r'$\lambda_{\gamma}$', r'$\lambda_{\psi}$',
        r'$\lambda_{m}$'
    )
    ymult = np.array((1., 1., 1., 1., 1., 1., 1.))
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sols[-1].lam)):
        axes_costates.append(fig_costates.add_subplot(2, 4, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        for jdx, sol in enumerate(sols):
            ax.plot(sol.t, sol.lam[idx, :] * ymult[idx], color=cols_gradient(jdx))

    fig_costates.tight_layout()

if PLOT_AUXILIARY:
    # PLOT FORCES
    ylabs = (r'$L$ [g]', r'$D$ [g]', r'$L/D$')

    fig_aero = plt.figure()

    ax_lift = fig_aero.add_subplot(311)
    ax_drag = fig_aero.add_subplot(312)
    ax_ld = fig_aero.add_subplot(313)

    for ax, lab in zip((ax_lift, ax_drag, ax_ld), ylabs):
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(lab)

    fig_aero.tight_layout()

    # PLOT ENERGY STATE
    xlabs = (t_label, r'$M$', t_label, r'$x_E$')
    ylabs = (r'Energy [ft$^2$/s$^2$]', 'h [ft]', r'$q_{\infty}$ [psf]', r'$x_N$')
    fig_energy_state = plt.figure()

    ax_e = fig_energy_state.add_subplot(221)
    ax_hv = fig_energy_state.add_subplot(222)
    ax_qdyn = fig_energy_state.add_subplot(223)
    ax_ne = fig_energy_state.add_subplot(224)

    for ax, x_lab, y_lab in zip((ax_e, ax_hv, ax_qdyn, ax_ne), xlabs, ylabs):
        ax.grid()
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)

    for idx, (sol, aux) in enumerate(zip(sols, auxiliaries)):
        ax_lift.plot(sol.t, aux['lift'] / aux['weight'], color=cols_gradient(idx))
        ax_drag.plot(sol.t, aux['drag'] / aux['weight'], color=cols_gradient(idx))
        ax_ld.plot(sol.t, aux['lift'] / aux['drag'], color=cols_gradient(idx))
        ax_ld.plot(aux['t_ld_dist_intersect'][0], aux['t_ld_dist_intersect'][1], '*', color=cols_gradient(idx))

        ax_e.plot(sol.t, aux['e'], color=cols_gradient(idx))
        ax_hv.plot(aux['mach'], aux['h'], color=cols_gradient(idx))
        ax_qdyn.plot(sol.t, aux['qdyn'], color=cols_gradient(idx))
        ax_ne.plot(aux['xe'], aux['xn'], color=cols_gradient(idx))

        if idx == 0:
            ax_ld.plot(sol.t, aux['ld_mg'], '--', label='n = 1', color=cols_gradient(idx))
            ax_ld.plot(sol.t, aux['ld_max'], ':', label='Max L/D', color=cols_gradient(idx))

            # ax_qdyn.plot(sol.t, aux['qdyn_min_drag'], '--', label='Min Drag', color=cols_gradient(idx))
        else:
            ax_ld.plot(sol.t, aux['ld_mg'], '--', color=cols_gradient(idx))
            ax_ld.plot(sol.t, aux['ld_max'], ':', color=cols_gradient(idx))

            # ax_qdyn.plot(sol.t, aux['qdyn_min_drag'], '--', color=cols_gradient(idx))

    ax_ld.legend()
    # ax_qdyn.legend()

    fig_aero.tight_layout()
    fig_energy_state.tight_layout()

plt.show()
