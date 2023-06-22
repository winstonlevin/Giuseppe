import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, dens_fun, sped_fun, lut_data
from glide_slope import get_glide_slope

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

gradient = mpl.colormaps['viridis'].colors

PLOT_COSTATE = True
PLOT_AUXILIARY = True
PLOT_REFERENCE = True
DATA = 'velocity'  # {altitude, velocity, crossrange}

with open('sol_set_range_' + DATA + '.data', 'rb') as f:
    sols = pickle.load(f)
    sols = [sols[0], sols[25], sols[50], sols[-1]]
    # sols = [sols[0]]

# Process Data
if len(sols) == 1:
    grad_idcs = np.array((0,), dtype=np.int32)
else:
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
    for key, val in zip(sol.annotations.states, list(sol.lam)):
        auxiliaries[idx]['lam_' + key] = val
    for key, val in zip(sol.annotations.controls, list(sol.u)):
        auxiliaries[idx][key] = val

    auxiliaries[idx]['g0'] = auxiliaries[idx]['mu'] / auxiliaries[idx]['Re'] ** 2
    auxiliaries[idx]['g'] = auxiliaries[idx]['mu'] / (auxiliaries[idx]['Re'] + auxiliaries[idx]['h']) ** 2
    auxiliaries[idx]['weight'] = auxiliaries[idx]['m'] * auxiliaries[idx]['g']

    auxiliaries[idx]['mach'] = np.asarray(
        auxiliaries[idx]['v'] / np.asarray(sped_fun(auxiliaries[idx]['h'])).flatten()
    ).flatten()
    auxiliaries[idx]['rho'] = np.asarray(dens_fun(auxiliaries[idx]['h'])).flatten()

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

    auxiliaries[idx]['AD0'] = (auxiliaries[idx]['qdyn'] * auxiliaries[idx]['s_ref']
                               * auxiliaries[idx]['cd0']) / auxiliaries[idx]['weight']
    auxiliaries[idx]['ADL'] = auxiliaries[idx]['eta'] * auxiliaries[idx]['weight'] / (
        auxiliaries[idx]['qdyn'] * auxiliaries[idx]['s_ref'] * auxiliaries[idx]['cl_alpha']
    )

    auxiliaries[idx]['load_asymptotic'] = np.empty(sol.t.shape)

    s_ref = auxiliaries[idx]['s_ref']
    eta = auxiliaries[idx]['eta']

    if idx == 0:
        aux_ref = auxiliaries[0]
        h_interp, v_interp, gam_interp, drag_interp = get_glide_slope(
            aux_ref['mu'], aux_ref['Re'], aux_ref['m'][0], aux_ref['s_ref'], aux_ref['eta'],
            _e_vals=aux_ref['e'], _h_min=0., _h_max=np.max(lut_data['h']), _mach_max=np.max(lut_data['M'])
        )

    auxiliaries[idx]['v_glide'] = v_interp(auxiliaries[idx]['e'])
    auxiliaries[idx]['h_glide'] = (auxiliaries[idx]['e'] - 0.5 * auxiliaries[idx]['v_glide']**2) / auxiliaries[idx]['g']
    auxiliaries[idx]['rho_glide'] = np.asarray(dens_fun(auxiliaries[idx]['h_glide'])).flatten()
    auxiliaries[idx]['qdyn_glide'] = 0.5 * auxiliaries[idx]['rho_glide'] * auxiliaries[idx]['v_glide'] ** 2
    auxiliaries[idx]['mach_glide'] = auxiliaries[idx]['v_glide'] \
                                     / np.asarray(sped_fun(auxiliaries[idx]['h_glide'])).flatten()
    auxiliaries[idx]['cd0_glide'] = np.asarray(cd0_table(auxiliaries[idx]['mach_glide']), dtype=float).flatten()
    auxiliaries[idx]['cl_alpha_glide'] = np.asarray(cl_alpha_table(auxiliaries[idx]['mach_glide']), dtype=float).flatten()
    auxiliaries[idx]['AD0_glide'] = (auxiliaries[idx]['qdyn_glide'] * auxiliaries[idx]['s_ref']
                                     * auxiliaries[idx]['cd0_glide']) / auxiliaries[idx]['weight']
    auxiliaries[idx]['ADL_glide'] = auxiliaries[idx]['eta'] * auxiliaries[idx]['weight'] / (
        auxiliaries[idx]['qdyn_glide'] * auxiliaries[idx]['s_ref'] * auxiliaries[idx]['cl_alpha_glide']
    )

    auxiliaries[idx]['dh'] = auxiliaries[idx]['h'] - auxiliaries[idx]['h_glide']
    auxiliaries[idx]['dv'] = auxiliaries[idx]['v'] - auxiliaries[idx]['v_glide']
    auxiliaries[idx]['dgam'] = auxiliaries[idx]['gam'] - gam_interp(auxiliaries[idx]['e'])
    auxiliaries[idx]['dalp'] = auxiliaries[idx]['alpha'] - auxiliaries[idx]['alpha_mg']
    auxiliaries[idx]['ddrag'] = (auxiliaries[idx]['drag'] - drag_interp(auxiliaries[idx]['e'])) / auxiliaries[idx]['weight']

    _cgam = np.cos(auxiliaries[idx]['gam'])
    _b = (
             auxiliaries[idx]['AD0'] + auxiliaries[idx]['ADL'] * _cgam**2
             - _cgam * (auxiliaries[idx]['AD0_glide'] + auxiliaries[idx]['ADL_glide'])
         ) / auxiliaries[idx]['ADL']

    auxiliaries[idx]['load_asymptotic'] = _cgam + np.sign(auxiliaries[idx]['h_glide'] - auxiliaries[idx]['h']) * (
        np.maximum(_b, 0)
    ) ** 0.5

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

if PLOT_REFERENCE:
    axes_states[4].plot(aux_ref['t'], gam_interp(aux_ref['e']) * ymult[4], 'k--', label='Glide Slope')
    axes_states[4].legend()

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

        if idx == 0 and PLOT_REFERENCE:
            if jdx == 0:
                ax.plot(sol.t, aux['alpha_ld'] * ymult[idx], ':', label='Max L/D', color=cols_gradient(jdx))
                ax.plot(sol.t, aux['alpha_mg'] * ymult[idx], '--', label=r'$n = 1$', color=cols_gradient(jdx))
                # ax.plot(sol.t, aux['alpha_noc'] * ymult[idx], '*', label=r'NOC', color=cols_gradient(jdx))
            else:
                ax.plot(sol.t, aux['alpha_ld'] * ymult[idx], ':', color=cols_gradient(jdx))
                ax.plot(sol.t, aux['alpha_mg'] * ymult[idx], '--', color=cols_gradient(jdx))
                # ax.plot(sol.t, aux['alpha_noc'] * ymult[idx], '*', color=cols_gradient(jdx))

if PLOT_REFERENCE:
    axes_u[0].legend()

fig_u.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = (
        r'$\lambda_{h}$', r'$\lambda_{x_N}$', r'$\lambda_{x_E}$',
        r'$\lambda_{V}$', r'$\lambda_{\gamma}$', r'$\lambda_{\psi}$',
        r'$\lambda_{m}$'
    )
    ymult_key = ['h_ref', 'x_ref', 'x_ref', 'v_ref', 'gam_ref', 'psi_ref', 'm_ref']
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sols[-1].lam)):
        axes_costates.append(fig_costates.add_subplot(2, 4, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        for jdx, (sol, aux) in enumerate(zip(sols, auxiliaries)):
            ax.plot(sol.t, sol.lam[idx, :] * aux[ymult_key[idx]], color=cols_gradient(jdx))

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

        ax_e.plot(sol.t, aux['e'], color=cols_gradient(idx))
        ax_hv.plot(aux['mach'], aux['h'], color=cols_gradient(idx))
        ax_qdyn.plot(sol.t, aux['qdyn'], color=cols_gradient(idx))
        ax_ne.plot(aux['xe'], aux['xn'], color=cols_gradient(idx))

        if PLOT_REFERENCE:
            if idx == 0:
                ax_ld.plot(sol.t, aux['ld_mg'], '--', label='n = 1', color=cols_gradient(idx))
                ax_ld.plot(sol.t, aux['ld_max'], ':', label='Max L/D', color=cols_gradient(idx))

                ax_lift.plot(sol.t, aux['load_asymptotic'], '--', label='SPM', color=cols_gradient(idx))

            else:
                ax_ld.plot(sol.t, aux['ld_mg'], '--', color=cols_gradient(idx))
                ax_ld.plot(sol.t, aux['ld_max'], ':', color=cols_gradient(idx))

                ax_lift.plot(sol.t, aux['load_asymptotic'], '--', color=cols_gradient(idx))

    if PLOT_REFERENCE:
        ax_ld.legend()

        ax_hv.plot(aux_ref['mach'], h_interp(aux_ref['e']), 'k--', label='Min{D} (E const)')
        ax_hv.legend()
        # ax_qdyn.legend()

    fig_aero.tight_layout()
    fig_energy_state.tight_layout()


# COMPARE TO GLIDE SLOPE
y_labs = [r'$h - h^*$ [ft]', r'$V - V^*$ [ft/s]', r'$\gamma - \gamma^*$ [deg]', r'$\alpha - \alpha_W$ [deg]', r'$D - D^*$ [g]']
y_keys = ['dh', 'dv', 'dgam', 'dalp', 'ddrag']
y_mult = np.array((1., 1., r2d, r2d, 1.))
n_vals = len(y_keys)

fig_perturbation = plt.figure()
axes_perturbation = []

for idx, (ylab, ykey, ymult) in enumerate(zip(y_labs, y_keys, y_mult)):
    axes_perturbation.append(fig_perturbation.add_subplot(n_vals, 1, idx + 1))
    ax = axes_perturbation[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylab)

    for jdx, aux in enumerate(auxiliaries):
        ax.plot(aux['t'], aux[ykey] * ymult, color=cols_gradient(jdx))


fig_perturbation.tight_layout()

plt.show()
