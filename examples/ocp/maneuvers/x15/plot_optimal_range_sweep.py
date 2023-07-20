import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from x15_aero_model import cla_fun, cd0_fun, cdl_fun, s_ref
from x15_atmosphere import dens_fun, sped_fun
from glide_slope import get_glide_slope

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

gradient = mpl.colormaps['viridis'].colors

PLOT_COSTATE = True
PLOT_AUXILIARY = True
PLOT_REFERENCE = True
DATA = 'sweep_envelope'  # {altitude, velocity, crossrange, sweep_envelope}

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
aux_ref = auxiliaries[0]
interp_dict = get_glide_slope(
    _m=sols[0].x[-1, 0], _h_min=0., _h_max=130e3, _mach_min=0.25, _mach_max=7.0, _use_qdyn_expansion=True
)

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

    auxiliaries[idx]['cla'] = np.asarray(cla_fun(auxiliaries[idx]['mach']), dtype=float).flatten()
    auxiliaries[idx]['cd0'] = np.asarray(cd0_fun(auxiliaries[idx]['mach']), dtype=float).flatten()
    auxiliaries[idx]['cdl'] = np.asarray(cdl_fun(auxiliaries[idx]['mach']), dtype=float).flatten()

    auxiliaries[idx]['ke'] = 0.5 * auxiliaries[idx]['v'] ** 2
    auxiliaries[idx]['pe'] = auxiliaries[idx]['g'] * auxiliaries[idx]['h']
    auxiliaries[idx]['e'] = auxiliaries[idx]['ke'] + auxiliaries[idx]['pe']

    auxiliaries[idx]['qdyn'] = auxiliaries[idx]['ke'] * auxiliaries[idx]['rho']
    _qdyn_s_ref = auxiliaries[idx]['qdyn'] * s_ref
    auxiliaries[idx]['cl'] = auxiliaries[idx]['cla'] * auxiliaries[idx]['alpha']
    auxiliaries[idx]['cd'] = auxiliaries[idx]['cd0'] + auxiliaries[idx]['cdl'] * auxiliaries[idx]['cl'] ** 2
    auxiliaries[idx]['lift'] = _qdyn_s_ref * auxiliaries[idx]['cl']
    auxiliaries[idx]['drag'] = _qdyn_s_ref * auxiliaries[idx]['cd']
    auxiliaries[idx]['alpha_mg'] = \
        auxiliaries[idx]['weight'] / (_qdyn_s_ref * auxiliaries[idx]['cla']) \
        * np.cos(auxiliaries[idx]['gam']) / np.cos(auxiliaries[idx]['phi'])
    auxiliaries[idx]['alpha_ld'] = \
        (auxiliaries[idx]['cd0'] / auxiliaries[idx]['cdl']) ** 0.5 / auxiliaries[idx]['cla']
    auxiliaries[idx]['ld_max'] = \
        auxiliaries[idx]['cla'] * auxiliaries[idx]['alpha_ld'] \
        / (auxiliaries[idx]['cd0']
           + auxiliaries[idx]['cdl'] * (auxiliaries[idx]['cla'] * auxiliaries[idx]['alpha_ld']) ** 2)
    auxiliaries[idx]['ld_mg'] = \
        auxiliaries[idx]['cla'] * auxiliaries[idx]['alpha_mg'] \
        / (auxiliaries[idx]['cd0']
           + auxiliaries[idx]['cdl'] * (auxiliaries[idx]['cla'] * auxiliaries[idx]['alpha_mg']) ** 2)

    auxiliaries[idx]['AD0'] = (_qdyn_s_ref * auxiliaries[idx]['cd0']) / auxiliaries[idx]['weight']
    auxiliaries[idx]['ADL'] = auxiliaries[idx]['cdl'] * auxiliaries[idx]['weight'] / _qdyn_s_ref

    auxiliaries[idx]['load_asymptotic'] = np.empty(sol.t.shape)

    auxiliaries[idx]['v_glide'] = interp_dict['v'](auxiliaries[idx]['e'])
    auxiliaries[idx]['h_glide'] = (auxiliaries[idx]['e'] - 0.5 * auxiliaries[idx]['v_glide']**2) / auxiliaries[idx]['g']
    auxiliaries[idx]['rho_glide'] = np.asarray(dens_fun(auxiliaries[idx]['h_glide'])).flatten()
    auxiliaries[idx]['qdyn_glide'] = 0.5 * auxiliaries[idx]['rho_glide'] * auxiliaries[idx]['v_glide'] ** 2
    auxiliaries[idx]['mach_glide'] = auxiliaries[idx]['v_glide'] \
                                     / np.asarray(sped_fun(auxiliaries[idx]['h_glide'])).flatten()
    auxiliaries[idx]['cd0_glide'] = np.asarray(cd0_fun(auxiliaries[idx]['mach_glide']), dtype=float).flatten()
    auxiliaries[idx]['cdl_glide'] = np.asarray(cdl_fun(auxiliaries[idx]['mach_glide']), dtype=float).flatten()
    auxiliaries[idx]['cla_glide'] = np.asarray(cla_fun(auxiliaries[idx]['mach_glide']), dtype=float).flatten()
    auxiliaries[idx]['AD0_glide'] = (auxiliaries[idx]['qdyn_glide'] * s_ref
                                     * auxiliaries[idx]['cd0_glide']) / auxiliaries[idx]['weight']
    auxiliaries[idx]['ADL_glide'] = auxiliaries[idx]['cdl'] * auxiliaries[idx]['weight'] / (
        auxiliaries[idx]['qdyn_glide'] * s_ref
    )

    auxiliaries[idx]['dh'] = auxiliaries[idx]['h'] - auxiliaries[idx]['h_glide']
    auxiliaries[idx]['dv'] = auxiliaries[idx]['v'] - auxiliaries[idx]['v_glide']
    auxiliaries[idx]['dgam'] = auxiliaries[idx]['gam'] - interp_dict['gam'](auxiliaries[idx]['e'])
    auxiliaries[idx]['dalp'] = auxiliaries[idx]['alpha'] - auxiliaries[idx]['alpha_mg']
    auxiliaries[idx]['ddrag'] = (auxiliaries[idx]['drag']
                                 - interp_dict['D'](auxiliaries[idx]['e'])) / auxiliaries[idx]['weight']

    # _cgam = np.cos(auxiliaries[idx]['gam'])
    _cgam = 1.
    _b = (
             auxiliaries[idx]['AD0'] + auxiliaries[idx]['ADL'] * _cgam**2
             - _cgam * (auxiliaries[idx]['AD0_glide'] + auxiliaries[idx]['ADL_glide'])
         ) / auxiliaries[idx]['ADL']

    auxiliaries[idx]['load_asymptotic'] = np.cos(auxiliaries[idx]['gam'])\
        + np.sign(auxiliaries[idx]['h_glide'] - auxiliaries[idx]['h']) * (
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
    axes_states[4].plot(aux_ref['t'], interp_dict['gam'](aux_ref['e']) * ymult[4], 'k--', label='Glide Slope')
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
    xlabs = (r'$E$ [ft$^2$/s$^2$]', r'$M$', r'$E$ [ft$^2$/s$^2$]', r'$x_E$ [ft]')
    ylabs = (r'$\gamma$ [deg]', 'h [ft]', r'$q_{\infty}$ [psf]', r'$x_N$ [ft]')
    fig_energy_state = plt.figure()

    ax_gam = fig_energy_state.add_subplot(221)
    ax_hv = fig_energy_state.add_subplot(222)
    ax_qdyn = fig_energy_state.add_subplot(223)
    ax_ne = fig_energy_state.add_subplot(224)

    for ax, x_lab, y_lab in zip((ax_gam, ax_hv, ax_qdyn, ax_ne), xlabs, ylabs):
        ax.grid()
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)

    for idx, (sol, aux) in enumerate(zip(sols, auxiliaries)):
        ax_lift.plot(sol.t, aux['lift'] / aux['weight'], color=cols_gradient(idx))
        ax_drag.plot(sol.t, aux['drag'] / aux['weight'], color=cols_gradient(idx))
        ax_ld.plot(sol.t, aux['lift'] / aux['drag'], color=cols_gradient(idx))

        ax_gam.plot(aux['e'], aux['gam'] * 180/np.pi, color=cols_gradient(idx))
        ax_hv.plot(aux['mach'], aux['h'], color=cols_gradient(idx))
        ax_qdyn.plot(aux['e'], aux['qdyn'], color=cols_gradient(idx))
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

        ax_gam.plot(aux_ref['e'], interp_dict['gam'](aux_ref['e']) * 180/np.pi, 'k--', label='Min{D} (E const)')
        ax_qdyn.plot(aux_ref['e'], aux_ref['qdyn_glide'], 'k--', label='Min{D} (E const)')

        ax_hv.plot(aux_ref['mach'], interp_dict['h'](aux_ref['e']), 'k--', label='Min{D} (E const)')
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
