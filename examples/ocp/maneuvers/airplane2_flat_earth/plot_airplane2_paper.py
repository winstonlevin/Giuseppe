import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from airplane2_aero_atm import mu, re, g0, mass, s_ref, CL0, CLa_fun, CD0_fun, CD1, CD2_fun, max_ld_fun,\
    sped_fun, dens_fun, max_ld_fun_mach, dens_deriv_fun, lut_data

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

with open('sol_set_range_sweep.data', 'rb') as f:
    sols = pickle.load(f)
    sols = [sols[0], sols[2], sols[1]]
    n_sols = len(sols)

# Process Data
r2d = 180 / np.pi
weight = mass * g0

# Create Dicts from solutions ------------------------------------------------------------------------------------------
sol_dicts = []
for idx, sol in enumerate(sols):
    sol_dicts.append({})
    sol_dict = sol_dicts[idx]

    sol_dict['t'] = sol.t

    for key, val in zip(sol.annotations.constants, sol.k):
        sol_dict[key] = val

    for key, x_val, lam_val in zip(sol.annotations.states, list(sol.x), list(sol.lam)):
        sol_dict[key] = x_val
        sol_dict['lam_' + key] = lam_val

    for key, val in zip(sol.annotations.controls, list(sol.u)):
        sol_dict[key] = val

    sol_dict['h'] = sol_dict['h_nd'] * sol_dict['h_scale']
    h = sol_dict['h']
    sol_dict['v'] = sol_dict['v_nd'] * sol_dict['v_scale']
    v = sol_dict['v']
    gam = sol_dict['gam']

    sol_dict['weight'] = weight

    mach = v / np.asarray(sped_fun(h)).flatten()
    sol_dict['mach'] = mach
    rho = np.asarray(dens_fun(h)).flatten()
    sol_dict['rho'] = rho

    qdyn = 0.5 * rho * v ** 2
    sol_dict['qdyn'] = qdyn
    CL = sol_dict['CL']
    sol_dict['CL'] = CL
    CD0 = np.asarray(CD0_fun(mach)).flatten()
    sol_dict['CD0'] = CD0
    CD2 = np.asarray(CD2_fun(mach)).flatten()
    sol_dict['CD2'] = CD2
    CD = CD0 + CD1 * CL + CD2 * CL**2
    sol_dict['CD'] = CD

    CLa = np.asarray(CLa_fun(mach)).flatten()
    sol_dict['CLa'] = CLa
    alpha = (CL - CL0) / CLa
    sol_dict['alpha'] = alpha
    max_ld_info = max_ld_fun(_CLa=CLa, _CD0=CD0, _CD2=CD2)
    sol_dict['alpha_max_ld'] = max_ld_info['alpha']
    sol_dict['LD_max'] = max_ld_info['LD']
    sol_dict['LD'] = CL / CD

    lift = qdyn * s_ref * CL
    sol_dict['lift'] = lift
    drag = qdyn * s_ref * CD
    sol_dict['drag'] = drag

    sol_dict['dgam_dt'] = lift / (mass * v) - g0/v * np.cos(gam)
    sol_dict['E'] = 0.5 * v**2 + g0 * h

# Calculate "equilibrium glide" ----------------------------------------------------------------------------------------
h_vals = np.sort(sol_dicts[-1]['h'])
CL_ld_max = np.max(sol_dicts[-1]['CL'])
CL_ld_min = np.min(sol_dicts[-1]['CL'])


def binary_search(_x_min, _x_max, _f, _f_val_target, max_iter: int = 1000, tol: float = 1e-3):
    increasing = _f(_x_max) > _f(_x_min)
    if increasing:
        def _f_wrapped(_x):
            return _f(_x)
    else:
        _f_val_target = -_f_val_target

        def _f_wrapped(_x):
            return -_f(_x)

    for _ in range(max_iter):
        # Binary search
        _x_guess = 0.5 * (_x_min + _x_max)
        _f_val = _f_wrapped(_x_guess)
        if _f_val < _f_val_target:
            # x too low, try higher
            _x_min = _x_guess
        else:
            # x too high, try lower
            _x_max = _x_guess
        if _x_max - _x_min < tol:
            break

    _x_guess = 0.5 * (_x_min + _x_max)
    return _x_guess


rho_vals = np.asarray(dens_fun(h_vals)).flatten()
a_vals = np.asarray(sped_fun(h_vals)).flatten()
v_vals = np.empty(h_vals.shape)

for idx, rho_val in enumerate(rho_vals):
    v_min = (2 * weight / (rho_val * s_ref * 1.1 * CL_ld_max)) ** 0.5
    v_max = (2 * weight / (rho_val * s_ref * 0.9 * CL_ld_min))**0.5
    v_vals[idx] = binary_search(
        v_min, v_max,
        lambda _v: 0.5 * rho_val * _v**2 * s_ref * float(max_ld_fun_mach(_v/a_vals[idx])['CL']) / weight - 1, 0.
    )

e_vals = 0.5 * v_vals**2 + g0 * h_vals
mach_vals = v_vals / a_vals
max_ld_vals_dict = max_ld_fun_mach(mach_vals)
beta_vals = - dens_deriv_fun(h_vals) / rho_vals
gam_ss_vals = -np.arcsin(
    (max_ld_vals_dict['LD'] * (1 + beta_vals * v_vals**2 / (2*g0)))**-1
)

# PLOTTING -------------------------------------------------------------------------------------------------------------
t_label = 'Time'
# t_label = 'Time [s]'

# PLOT h and gam vs. Mach
ylabs = (r'$h$', r'$\gamma$')
# ylabs = (r'$h$ [1,000 ft]', r'$\gamma$ [deg]')
ymult = np.array((1e-3, r2d))
fig_states = plt.figure()
axes_states = []
t_lim = 350.

# ylabs_troubleshoot = (r'Mach [-]', r'$Q_{\infty}$ [psf]', r'$d\gamma/dt$ [deg/s]', r'$L/D$ [-]')
# ymult_troubleshoot = np.array((1., 1., r2d, 1.))
ylabs_troubleshoot = (r'$d\gamma/dt$', r'$L/D$')
# ylabs_troubleshoot = (r'$d\gamma/dt$ [deg/s]', r'$L/D$ [-]')
ymult_troubleshoot = np.array((r2d, 1.))
fig_troubleshoot = plt.figure()
axes_troubleshoot = []

for idx, y_lab in enumerate(ylabs):
    axes_states.append(fig_states.add_subplot(2, 1, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(y_lab)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

for idx, y_lab in enumerate(ylabs_troubleshoot):
    axes_troubleshoot.append(fig_troubleshoot.add_subplot(2, 1, idx+1))
    ax = axes_troubleshoot[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(y_lab)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

for sol_dict in sol_dicts:
    xdata = sol_dict['t']
    idces, = np.nonzero(xdata < t_lim)
    ydata = (sol_dict['h'][idces], sol_dict['gam'][idces])
    # ydata_troubleshoot = (
    #     sol_dict['mach'][idces], sol_dict['qdyn'][idces], sol_dict['dgam_dt'][idces], sol_dict['LD'][idces]
    # )
    # yaux_troubleshoot = (None, None, None, sol_dict['LD_max'][idces])
    ydata_troubleshoot = (sol_dict['dgam_dt'][idces], sol_dict['LD'][idces])
    yaux_troubleshoot = (0.*sol_dict['dgam_dt'][idces], sol_dict['LD_max'][idces])
    xdata = sol_dict['t'][idces]

    for idx, y in enumerate(ydata):
        ax = axes_states[idx]
        ax.plot(xdata, y * ymult[idx])

    for idx, y in enumerate(ydata_troubleshoot):
        ax = axes_troubleshoot[idx]
        ax.plot(xdata, y * ymult_troubleshoot[idx])

        if yaux_troubleshoot[idx] is not None:
            ax.plot(xdata, yaux_troubleshoot[idx], 'k--')

fig_states.tight_layout()
fig_troubleshoot.tight_layout()

# Get staged plots -----------------------------------------------------------------------------------------------------
# Hide all lines
for ax in axes_troubleshoot:
    for line in ax.lines:
        line.set_alpha(0.)

fig_troubleshoot.savefig('fig_dgamdt_ld_0.svg')

# Make lines appear sequentially
for idx in range(n_sols):
    for ax in axes_troubleshoot:
        n_lines = len(ax.lines)
        n_reveal = int(n_lines / n_sols)

        lines_mod = ax.lines[n_reveal*idx:n_reveal*(idx+1)]
        for line in lines_mod:
            line.set_alpha(1.)
        if idx > 0:
            lines_mod = ax.lines[n_reveal*(idx-1):n_reveal*(idx)]
            for line in lines_mod:
                line.set_alpha(0.25)

    fig_troubleshoot.savefig('fig_dgamdt_ld_' + str(idx + 1) + '.svg')

# Hide all lines
for ax in axes_states:
    for line in ax.lines:
        line.set_alpha(0.)

fig_states.savefig('fig_h_gam_0.svg')

# Make lines appear sequentially
for idx in range(n_sols):
    for ax in axes_states:
        n_lines = len(ax.lines)
        n_reveal = int(n_lines / n_sols)

        lines_mod = ax.lines[n_reveal*idx:n_reveal*(idx+1)]
        for line in lines_mod:
            line.set_alpha(1.)
        if idx > 0:
            lines_mod = ax.lines[n_reveal*(idx-1):n_reveal*(idx)]
            for line in lines_mod:
                line.set_alpha(0.25)

    fig_states.savefig('fig_h_gam_' + str(idx + 1) + '.svg')

# Plot equilibrium glide -----------------------------------------------------------------------------------------------
fig_he = plt.figure()
ax_he = fig_he.add_subplot(111)
ax_he.grid()
ax_he.set_xlabel(r'$E$', size=20)
ax_he.set_ylabel(r'$h$', size=20)
ax_he.set_xticklabels([])
ax_he.set_yticklabels([])
ax_he.plot(e_vals, h_vals, 'k--', linewidth=4)
fig_he.tight_layout()
fig_he.savefig('fig_he.svg')

# Plot steady-state FPA ------------------------------------------------------------------------------------------------
e_min = 1.8e6
fig_gam_ss = plt.figure()
ax_gam_ss = fig_gam_ss.add_subplot(111)
ax_gam_ss.grid()
ax_gam_ss.set_xlabel(r'$E$', size=20)
ax_gam_ss.set_ylabel(r'$\gamma$', size=20)
ax_gam_ss.set_xticklabels([])
ax_gam_ss.set_yticklabels([])
for sol_dict in sol_dicts:
    e_idcs, = np.nonzero(sol_dict['E'] > e_min)
    ax_gam_ss.plot(sol_dict['E'][e_idcs], sol_dict['gam'][e_idcs])
e_idcs, = np.nonzero(e_vals > e_min)
ax_gam_ss.plot(e_vals[e_idcs], gam_ss_vals[e_idcs], 'k--', linewidth=2, label=r'$\gamma_{\mathrm{ss}}$')
ax_gam_ss.legend(fontsize=20)
fig_gam_ss.tight_layout()
fig_gam_ss.savefig('fig_gam_ss.svg')

# Plot aerodynamic model
mach_vals_plot = np.linspace(lut_data['M'][0], lut_data['M'][-1], 100)
ld_data_vals_plot = max_ld_fun_mach(mach_vals_plot)

fig_aerodynamics = plt.figure()

ax_ld = fig_aerodynamics.add_subplot(211)
ax_ld.grid()
ax_ld.set_ylabel(r'Max $L/D$')
ax_ld.plot(mach_vals_plot, ld_data_vals_plot['LD'], linewidth=2)

ax_cl = fig_aerodynamics.add_subplot(212)
ax_cl.grid()
ax_cl.set_xlabel(r'Mach')
ax_cl.set_ylabel(r'$C_L^*$')
ax_cl.plot(mach_vals_plot, ld_data_vals_plot['CL'], linewidth=2)

fig_aerodynamics.tight_layout()
fig_aerodynamics.savefig('fig_aero_model.svg')

plt.show()
