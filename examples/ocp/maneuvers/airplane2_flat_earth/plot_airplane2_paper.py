import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from airplane2_aero_atm import mu, re, g0, mass, s_ref, CL0, CLa_fun, CD0_fun, CD1, CD2_fun, max_ld_fun,\
    sped_fun, dens_fun

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

with open('sol_set_range_sweep.data', 'rb') as f:
    sols = pickle.load(f)
    sols = [sols[0], sols[2], sols[1]]
    n_sols = len(sols)

# Process Data
r2d = 180 / np.pi

# Create Dicts
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

    weight = mass * g0
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


plt.show()
