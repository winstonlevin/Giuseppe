import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']
DATA = 2
PLOT_AUX = True

# Load data
if DATA == 0:
    with open('guess.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan
        sol.h_u = np.nan * sol.u.copy()
        sol.eig_h_uu = np.nan * sol.u.copy()
elif DATA == 1:
    with open('seed_sol.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set.data', 'rb') as f:
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

# Process data
r2d = 180/np.pi
v = (2*(k_dict['E'] - k_dict['g'] * x_dict['h']))**0.5
ham = 1. + lam_dict['h'] * v * np.sin(u_dict['gam']) + lam_dict['x'] * v * np.cos(u_dict['gam'])
gam_analytic = np.arctan2(-lam_dict['h'], -lam_dict['x'])

if PLOT_AUX:
    lam_h2 = lam_dict['h']**2
    lam_h2_analytic = lam_dict['h'][0]**2 - 1/v[0]**2 + 1/v**2
    r_gam = (lam_dict['h']**2 + lam_dict['x']**2)**0.5
    r_gam_analytic = 1 / v
    v_max_idx = np.argmax(v)
    v_max_analytic = np.abs(1 / np.mean(lam_dict['x']))
    # v_phase = np.arcsin(lam_)
    # v_analytic =
else:
    lam_h2 = None
    lam_h2_analytic = None
    r_gam = None
    r_gam_analytic = None

# PLOTTING -------------------------------------------------------------------------------------------------------------
t_label = 'Time [s]'

# PLOT STATES/CONTROLS
ylabs = (r'$h$ [m]', r'$V$ [m/s]', r'$x$ [m]', r'$\gamma$ [deg]')
ymult = np.array((1., 1., 1., r2d))
ydata = (x_dict['h'], v, x_dict['x'], u_dict['gam'])
yaux = (None, None, None, gam_analytic)
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(ydata):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

    if yaux[idx] is not None:
        ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--')

fig_states.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{h}$ [s/m]', r'$\lambda_{x}$ [s/m]', r'$H_u$ [1/rad]', r'$H_{uu}$ [1/rad$^2$]', r'$H$ [-]')
ymult = np.array((1., 1., 1., 1., 1.))
ydata = (lam_dict['h'], lam_dict['x'], sol.h_u.flatten(), sol.eig_h_uu.flatten(), ham)
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(ydata):
    axes_costates.append(fig_costates.add_subplot(3, 2, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate * ymult[idx])

fig_costates.tight_layout()

if PLOT_AUX:
    ydata = (lam_h2, r_gam,)
    yaux = (lam_h2_analytic, r_gam_analytic,)
    yerr = (lam_h2 - lam_h2_analytic, r_gam - r_gam_analytic,)
    ylabs_names = (r'$\lambda_h^2$', r'$R_{\gamma}$')
    ylabs_units = (r'[s$^2$/m$^s$]', r'[s/m]')
    n_aux = len(ydata)
    fig_aux = plt.figure()
    axes_aux = []

    for idx, y in enumerate(ydata):
        axes_aux.append(fig_aux.add_subplot(n_aux, 2, 2*idx + 1))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs_names[idx] + ' ' + ylabs_units[idx])
        ax.plot(sol.t, y * ymult[idx])
        ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--')

        axes_aux.append(fig_aux.add_subplot(n_aux, 2, 2 * idx + 2))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs_names[idx] + ' err. ' + ylabs_units[idx])
        ax.plot(sol.t, yerr[idx] * ymult[idx])

    fig_aux.tight_layout()

plt.show()
