import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from space_shuttle_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, sped_fun, dens_fun
from glide_slope import get_glide_slope

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOT_COSTATE = True
PLOT_AUXILIARY = True
DATA = 0

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

x_dict['h'] = x_dict['h_nd'] * k_dict['h_scale']
x_dict['v'] = x_dict['v_nd'] * k_dict['v_scale']

g = mu / (re + x_dict['h']) ** 2
weight = mass * g

mach = x_dict['v'] / np.asarray(sped_fun(x_dict['h'])).flatten()
rho = np.asarray(dens_fun(x_dict['h'])).flatten()

qdyn = 0.5 * rho * x_dict['v'] ** 2
lift = u_dict['lift_nd'] * g0 * mass
drag = qdyn * s_ref * CD0 + CD1 * lift + CD2 / (qdyn * s_ref) * lift**2

# Glide Slope
ke = 0.5 * x_dict['v']**2
pe = mu / re - mu / (re + x_dict['h'])
e = ke + pe
glide_dict = get_glide_slope(e_vals=e, remove_nan=False)
e_glide = glide_dict['E']
h_glide = glide_dict['h']
v_glide = glide_dict['v']
gam_glide = glide_dict['gam']
g_glide = glide_dict['g']
u_glide = glide_dict['u']

# Convert E, h, gam costates to V, h, gam costates
# d(E)/dt = g d(h)/dt + V d(V)/dt
# -> lam_v = V lam_E
# -> lam_h = lam_h + g lam_E
lam_v_glide = glide_dict['lam_E'] * v_glide
lam_h_glide = glide_dict['lam_h'] + g_glide * glide_dict['lam_E']
lam_gam_glide = glide_dict['lam_gam']

glide_dict_full = get_glide_slope()
glide_dict_full_flat = get_glide_slope(flat_earth=True)

# PLOTTING -------------------------------------------------------------------------------------------------------------
t_label = 'Time [s]'

# PLOT STATES
ylabs = (r'$h$ [1,000 ft]', r'$\theta$ [deg]', r'$V$ [1,000 ft/s]', r'$\gamma$ [deg]')
ymult = np.array((1e-3 * k_dict['h_scale'], r2d, 1e-3 * k_dict['v_scale'], r2d))
yaux = (h_glide/k_dict['h_scale'], None, v_glide/k_dict['v_scale'], gam_glide)
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

    if yaux[idx] is not None:
        ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--', label='Glide Slope')

axes_states[0].legend()

fig_states.tight_layout()

# PLOT U
ylabs = (r'$L/W_0$ [g]',)
ymult = np.array((1.,))
yaux = (u_glide,)
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(sol.u)):
    axes_u.append(fig_u.add_subplot(1, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl * ymult[idx])
    ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--', label='Glide Slope')
axes_u[0].legend()
fig_u.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = (
        r'$\lambda_{h}$', r'$\lambda_{\theta}$', r'$\lambda_{V}$', r'$\lambda_{\gamma}$'
    )
    yaux = (lam_h_glide * k_dict['h_scale'], None, lam_v_glide * k_dict['v_scale'], lam_gam_glide)
    ymult = np.array((1 / k_dict['h_scale'], 1., 1 / k_dict['v_scale'], 1.))
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sol.lam)):
        axes_costates.append(fig_costates.add_subplot(2, 2, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t, costate * ymult[idx])

        if yaux[idx] is not None:
            ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--', label='Glide Slope')

    axes_costates[0].legend()

    fig_costates.tight_layout()

if PLOT_AUXILIARY:
    if sol.h_u is not None:
        h_u = sol.h_u[0, :]
    else:
        h_u = None
    if sol.eig_h_uu is not None:
        eig_h_uu = sol.eig_h_uu[0, :]
    else:
        eig_h_uu = None
    ydata = (h_u, eig_h_uu)
    yaux = (0 * sol.t, 0 * sol.t)
    ylabs = (r'$H_u$ [1/rad]', r'eig($H_{uu}$) [1/rad$^2$]')
    yauxlabs = (r'$H_u = 0$', r'$H_{uu} > 0$')

    fig_aux = plt.figure()
    axes_aux = []

    for idx, y in enumerate(ydata):
        axes_aux.append(fig_aux.add_subplot(1, 2, idx+1))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        if y is not None:
            ax.plot(sol.t, y)
        if yaux[idx] is not None:
            ax.plot(sol.t, yaux[idx], 'k--', label=yauxlabs[idx])
        if yauxlabs[idx] is not None:
            ax.legend()
    fig_aux.tight_layout()

    fig_hv = plt.figure()
    ax_hv = fig_hv.add_subplot(111)
    ax_hv.grid()
    ax_hv.plot(x_dict['v'] / 1e3, x_dict['h'] / 1e3)
    ax_hv.plot(glide_dict_full['v'] / 1e3, glide_dict_full['h'] / 1e3,
               'k--', label='Glide Slope (Sph.)')
    ax_hv.plot(glide_dict_full_flat['v'] / 1e3, glide_dict_full_flat['h'] / 1e3,
               'k:', label='Glide Slope (Flat)')
    ax_hv.set_xlabel(r'$V$ [1,000 ft/s]')
    ax_hv.set_ylabel(r'$h$ [1,000 ft]')

plt.show()
