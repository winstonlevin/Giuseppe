import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOT_COSTATE = True
PLOT_AUXILIARY = True
DATA = 2

if DATA == 0:
    with open('guess_hl20.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan
elif DATA == 1:
    with open('seed_sol_hl20.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set_hl20.data', 'rb') as f:
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
g = k_dict['mu'] / (k_dict['rm'] + x_dict['h']) ** 2
g0 = k_dict['mu'] / k_dict['rm'] ** 2
weight = k_dict['mass'] * g

gas_constant_mars = 8.31446261815324 / 43.34  # Universal gas constant R / mean molecular weight M [J/kg-K]
heat_ratio_mars = 1.29  # Mars' specific heat ratio [-]
temperature = g0 * k_dict['h_ref'] / gas_constant_mars  # Sea-level temperature [K]
speed_of_sound = (heat_ratio_mars * gas_constant_mars * temperature) ** 0.5  # Speed of sound [m/s]
mach = x_dict['v'] / speed_of_sound

cl = k_dict['CL0'] + k_dict['CL1'] * u_dict['alpha']
cd = k_dict['CD0'] + k_dict['CD1'] * u_dict['alpha'] * k_dict['CD2'] * u_dict['alpha'] ** 2
rho = k_dict['rho0'] * np.exp(-x_dict['h'] / k_dict['h_ref'])
qdyn = 0.5 * rho * x_dict['v'] ** 2
lift = qdyn * k_dict['s_ref'] * cl
lift_g = lift / weight
drag = qdyn * k_dict['s_ref'] * cd
drag_g = drag / weight

# PLOTS ----------------------------------------------------------------------------------------------------------------
t_label = 'Time [s]'

# PLOT STATES
ylabs = (r'$h$ [km]', r'$\theta$ [deg]', r'$V$ [km/s]', r'$\gamma$ [deg]')
ymult = np.array((1e-3, r2d, 1e-3, r2d))
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

fig_states.tight_layout()

# PLOT CONTROL
ylabs = (r'$\alpha$ [deg]',)
ymult = np.array((r2d,))
fig_controls = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(sol.u)):
    axes_u.append(fig_controls.add_subplot(1, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl * ymult[idx])

fig_controls.tight_layout()

plt.show()
