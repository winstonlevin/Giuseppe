import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

DATA = 1

with open('guess_hl20.data', 'rb') as f:
    sol = pickle.load(f)
    sol.cost = np.nan

# Create Dict of constants
k_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val

# PROCESS DATA ---------------------------------------------------------------------------------------------------------
r2d = 180 / np.pi
g0 = k_dict['mu'] / k_dict['rm'] ** 2
g = g0
weight = k_dict['mass'] * g
s_ref = k_dict['s_ref']

gas_constant_mars = 8.31446261815324 / 43.34  # Universal gas constant R / mean molecular weight M [J/kg-K]
heat_ratio_mars = 1.29  # Mars' specific heat ratio [-]
temperature = g0 * k_dict['h_ref'] / gas_constant_mars  # Sea-level temperature [K]
speed_of_sound = (heat_ratio_mars * gas_constant_mars * temperature) ** 0.5  # Speed of sound [m/s]

qdyn_vals = np.logspace(2, 4, 1000)

CL0 = k_dict['CL0']
CL1 = k_dict['CL1']
CD0 = k_dict['CD0']
CD1 = k_dict['CD1']
CD2 = k_dict['CD2']

alpha_max = k_dict['alpha_max']
alpha_min = -alpha_max
alpha_stall_max_vals = alpha_max + 0 * qdyn_vals
alpha_stall_min_vals = alpha_min + 0 * qdyn_vals

n_max = k_dict['n_max']
n_min = -n_max

alpha_n_max_vals = weight * n_max / (qdyn_vals * s_ref * CL1) - CL0 / CL1
alpha_n_min_vals = weight * n_min / (qdyn_vals * s_ref * CL1) - CL0 / CL1

alpha_max_vals = np.minimum(alpha_stall_max_vals, alpha_n_max_vals)
alpha_min_vals = np.maximum(alpha_stall_min_vals, alpha_n_min_vals)

# SMOOTH LIMITS --------------------------------------------------------------------------------------------------------
k_vals = np.array((1, 10, 100, 1000))

alpha_max_smooth = []
alpha_min_smooth = []

for k_val in k_vals:
    alpha_max_smooth.append(
        -np.log(np.exp(-alpha_stall_max_vals*k_val) + np.exp(-alpha_n_max_vals*k_val))/k_val
    )

    alpha_min_smooth.append(
        np.log(np.exp(alpha_stall_min_vals*k_val) + np.exp(alpha_n_min_vals*k_val))/k_val
    )

# PLOTTING -------------------------------------------------------------------------------------------------------------
qdyn_lab = r'$Q_{\infty}$ [N / m$^2$]'

fig_bounds = plt.figure()
axes_bounds = []

ydata = (alpha_max_vals, alpha_min_vals)
reflists = (alpha_max_smooth, alpha_min_smooth)
ylabs = (r'$\alpha_{max}$ [deg]', r'$\alpha_{min}$ [deg]')
ymult = (r2d, r2d)

for idx, y in enumerate(ydata):
    axes_bounds.append(fig_bounds.add_subplot(2, 1, idx + 1))
    ax = axes_bounds[-1]
    ax.grid()
    ax.set_xlabel(qdyn_lab)
    ax.set_ylabel(ylabs[idx])
    ax.plot(qdyn_vals, y * ymult[idx])

    for k_val, y_smooth in zip(k_vals, reflists[idx]):
        ax.plot(qdyn_vals, y_smooth * ymult[idx], label=f'k = {k_val}')

    ax.legend()

fig_bounds.tight_layout()

plt.show()
