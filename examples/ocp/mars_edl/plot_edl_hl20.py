import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOT_COSTATE = True
PLOT_AUXILIARY = True
RESCALE_COSTATES = False
REG_METHOD = 'sin'
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

# PROCESS DATA ---------------------------------------------------------------------------------------------------------
# De-regularize control
alpha_max = k_dict['alpha_max']
alpha_min = -alpha_max
eps_alpha = k_dict['eps_alpha']
alpha_reg = sol.u[0, :]

if REG_METHOD in ['trig', 'sin']:
    alpha = 0.5 * (alpha_max - alpha_min) * np.sin(alpha_reg) + 0.5 * (alpha_max + alpha_min)
    dcost_alpha_dt = -eps_alpha * np.cos(alpha_reg)
elif REG_METHOD in ['atan', 'arctan']:
    alpha = (alpha_max - alpha_min) / np.pi * np.arctan(alpha_reg / eps_alpha) \
                       + (alpha_max + alpha_min) / 2
    dcost_alpha_dt = eps_alpha * np.log(1 + alpha_reg ** 2 / eps_alpha ** 2) / np.pi
else:
    alpha = np.nan * sol.t
    dcost_alpha_dt = np.nan * sol.t

u_dict['alpha'] = alpha

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

dv_dt = -drag/k_dict['mass'] - g * np.sin(x_dict['gam'])
dv = x_dict["v"][-1] - x_dict["v"][0]
cost_dv = k_dict["k_cost_v"]/k_dict["v_scale"] * dv

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
ylabs = (r'$\alpha$ [deg]', r'$\alpha_{reg}$ [deg]')
ymult = np.array((r2d, 1))
fig_controls = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list((alpha, alpha_reg))):
    axes_u.append(fig_controls.add_subplot(2, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl * ymult[idx])

    if PLOT_AUXILIARY and idx == 0:
        ax.plot(sol.t, 0*sol.t + alpha_max * ymult[idx], 'k--')
        ax.plot(sol.t, 0*sol.t + alpha_min * ymult[idx], 'k--')

fig_controls.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = (r'$\lambda_{h}$', r'$\lambda_{\theta}$', r'$\lambda_{V}$', r'$\lambda_{\gamma}$')
    if RESCALE_COSTATES:
        ymult = np.array((k_dict['h_scale'], k_dict['theta_scale'], k_dict['v_scale'], k_dict['gam_scale']))
    else:
        ymult = np.array((1., 1., 1., 1.))
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sol.lam)):
        axes_costates.append(fig_costates.add_subplot(2, 2, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t, costate * ymult[idx])

    fig_costates.tight_layout()

if PLOT_AUXILIARY:
    # PLOT COST CONTRIBUTIONS
    ydata = (dv_dt * k_dict["k_cost_v"] / k_dict["v_scale"],
             alpha * k_dict["k_cost_alpha"] / k_dict["alpha_scale"],
             dcost_alpha_dt)
    ylabs = (r'$J(\Delta{V})$', r'$J(\alpha)$', r'$\Delta{J_{\alpha}}$')
    sup_title = f'J = {sol.cost}\nJ(DV) = {cost_dv} [{abs(cost_dv / sol.cost):.2%} of cost]'

    fig_cost = plt.figure()
    axes_cost = []

    for idx, cost in enumerate(ydata):
        axes_cost.append(fig_cost.add_subplot(3, 1, idx + 1))
        ax = axes_cost[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t, cost)

    fig_cost.suptitle(sup_title)
    fig_cost.tight_layout()

plt.show()
