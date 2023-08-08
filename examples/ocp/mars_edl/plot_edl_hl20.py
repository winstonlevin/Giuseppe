import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOT_COSTATE = True
PLOT_AUXILIARY = True
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
r2d = 180 / np.pi
r = k_dict['rm'] + x_dict['h']
g = k_dict['mu'] / r ** 2
g0 = k_dict['mu'] / k_dict['rm'] ** 2
weight = k_dict['mass'] * g

gas_constant_mars = 8.31446261815324 / 43.34  # Universal gas constant R / mean molecular weight M [J/kg-K]
heat_ratio_mars = 1.29  # Mars' specific heat ratio [-]
temperature = g0 * k_dict['h_ref'] / gas_constant_mars  # Sea-level temperature [K]
speed_of_sound = (heat_ratio_mars * gas_constant_mars * temperature) ** 0.5  # Speed of sound [m/s]
if 'gam' in x_dict:
    v = x_dict['v']
    vx = v * np.cos(x_dict['gam'])
    xlabs = (r'$h$ [km]', r'$\theta$ [deg]', r'$V$ [km/s]', r'$\gamma$ [deg]')
    xmult = np.array((1e-3, r2d, 1e-3, r2d))
else:
    v = (x_dict['vx'] ** 2 + x_dict['vn'] ** 2) ** 0.5
    vx = x_dict['vx']
    xlabs = (r'$h$ [km]', r'$\theta$ [deg]', r'$V_x$ [km/s]', r'$V_n$ [km/s]')
    xmult = np.array((1e-3, r2d, 1e-3, 1e-3))

mach = v / speed_of_sound

CL0 = k_dict['CL0']
CL1 = k_dict['CL1']
CD0 = k_dict['CD0']
CD1 = k_dict['CD1']
CD2 = k_dict['CD2']

# (Smoothed) Control Limits
alpha_max = k_dict['alpha_max']
alpha_min = k_dict['alpha_min']
n2_max = k_dict['n2_max']
n2_min = k_dict['n2_min']

rho = k_dict['rho0'] * np.exp(-x_dict['h'] / k_dict['h_ref'])
qdyn = 0.5 * rho * v ** 2
s_ref = k_dict['s_ref']

alpha_upper_limit = alpha_max + 0*sol.t
alpha_lower_limit = alpha_min + 0*sol.t
alpha_upper_limit_smooth = alpha_max + 0*sol.t
alpha_lower_limit_smooth = alpha_min + 0*sol.t

# De-regularize control
eps_alpha = k_dict['eps_alpha']
alpha_reg = sol.u[0, :]

if REG_METHOD in ['trig', 'sin']:
    alpha = 0.5 * ((alpha_upper_limit_smooth - alpha_lower_limit_smooth) * np.sin(alpha_reg)
                   + (alpha_upper_limit_smooth + alpha_lower_limit_smooth))
    dcost_alpha_dt = -eps_alpha * np.cos(alpha_reg)
elif REG_METHOD in ['atan', 'arctan']:
    alpha = (alpha_upper_limit_smooth - alpha_lower_limit_smooth) / np.pi * np.arctan(alpha_reg / eps_alpha) \
            + (alpha_upper_limit_smooth + alpha_lower_limit_smooth) / 2
    dcost_alpha_dt = eps_alpha * np.log(1 + alpha_reg ** 2 / eps_alpha ** 2) / np.pi
else:
    alpha = np.nan * sol.t
    dcost_alpha_dt = np.nan * sol.t

u_dict['alpha'] = alpha

# Aerodynamic Analysis
cl = CL0 + CL1 * alpha
cd = CD0 + CD1 * alpha + CD2 * alpha ** 2
lift = qdyn * k_dict['s_ref'] * cl
lift_g = lift / weight
drag = qdyn * k_dict['s_ref'] * cd
drag_g = drag / weight

alpha_max_ld = - CL0/CL1 + ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5
alpha_min_ld = - CL0/CL1 - ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5

ld_max = (CL0 + CL1 * alpha_max_ld) / (CD0 + CD1 * alpha_max_ld + CD2 * alpha_max_ld ** 2)
ld_min = (CL0 + CL1 * alpha_min_ld) / (CD0 + CD1 * alpha_min_ld + CD2 * alpha_min_ld ** 2)

dtheta_dt = vx / r
dtheta = x_dict["theta"][-1] - x_dict["theta"][0]
cost_dtheta = dtheta / k_dict["theta_scale"]
dcost_h = k_dict['eps_h'] / np.cos(
    np.pi/2 * (2 * x_dict['h'] - k_dict['h_max'] - k_dict['h_min']) / (k_dict['h_max'] - k_dict['h_min'])
) - k_dict['eps_h']

heat_rate_max = k_dict['heat_rate_max']
heat_rate_min = -heat_rate_max
heat_rate = k_dict['k'] * (rho / k_dict['rn']) * v ** 3

# PLOTS ----------------------------------------------------------------------------------------------------------------
t_label = 'Time [s]'

# PLOT STATES
ylabs = xlabs
ymult = xmult
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
        ax.plot(sol.t, alpha_upper_limit_smooth * ymult[idx], '--', color='0.5')
        ax.plot(sol.t, alpha_lower_limit_smooth * ymult[idx], '--', color='0.5')
        ax.plot(sol.t, alpha_upper_limit * ymult[idx], 'k--')
        ax.plot(sol.t, alpha_lower_limit * ymult[idx], 'k--')
    elif PLOT_AUXILIARY and idx == 1:
        ax.plot(sol.t, 0*sol.t + np.pi/2, 'k--')
        ax.plot(sol.t, 0*sol.t - np.pi/2, 'k--')

fig_controls.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = (r'$\lambda_{h}$', r'$\lambda_{\theta}$', r'$\lambda_{V_x}$', r'$\lambda_{V_n}$')
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
    # PLOT AERODYNAMICS
    ydata = (lift_g, drag_g, lift/drag)
    ylabs = (r'$L$ [g]', r'$D$ [g]', r'L/D')

    fig_aero = plt.figure()
    axes_aero = []

    for idx, y in enumerate(ydata):
        axes_aero.append(fig_aero.add_subplot(3, 1, idx + 1))
        ax = axes_aero[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t, y)

        if idx == 2:
            ax.plot(sol.t, 0*sol.t + ld_max, 'k--')
            ax.plot(sol.t, 0*sol.t + ld_min, 'k--')

    # PLOT HEAT RATE
    ydata = (heat_rate, qdyn)
    ylabs = (r'Heat Rate [W/m$^2$]', r'$Q_{\infty}$ [N/m$^2$]')

    fig_aux = plt.figure()
    axes_aux = []

    for idx, y in enumerate(ydata):
        axes_aux.append(fig_aux.add_subplot(2, 1, idx+1))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t, y)

    # ax_heat = fig_heat.add_subplot(111)
    # ax_heat.grid()
    # ax_heat.set_xlabel('t_label')
    # ax_heat.set_ylabel(r'Heat Rate [W/m$^2$]')
    # ax_heat.plot(sol.t, heat_rate)
    # ax_heat.plot(sol.t, 0*sol.t + heat_rate_max, 'k--')
    # ax_heat.plot(sol.t, 0 * sol.t + heat_rate_min, 'k--')

    fig_aux.tight_layout()

    # PLOT COST CONTRIBUTIONS
    ydata = ((dtheta_dt,
             dcost_alpha_dt,
             dcost_h))
    ylabs = (r'$J(\Delta{\theta})$', r'$\Delta{J_{\alpha}}$', r'$\Delta{J_{h}}$')
    sup_title = f'J = {sol.cost}\nJ(Dtheta) = {cost_dtheta} [{abs(cost_dtheta / sol.cost):.2%} of cost]'

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
