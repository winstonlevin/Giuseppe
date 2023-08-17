import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.interpolate import PchipInterpolator

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

PLOT_COSTATE = True
RESCALE_COSTATES = True

PLOT_AUXILIARY = True
PLOT_SWEEP = False
OPTIMIZATION = 'min_time'

# REG_METHOD = 'sin'
REG_METHOD = None
DATA = 1

if DATA == 0:
    with open('guess_hl20.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan

    sols = [sol]
elif DATA == 1:
    with open('seed_sol_hl20.data', 'rb') as f:
        sol = pickle.load(f)

    sols = [sol]
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
h = x_dict['h_nd'] * k_dict['h_scale']
theta = x_dict['theta_nd'] * k_dict['theta_scale']
v = x_dict['v_nd'] * k_dict['v_scale']
gam = x_dict['gam_nd'] * k_dict['gam_scale']
alpha = x_dict['alpha_nd'] * k_dict['alpha_scale']

r2d = 180 / np.pi
r = k_dict['rm'] + h
g = k_dict['mu'] / r ** 2
g0 = k_dict['mu'] / k_dict['rm'] ** 2
weight = k_dict['mass'] * g

gas_constant_mars = 8.31446261815324 / 43.34  # Universal gas constant R / mean molecular weight M [J/kg-K]
heat_ratio_mars = 1.29  # Mars' specific heat ratio [-]
temperature = g0 * k_dict['h_ref'] / gas_constant_mars  # Sea-level temperature [K]
speed_of_sound = (heat_ratio_mars * gas_constant_mars * temperature) ** 0.5  # Speed of sound [m/s]

xlabs = (r'$h$ [km]', r'$\theta$ [deg]', r'$V$ [km/s]', r'$\gamma$ [deg]', r'$\alpha$ [deg]')
lamlabs = (r'$\lambda_{h}$', r'$\lambda_{\theta}$', r'$\lambda_{V}$', r'$\lambda_{\gamma}$', r'$\lambda_{\alpha}$')
xmult = np.array(
    (1e-3 * k_dict['h_scale'], r2d * k_dict['theta_scale'], 1e-3 * k_dict['v_scale'], r2d * k_dict['gam_scale'],
     r2d * k_dict['alpha_scale'])
)
if RESCALE_COSTATES:
    lammult = np.array((
        1/k_dict['h_scale'], 1/k_dict['theta_scale'], 1/k_dict['v_scale'], 1/k_dict['gam_scale'],
        1/k_dict['alpha_scale']
    ))
else:
    lammult = np.ones((len(lam_dict),))

mach = v / speed_of_sound

CL0 = k_dict['CL0']
CL1 = k_dict['CL1']
CD0 = k_dict['CD0']
CD1 = k_dict['CD1']
CD2 = k_dict['CD2']

# Control Limits
alpha_rate_max = k_dict['alpha_rate_max']
alpha_rate_min = k_dict['alpha_rate_min']

# De-regularize control
eps_alpha_rate = k_dict['eps_alpha_rate']
alpha_rate_reg = sol.u[0, :]

if REG_METHOD in ['trig', 'sin']:
    def reg2ctrl(_u_reg, _u_max, _u_min, _eps_u):
        return 0.5 * ((_u_max - _u_min) * np.sin(_u_reg) + (_u_max + _u_min))

    def reg2cost(_u_reg, _eps_u):
        return -_eps_u * np.cos(_u_reg)
elif REG_METHOD in ['atan', 'arctan']:
    def reg2ctrl(_u_reg, _u_max, _u_min, _eps_u):
        return (_u_max - _u_min) / np.pi * np.arctan(_u_reg / _eps_u) + 0.5 * (_u_max + _u_min)

    def reg2cost(_u_reg, _eps_u):
        _eps_u * np.log(1 + _u_reg ** 2 / _eps_u ** 2 ) / np.pi
else:
    def reg2ctrl(_u_reg, _u_max, _u_min, _eps_u):
        return _u_reg

    def reg2cost(_u_reg, _eps_u):
        return 0 * _u_reg

alpha_rate = reg2ctrl(alpha_rate_reg, alpha_rate_max, alpha_rate_min, eps_alpha_rate)
dcost_alpha_dt = reg2cost(alpha_rate_reg, eps_alpha_rate)

# Aerodynamic Analysis
s_ref = k_dict['s_ref']
rho = k_dict['rho0'] * np.exp(-h / k_dict['h_ref'])
qdyn = 0.5 * rho * v**2

# # Polynomial CL/CD Model
# cl = CL0 + CL1 * alpha
# cd = CD0 + CD1 * alpha + CD2 * alpha ** 2

# Trigonometric CL/CD Model
cl = CL1 * 0.5 * np.sin(2 * (alpha + CL0/CL1))
cd = CD0 - CD1**2/(4*CD2) + CD2 * np.sin(alpha + CD1/(2*CD2))**2

lift = qdyn * k_dict['s_ref'] * cl
lift_g = lift / weight
drag = qdyn * k_dict['s_ref'] * cd
drag_g = drag / weight

alpha_max_ld = - CL0/CL1 + ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5
alpha_min_ld = - CL0/CL1 - ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5

ld_max = (CL0 + CL1 * alpha_max_ld) / (CD0 + CD1 * alpha_max_ld + CD2 * alpha_max_ld ** 2)
ld_min = (CL0 + CL1 * alpha_min_ld) / (CD0 + CD1 * alpha_min_ld + CD2 * alpha_min_ld ** 2)

g_load2 = (lift**2 + drag**2) / (k_dict['mass'] * g0) ** 2
g_load = g_load2 ** 0.5
g_load_max = k_dict['n2_max'] ** 0.5

if OPTIMIZATION == 'max_range':
    dcost_dt = (-v * np.cos(gam) / r) / k_dict['theta_scale']
    cost = -(theta[-1] - theta[0]) / k_dict['theta_scale']
    cost_lab = r'$J(\Delta{\theta})$'
elif OPTIMIZATION == 'min_time':
    dcost_dt = 0*sol.t + 1.
    cost = sol.t[-1] - sol.t[0]
    cost_lab = r'$J(\Delta{t})$'
else:
    dcost_dt = 0 * sol.t
    cost = 0.
    cost_lab = '[Invalid Opt. Selected]'

if OPTIMIZATION == 'min_time':
    eps_utm = k_dict['eps_n2']
    x_max_utm = k_dict['n2_max']
    x_min_utm = k_dict['n2_min']
    x_utm = g_load2
    lab_utm = r'$\Delta{J_{n^2}}$'
else:
    eps_utm = k_dict['eps_h']
    x_max_utm = k_dict['h_max'] / k_dict['h_scale']
    x_min_utm = k_dict['h_min'] / k_dict['h_scale']
    x_utm = h / k_dict['h_scale']
    lab_utm = r'$\Delta{J_{h}}$'

dcost_utm = eps_utm / np.cos(
    np.pi / 2 * (2 * x_utm - x_max_utm - x_min_utm) / (x_max_utm - x_min_utm)
) - eps_utm

heat_rate_max = k_dict['heat_rate_max']
heat_rate_min = -heat_rate_max
heat_rate = k_dict['k'] * (rho / k_dict['rn']) * v ** 3

mass = k_dict['mass']
mu = k_dict['mu']
rm = k_dict['rm']
rho0 = k_dict['rho0']
h_ref = k_dict['h_ref']
CL_max_ld = CL0 + CL1 * alpha_max_ld

h_glide = np.linspace(0., 80e3, 1000)
r_glide = rm + h_glide
g_glide = mu / r_glide**2
rho_glide = rho0 * np.exp(-h_glide/h_ref)
v_glide = _v = ((mass * g_glide) / (0.5 * rho_glide * s_ref * CL_max_ld + mass / r_glide)) ** 0.5
qdyn_glide = mass * (g_glide - v_glide**2 / r_glide) / (s_ref * CL_max_ld)
qdyn_glide_interp = PchipInterpolator(x=h_glide, y=qdyn_glide)

# Time scale
tf = sol.t[-1]

if tf < 60 * 3:
    # Order of seconds
    t_label = 'Time [s]'
    t_mult = 1.
elif tf < 60 * 60 * 3:
    # Order of minutes
    t_label = 'Time [min]'
    t_mult = 1./60.
elif tf < 60 * 60 * 25 * 3:
    # Order of hours
    t_label = 'Time [hours]'
    t_mult = 1. / (60. * 60.)
else:
    # Order of days
    t_label = 'Time [days]'
    t_mult = 1. / (60. * 60. * 24.)

# PLOTS ----------------------------------------------------------------------------------------------------------------
# PLOT STATES
ylabs = xlabs
ymult = xmult
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(3, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    if PLOT_SWEEP:
        for sol_sweep in sols:
            ax.plot(sol_sweep.t * t_mult, sol_sweep.x[idx, :] * ymult[idx])
    else:
        ax.plot(sol.t * t_mult, state * ymult[idx])

    if idx == 0:
        ax.plot(sol.t * t_mult, 0*sol.t + k_dict['h_min'] / k_dict['h_scale'] * ymult[idx], 'k--')

fig_states.tight_layout()

# PLOT CONTROL
ylabs = (r'$\dot{\alpha}$ [deg/s]', r'$\dot{\alpha}_{reg}$ [-]')
ymult = np.array((r2d, 1))
fig_controls = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list((alpha_rate, alpha_rate_reg))):
    axes_u.append(fig_controls.add_subplot(2, 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t * t_mult, ctrl * ymult[idx])

    if idx == 0:
        ax.plot(sol.t * t_mult, 0*sol.t + k_dict['alpha_rate_max'] * ymult[idx], 'k--')
        ax.plot(sol.t * t_mult, 0*sol.t + k_dict['alpha_rate_min'] * ymult[idx], 'k--')
    elif idx == 1:
        ax.plot(sol.t * t_mult, 0*sol.t + np.pi/2, 'k--')
        ax.plot(sol.t * t_mult, 0*sol.t - np.pi/2, 'k--')

fig_controls.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = lamlabs
    ymult = lammult
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sol.lam)):
        axes_costates.append(fig_costates.add_subplot(3, 2, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t * t_mult, costate * ymult[idx])

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
        ax.plot(sol.t * t_mult, y)

        if idx == 2:
            ax.plot(sol.t * t_mult, 0*sol.t + ld_max, 'k--')
            ax.plot(sol.t * t_mult, 0*sol.t + ld_min, 'k--')

    # PLOT H-v
    fig_hv = plt.figure()
    ax_hv = fig_hv.add_subplot(111)
    ax_hv.grid()
    ax_hv.set_xlabel(xlabs[2])
    ax_hv.set_ylabel(xlabs[0])

    if PLOT_SWEEP:
        for sol_sweep in sols:
            ax_hv.plot(sol_sweep.x[2, :] * xmult[2], sol_sweep.x[0, :] * xmult[0])
    else:
        ax_hv.plot(sol.x[2, :] * xmult[2], sol.x[0, :] * xmult[0])

    ax_hv.plot(v_glide / k_dict['v_scale'] * xmult[2], h_glide / k_dict['h_scale'] * xmult[0],
               'k--', label='Glide Slope')
    ax_hv.legend()

    # PLOT CONSTRAINTS
    ydata = (heat_rate, qdyn, g_load)
    yaux = (0*sol.t + k_dict['heat_rate_max'], qdyn_glide_interp(h), 0*sol.t + g_load_max)
    ylabs = (r'Heat Rate [W/m$^2$]', r'$Q_{\infty}$ [N/m$^2$]', 'G-Load [g\'s]')
    yauxlabs = ('Max Heat Rate', 'Gliding Dynamic Pressure', 'Max G-Load')

    fig_aux = plt.figure()
    axes_aux = []

    for idx, y in enumerate(ydata):
        axes_aux.append(fig_aux.add_subplot(3, 1, idx+1))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t * t_mult, y)
        ax.plot(sol.t * t_mult, yaux[idx], 'k--', label=yauxlabs[idx])
        ax.legend()

    # ax_heat = fig_heat.add_subplot(111)
    # ax_heat.grid()
    # ax_heat.set_xlabel('t_label')
    # ax_heat.set_ylabel(r'Heat Rate [W/m$^2$]')
    # ax_heat.plot(sol.t * t_mult, heat_rate)
    # ax_heat.plot(sol.t * t_mult, 0*sol.t + heat_rate_max, 'k--')
    # ax_heat.plot(sol.t * t_mult, 0 * sol.t + heat_rate_min, 'k--')

    fig_aux.tight_layout()

    # PLOT COST CONTRIBUTIONS
    ydata = ((dcost_dt,
              dcost_alpha_dt,
             dcost_utm))
    ylabs = (cost_lab, r'$\Delta{J_{\dot{\alpha}}}$', lab_utm)
    sup_title = f'J_UTM = {sol.cost}\nJ = {cost} [{abs(cost / sol.cost):.2%} of cost]'

    fig_cost = plt.figure()
    axes_cost = []

    for idx, cost in enumerate(ydata):
        axes_cost.append(fig_cost.add_subplot(3, 1, idx + 1))
        ax = axes_cost[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t * t_mult, cost)

    fig_cost.suptitle(sup_title)
    fig_cost.tight_layout()

plt.show()
