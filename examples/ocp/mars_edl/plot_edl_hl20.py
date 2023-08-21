from typing import List
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.interpolate import PchipInterpolator

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']
gradient = mpl.colormaps['viridis'].colors

PLOT_COSTATE = True
RESCALE_COSTATES = True

PLOT_AUXILIARY = True
PLOT_SWEEP = False
OPTIMIZATION = 'min_energy'

# REG_METHOD = 'sin'
REG_METHOD = None
DATA = 2
SWEEP = None

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
    if SWEEP == 'gam':
        with open('sol_set_hl20_gam.data', 'rb') as f:
            sols = pickle.load(f)
            sol = sols[-100]
    else:
        with open('sol_set_hl20.data', 'rb') as f:
            sols = pickle.load(f)
            sol = sols[-1]

# Process Data
# Generate color gradient
if len(sols) == 1:
    grad_idcs = np.array((0,), dtype=np.int32)
else:
    grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(sols))))


def cols_gradient(n):
    return gradient[grad_idcs[n]]


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
alpha = u_dict['alpha']

r2d = 180 / np.pi
mass = k_dict['mass']
mu = k_dict['mu']
rm = k_dict['rm']
r = rm + h
g = mu / r ** 2
g0 = mu / rm ** 2
weight = mass * g
weight0 = mass * g0
energy = g * h + 0.5 * v ** 2

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

# Aerodynamic Analysis
s_ref = k_dict['s_ref']
rho0 = k_dict['rho0']
h_ref = k_dict['h_ref']
rho = rho0 * np.exp(-h / h_ref)
qdyn = 0.5 * rho * v**2

# # Polynomial CL/CD Model
# cl = CL0 + CL1 * alpha
# cd = CD0 + CD1 * alpha + CD2 * alpha ** 2

# Trigonometric CL/CD Model
cl = CL1 * 0.5 * np.sin(2 * (alpha + CL0/CL1))
cl_min = -CL1 * 0.5
cl_max = CL1 * 0.5
cd = CD0 - CD1**2/(4*CD2) + CD2 * np.sin(alpha + CD1/(2*CD2))**2
cd_min = CD0 - CD1**2/(4*CD2)
cd_max = CD0 - CD1**2/(4*CD2) + CD2

lift = qdyn * s_ref * cl
lift_min = qdyn * s_ref * cl_min
lift_max = qdyn * s_ref * cl_max
lift_g = lift / weight
lift_min_g = lift_min / weight
lift_max_g = lift_max / weight
drag = qdyn * s_ref * cd
drag_min = qdyn * s_ref * cd_min
drag_max = qdyn * s_ref * cd_max
drag_g = drag / weight
drag_min_g = drag_min / weight
drag_max_g = drag_max / weight


alpha_max_ld = - CL0/CL1 + ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5
alpha_min_ld = - CL0/CL1 - ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5

ld_max = (CL0 + CL1 * alpha_max_ld) / (CD0 + CD1 * alpha_max_ld + CD2 * alpha_max_ld ** 2)
ld_min = (CL0 + CL1 * alpha_min_ld) / (CD0 + CD1 * alpha_min_ld + CD2 * alpha_min_ld ** 2)

n_max_g = k_dict['n_max'] * g0 / g
n_min_g = k_dict['n_min'] * g0 / g

lift_max_con_g = np.minimum(lift_max_g, n_max_g)
lift_min_con_g = np.maximum(lift_min_g, n_min_g)
drag_max_con_g = np.minimum(drag_max_g, n_max_g)

lift_frac = (lift_g - 0.5 * (lift_max_con_g + lift_min_con_g)) / (0.5 * (lift_max_con_g - lift_min_con_g))
drag_frac = (drag_g - 0.5 * (drag_max_con_g + drag_min_g)) / (0.5 * (drag_max_con_g - drag_min_g))

t_sweep_min = 0.
t_sweep_max = 0.
lift_sweep_frac: List[np.array] = [None] * len(sols)
drag_sweep_frac: List[np.array] = [None] * len(sols)
ld_sweep: List[np.array] = [None] * len(sols)

if PLOT_SWEEP:
    for idx, sweep_sol in enumerate(sols):
        _x_dict = {}
        for key, x_val, lam_val in zip(sweep_sol.annotations.states, list(sweep_sol.x), list(sweep_sol.lam)):
            _x_dict[key] = x_val
        _h = _x_dict['h_nd'] * k_dict['h_scale']
        _v = _x_dict['v_nd'] * k_dict['v_scale']
        _alpha = sweep_sol.u[0, :]

        _rho = rho0 * np.exp(-_h/h_ref)
        _qdyn = 0.5 * _rho * _v ** 2
        _g = mu / (rm + _h)**2
        _weight = _g * mass

        _s_ref_qdyn_w = s_ref * _qdyn / _weight
        _cl = CL1 * 0.5 * np.sin(2 * (_alpha + CL0/CL1))
        _cd = CD0 - CD1**2/(4*CD2) + CD2 * np.sin(_alpha + CD1/(2*CD2))**2
        _lift_g = _s_ref_qdyn_w * _cl
        _drag_g = _s_ref_qdyn_w * _cd

        _n_max_g = k_dict['n_max'] * g0 / _g
        _n_min_g = k_dict['n_min'] * g0 / _g
        _lift_max_g = _s_ref_qdyn_w * cl_max
        _lift_min_g = _s_ref_qdyn_w * cl_min
        _drag_max_g = _s_ref_qdyn_w * cd_max
        _drag_min_g = _s_ref_qdyn_w * cd_min
        _lift_max_con_g = np.minimum(_lift_max_g, _n_max_g)
        _lift_min_con_g = np.maximum(_lift_min_g, _n_min_g)
        _drag_max_con_g = np.minimum(_drag_max_g, _n_max_g)

        lift_sweep_frac[idx] = (_lift_g - 0.5 * (_lift_max_con_g + _lift_min_con_g)) / (
                    0.5 * (_lift_max_con_g - _lift_min_con_g))
        drag_sweep_frac[idx] = (_drag_g - 0.5 * (_drag_max_con_g + _drag_min_g)) / (
                    0.5 * (_drag_max_con_g - _drag_min_g))
        ld_sweep[idx] = _lift_g / _drag_g

        # Discard data where there is essentiall no lift or drag available
        _invalid_lift = np.where(_lift_max_con_g - _lift_min_con_g < 1e-3)
        _invalid_drag = np.where(_drag_max_con_g - _drag_min_g < 1e-3)
        lift_sweep_frac[idx][_invalid_lift] = np.nan
        drag_sweep_frac[idx][_invalid_drag] = np.nan
        ld_sweep[idx][_invalid_lift] = np.nan
        ld_sweep[idx][_invalid_drag] = np.nan

        if sol.t[0] < t_sweep_min:
            t_sweep_min = sol.t[0]
        if sol.t[-1] > t_sweep_max:
            t_sweep_max = sol.t[-1]

t_sweep_span = np.array((t_sweep_min, t_sweep_max))

# Hamiltonian Analysis
lam_h = lam_dict['h_nd'] / k_dict['h_scale']
lam_v = lam_dict['v_nd'] / k_dict['v_scale']
lam_gam = lam_dict['gam_nd'] / k_dict['gam_scale']

cd_alpha = CD2 * np.sin(2 * (alpha + CD1 / (2 * CD2)))
cl_alpha = CL1 * np.cos(2 * (alpha + CL0 / CL1))

if OPTIMIZATION == 'max_range':
    dcost_dt = (-v * np.cos(gam) / r) / k_dict['theta_scale']
    cost = -(theta[-1] - theta[0]) / k_dict['theta_scale']
    cost_lab = r'$J(\Delta{\theta})$'
    dham_du = -lam_v * qdyn * s_ref * cd_alpha / mass + lam_gam * qdyn * s_ref / (mass * v)
elif OPTIMIZATION == 'min_time':
    dcost_dt = 0*sol.t + 1.
    cost = sol.t[-1] - sol.t[0]
    cost_lab = r'$J(\Delta{t})$'
    dham_du = -lam_v * qdyn * s_ref * cd_alpha / mass + lam_gam * qdyn * s_ref / (mass * v) \
              + 2 * (k_dict['eps_cost_alpha'] / k_dict['alpha_scale'] ** 2) * alpha
elif OPTIMIZATION == 'min_energy':
    dh_dt = v * np.sin(gam)
    dg_dh = -2 * g / r
    dv_dt = -drag/mass - g * np.sin(gam)
    g_scale = mu / (rm + k_dict['h_scale'])**2
    e_scale = g_scale * k_dict['h_scale'] + 0.5 * k_dict['v_scale'] ** 2
    dcost_dt = (dh_dt * (dg_dh * h + g) + v * dv_dt)/e_scale
    cost = (energy[-1] - energy[0])/e_scale
    cost_lab = r'$J(\Delta{E})$'
    dham_du = -lam_v * qdyn * s_ref * cd_alpha / mass + lam_gam * qdyn * s_ref / (mass * v)
else:
    dcost_dt = np.nan * sol.t
    cost = 0.
    cost_lab = '[Invalid Opt. Selected]'
    dham_dt = np.nan * sol.t


def dcost_utm_fun(_x_utm, _x_min_utm, _x_max_utm, _eps_utm):
    return _eps_utm / np.cos(
        np.pi / 2 * (2 * _x_utm - _x_max_utm - _x_min_utm) / (_x_max_utm - _x_min_utm)
    ) - _eps_utm


if OPTIMIZATION == 'min_time':
    dcost_utm = dcost_utm_fun(lift / weight0, k_dict['n_min'], k_dict['n_max'], k_dict['eps_n'])\
                + dcost_utm_fun(drag / weight0, k_dict['n_min'], k_dict['n_max'], k_dict['eps_n'])
    lab_utm = r'$\Delta{J_{n^2}}$'
else:
    dcost_utm = dcost_utm_fun(
        x_dict['h_nd'], k_dict['h_min'] / k_dict['h_scale'], k_dict['h_max'] / k_dict['h_scale'], k_dict['eps_h']
    )
    lab_utm = r'$\Delta{J_{h}}$'

heat_rate_max = k_dict['heat_rate_max']
heat_rate_min = -heat_rate_max
heat_rate = k_dict['k'] * (rho / k_dict['rn']) * v ** 3

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

dh_dt = v * np.sin(gam)
dtheta_dt = v * np.cos(gam) / r
dv_dt = -drag / mass - g * np.sin(gam)
dgam_dt = lift / (mass * v) + (v/r - g/v) * np.cos(gam)

# PLOTS ----------------------------------------------------------------------------------------------------------------
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

    if PLOT_SWEEP:
        for sol_idx, sol_sweep in enumerate(sols):
            ax.plot(sol_sweep.t * t_mult, sol_sweep.x[idx, :] * ymult[idx], color=cols_gradient(sol_idx))
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

for idx, ctrl in enumerate(list(sol.u)):
    axes_u.append(fig_controls.add_subplot(len(u_dict), 1, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    if PLOT_SWEEP:
        for sol_idx, sol_sweep in enumerate(sols):
            ax.plot(sol_sweep.t * t_mult, sol_sweep.u[idx, :] * ymult[idx], color=cols_gradient(sol_idx))
    else:
        ax.plot(sol.t * t_mult, ctrl * ymult[idx])

fig_controls.tight_layout()

if PLOT_COSTATE:
    # PLOT COSTATES
    ylabs = lamlabs
    ymult = lammult
    fig_costates = plt.figure()
    axes_costates = []

    for idx, costate in enumerate(list(sol.lam)):
        axes_costates.append(fig_costates.add_subplot(2, 2, idx + 1))
        ax = axes_costates[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])

        if PLOT_SWEEP:
            for sol_idx, sol_sweep in enumerate(sols):
                ax.plot(sol_sweep.t * t_mult, sol_sweep.lam[idx, :] * ymult[idx], color=cols_gradient(sol_idx))
        else:
            ax.plot(sol.t * t_mult, costate * ymult[idx])

    fig_costates.tight_layout()

if PLOT_AUXILIARY:
    # PLOT AERODYNAMICS
    if PLOT_SWEEP:
        ydata = (lift_sweep_frac, drag_sweep_frac, ld_sweep)
        ylabs = (r'%$L$ [g]', r'%$D$ [g]', r'L/D')
        ymult = (100., 100., 1.)
        ymax = (1., 1., ld_max)
        ymin = (-1., -1., ld_min)

        fig_aero = plt.figure()
        axes_aero = []

        for idx, ylist in enumerate(ydata):
            axes_aero.append(fig_aero.add_subplot(3, 1, idx + 1))
            ax = axes_aero[-1]
            ax.grid()
            ax.set_xlabel(t_label)
            ax.set_ylabel(ylabs[idx])
            for sol_idx, (sweep_sol, y) in enumerate(zip(sols, ylist)):
                ax.plot(sweep_sol.t * t_mult, y * ymult[idx], color=cols_gradient(sol_idx))
            ax.plot(t_sweep_span * t_mult, 0*t_sweep_span + ymax[idx] * ymult[idx], 'k--')
            ax.plot(t_sweep_span * t_mult, 0*t_sweep_span + ymin[idx] * ymult[idx], 'k--')
    else:
        ydata = (lift_g, drag_g, lift/drag)
        ymax = (lift_max_con_g, drag_max_con_g, 0 * sol.t + ld_max)
        ymin = (lift_min_con_g, drag_min_g, 0 * sol.t + ld_min)
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
            ax.plot(sol.t * t_mult, ymin[idx], 'k--')
            ax.plot(sol.t * t_mult, ymax[idx], 'k--')

    # PLOT H-v
    fig_hv = plt.figure()
    ax_hv = fig_hv.add_subplot(111)
    ax_hv.grid()
    ax_hv.set_xlabel(xlabs[2])
    ax_hv.set_ylabel(xlabs[0])

    if PLOT_SWEEP:
        for sol_idx, sol_sweep in enumerate(sols):
            ax_hv.plot(sol_sweep.x[2, :] * xmult[2], sol_sweep.x[0, :] * xmult[0], color=cols_gradient(sol_idx))
    else:
        ax_hv.plot(sol.x[2, :] * xmult[2], sol.x[0, :] * xmult[0])

    ax_hv.plot(v_glide / k_dict['v_scale'] * xmult[2], h_glide / k_dict['h_scale'] * xmult[0],
               'k--', label='Glide Slope')
    ax_hv.legend()

    # PLOT DERIVATIVES
    ydata = (dh_dt, dtheta_dt, dv_dt, dgam_dt)
    ylabs = (r'$dh/dt$ [m/s]', r'$d{\theta}/dt$ [deg/s]', r'$dV/dt$ [m/s$^2$]', r'$d{\gamma}/dt$ [deg/s]')
    ymult = (1., r2d, 1., r2d)
    fig_deriv = plt.figure()
    axes_deriv = []

    for idx, y in enumerate(ydata):
        axes_deriv.append(fig_deriv.add_subplot(2, 2, idx + 1))
        ax = axes_deriv[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t * t_mult, y * ymult[idx])

    fig_deriv.tight_layout()

    # PLOT CONSTRAINTS
    ydata = (heat_rate, qdyn, dham_du)
    if np.max(heat_rate) < 0.1 * heat_rate_max:
        heat_rate_aux = np.nan * sol.t
        heat_rate_aux_lab = None
    else:
        heat_rate_aux = 0 * sol.t + k_dict['heat_rate_max']
        heat_rate_aux_lab = 'Max Heat Rate'
    yaux = (heat_rate_aux, qdyn_glide_interp(h), 0 * sol.t)
    ylabs = (r'Heat Rate [W/m$^2$]', r'$Q_{\infty}$ [N/m$^2$]', r'$H_u$ [1/rad]')
    yauxlabs = (heat_rate_aux_lab, 'Gliding Dynamic Pressure', r'$H_u = 0$')

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
        if yauxlabs[idx] is not None:
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
             dcost_utm))
    ylabs = (cost_lab, lab_utm)
    sup_title = f'J_UTM = {sol.cost}\nJ = {cost} [{abs(cost / sol.cost):.2%} of cost]'

    fig_cost = plt.figure()
    axes_cost = []

    for idx, cost in enumerate(ydata):
        axes_cost.append(fig_cost.add_subplot(2, 1, idx + 1))
        ax = axes_cost[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs[idx])
        ax.plot(sol.t * t_mult, cost)

    fig_cost.suptitle(sup_title)
    fig_cost.tight_layout()

plt.show()
