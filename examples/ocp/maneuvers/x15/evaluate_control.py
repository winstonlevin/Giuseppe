from typing import Callable, Tuple

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from x15_aero_model import cla_fun, cd0_fun, cdl_fun, s_ref, thrust_max, thrust_frac_min, Isp, qdyn_max
from x15_atmosphere import mu, Re, g0, atm, dens_fun, sped_fun
from glide_slope import get_glide_slope, get_glide_slope_neighboring_feedback

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARISON = 'max_range'
AOA_LAW = 'energy_climb'  # {weight, max_ld, energy_climb, 0}
ROLL_LAW = 'regulator'  # {0, regulator}
THRUST_LAW = '0'  # {0, min, max}

if COMPARISON == 'max_range':
    with open('sol_set_range_sweep_envelope.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[0]

# Create Dicts
k_dict = {}
p_dict = {}
x_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val
for key, val in zip(sol.annotations.parameters, sol.p):
    p_dict[key] = val
for key, val in zip(sol.annotations.states, list(sol.x)):
    x_dict[key] = val
for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val


h_min = 0.
h_max = 130e3
mach_min = 0.25
mach_max = 7.5
h_interp, v_interp, gam_interp, drag_interp = get_glide_slope(
    x_dict['m'][0],
    _h_min=h_min, _h_max=h_max, _mach_min=mach_min, _mach_max=mach_max
)

k_h_interpolant, k_gam_interpolant = get_glide_slope_neighboring_feedback(
    x_dict['m'][0], h_interp
)

# CONTROL LIMITS
alpha_max = 10 * np.pi / 180
alpha_min = -alpha_max
phi_max = np.inf
phi_min = -np.inf
thrust_frac_max = 1.

# STATE LIMITS
gam_max = 85 * np.pi / 180
gam_min = -gam_max

limits_dict = {
    'h_min': h_min, 'h_max': h_max,
    'mach_min': mach_min, 'mach_max': mach_max,
    'gam_min': gam_min, 'gam_max': gam_max,
    'e_min': 0.
}

# CONTROL GAINS
phi_regulation_gain = 1


# ---- DYNAMICS & CONTROL LAWS -----------------------------------------------------------------------------------------
def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


def generate_constant_ctrl(_const: float) -> Callable:
    def _const_control(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
        return _const
    return _const_control


def alpha_max_ld(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _mach = saturate(_x[3] / atm.speed_of_sound(_x[0]), mach_min, mach_max)
    _alpha = float(cd0_fun(_mach) / cdl_fun(_mach)) ** 0.5 * float(cla_fun(_mach))
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def alpha_weight(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _mach = saturate(_x[3] / atm.speed_of_sound(_x[0]), mach_min, mach_max)
    _qdyn = 0.5 * atm.density(_x[0]) * _x[3] ** 2
    _weight = _x[6] * mu / (_x[0] + Re) ** 2
    _alpha = _weight / float(_qdyn * s_ref * cla_fun(_mach))
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def drag_accel(_qdyn, _mach, _weight, _k_dict) -> Tuple[float, float]:
    _ad0 = float(_qdyn * s_ref * cd0_fun(_mach)) / _weight
    _adl = float(cdl_fun(_mach)) * _weight / (_qdyn * s_ref)
    return _ad0, _adl


def alpha_energy_climb(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    # Conditions at current state
    _mach = saturate(_x[3] / atm.speed_of_sound(_x[0]), mach_min, mach_max)
    _qdyn = 0.5 * atm.density(_x[0]) * _x[3] ** 2
    _g = mu / (_x[0] + Re) ** 2
    _weight = _x[6] * _g
    _ad0, _adl = drag_accel(_qdyn, _mach, _weight, _k_dict)
    _cgam = np.cos(_x[4])

    # Conditions at glide slope
    _e = 0.5 * _x[3]**2 + _g * _x[0]
    _h_glide = saturate(h_interp(_e), h_min, h_max)
    _g_glide = mu / (_x[0] + Re) ** 2
    _a_glide = atm.speed_of_sound(_h_glide)
    _v_glide = (max(2 * (_e - _h_glide * _g_glide), _a_glide * mach_min)) ** 0.5
    _mach_glide = saturate(_v_glide / _a_glide, mach_min, mach_max)
    _qdyn_glide = 0.5 * atm.density(_h_glide) * _v_glide ** 2
    _weight_glide = _x[6] * _g_glide
    _ad0_glide, _adl_glide = drag_accel(_qdyn_glide, _mach_glide, _weight_glide, _k_dict)

    # Energy climb to achieve glide slope
    _load_factor0 = _cgam
    # _load_factor0 = 1.

    # # _radicand = (_ad0 + _adl * _cgam ** 2 - (_ad0_glide + _adl_glide) / _cgam) / _adl
    # _radicand = (_ad0 + _adl - (_ad0_glide + _adl_glide)) / _adl
    # _load_factor_h = np.sign(_h_glide - _x[0]) * (max(_radicand, 0)) ** 0.5

    _load_factor_h = (_h_glide - _x[0]) * k_h_interpolant(_e)

    # _load_factor_gam = 0.
    _load_factor_gam = (gam_interp(_e) - _x[4]) * k_gam_interpolant(_e)

    _load_factor = _load_factor0 + _load_factor_h + _load_factor_gam

    # Convert Load Factor to AOA
    _alpha = _weight * _load_factor / float(_qdyn * s_ref * cla_fun(_mach))
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def phi_regulator(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _phi = -phi_regulation_gain * (_x[5])
    _phi = saturate(_phi, phi_min, phi_max)
    return _phi


def generate_ctrl_law() -> Callable:
    if AOA_LAW == 'max_ld':
        _aoa_ctrl = alpha_max_ld
    elif AOA_LAW == 'weight':
        _aoa_ctrl = alpha_weight
    elif AOA_LAW == 'energy_climb':
        _aoa_ctrl = alpha_energy_climb
    else:
        _aoa_ctrl = generate_constant_ctrl(0.)

    if ROLL_LAW == '0':
        _roll_ctrl = generate_constant_ctrl(0.)
    elif ROLL_LAW == 'regulator':
        _roll_ctrl = phi_regulator

    if THRUST_LAW == '0':
        _thrust_ctrl = generate_constant_ctrl(0.)
    elif THRUST_LAW == 'min':
        _thrust_ctrl = generate_constant_ctrl(0.3)
    elif THRUST_LAW == 'max':
        _thrust_ctrl = generate_constant_ctrl(1.)

    def _ctrl_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
        return np.array((
            _aoa_ctrl(_t, _x, _p_dict, _k_dict),
            _roll_ctrl(_t, _x, _p_dict, _k_dict),
            _thrust_ctrl(_t, _x, _p_dict, _k_dict),
        ))
    return _ctrl_law


def eom(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _xn = _x[1]
    _xe = _x[2]
    _v = _x[3]
    _gam = _x[4]
    _psi = _x[5]
    _m = _x[6]

    _alpha = _u[0]
    _phi = _u[1]
    _f_thrust = _u[2]

    _g = mu / (_h + Re)**2
    _mach = _v / atm.speed_of_sound(_h)
    _qdyn = 0.5 * atm.density(_h) * _v**2
    _cl = float(cla_fun(_mach)) * _alpha
    _lift = _qdyn * s_ref * _cl
    _drag = _qdyn * s_ref * float(cd0_fun(_mach) + cdl_fun(_mach) * _cl ** 2)
    _thrust = _f_thrust * thrust_max

    _dh = _v * np.sin(_gam)
    _dxn = _v * np.cos(_gam) * np.cos(_psi)
    _dxe = _v * np.cos(_gam) * np.sin(_psi)

    _dv = (_thrust * np.cos(_alpha) - _drag) / _m - _g * np.sin(_gam)
    _dgam = (_thrust * np.sin(_alpha) + _lift) * np.cos(_phi) / (_m * _v) - _g / _v * np.cos(_gam)
    _dpsi = (_thrust * np.sin(_alpha) + _lift) * np.sin(_phi) / (_m * _v * np.cos(_gam))

    _dm = -_thrust / (Isp * g0)

    return np.array((_dh, _dxn, _dxe, _dv, _dgam, _dpsi, _dm))


def generate_termination_events(_ctrl_law, _p_dict, _k_dict, _limits_dict):
    def min_altitude_event(_t: float, _x: np.array) -> float:
        return _x[0] - _limits_dict['h_min']

    def max_altitude_event(_t: float, _x: np.array) -> float:
        return _limits_dict['h_max'] - _x[0]

    def min_mach_event(_t: float, _x: np.array) -> float:
        _mach = _x[3] / atm.speed_of_sound(_x[0])
        return _mach - _limits_dict['mach_min']

    def max_mach_event(_t: float, _x: np.array) -> float:
        _mach = _x[3] / atm.speed_of_sound(_x[0])
        return _limits_dict['mach_max'] - _mach

    def min_fpa_event(_t: float, _x: np.array) -> float:
        return _x[4] - _limits_dict['gam_min']

    def max_fpa_event(_t: float, _x: np.array) -> float:
        return _limits_dict['gam_max'] - _x[4]

    def min_e_event(_t: float, _x: np.array) -> float:
        _e = mu / (Re + _x[0])**2 * _x[0] + 0.5 * _x[3]**2
        return _e - _limits_dict['e_min']

    events = [min_altitude_event, max_altitude_event,
              min_mach_event, max_mach_event,
              min_fpa_event, max_fpa_event,
              min_e_event]

    for event in events:
        event.terminal = True
        event.direction = 0

    return events


# ---- RUN SIM ---------------------------------------------------------------------------------------------------------

n_sols = len(sols)
ivp_sols_dict = [{}] * n_sols
h0_arr = np.empty((n_sols,))
v0_arr = np.empty((n_sols,))
opt_arr = np.empty((n_sols,))

for idx, sol in enumerate(sols):
    for key, val in zip(sol.annotations.states, list(sol.x)):
        x_dict[key] = val
    e_opt = x_dict['h'] * mu / (Re + x_dict['h']) ** 2 + 0.5 * x_dict['v'] ** 2

    t0 = sol.t[0]
    tf = sol.t[-1]

    t_span = np.array((t0, tf))
    x0 = sol.x[:, 0]

    ctrl_law = generate_ctrl_law()
    limits_dict['e_min'] = np.min(e_opt)
    termination_events = generate_termination_events(ctrl_law, p_dict, k_dict, limits_dict)

    ivp_sol = sp.integrate.solve_ivp(
        fun=lambda t, x: eom(t, x, ctrl_law(t, x, p_dict, k_dict), p_dict, k_dict),
        t_span=t_span,
        y0=x0,
        events=termination_events
    )

    ivp_sols_dict[idx] = {
        't': ivp_sol.t,
        'x': ivp_sol.y,
        'optimality': ivp_sol.y[1, -1] / x_dict['xn'][-1]
    }

    h0_arr[idx] = x_dict['h'][0]
    v0_arr[idx] = x_dict['v'][0]
    opt_arr[idx] = ivp_sols_dict[idx]['optimality']

# Dynamic Pressure Limits
h_qdyn_max = np.linspace(0., 130e3, 1_000)
rho_qdyn_max = np.asarray(dens_fun(h_qdyn_max)).flatten()
v_qdyn_max = (2 * qdyn_max / rho_qdyn_max) ** 0.5

# ---- PLOTTING --------------------------------------------------------------------------------------------------------
gradient = mpl.colormaps['viridis'].colors

if len(sols) == 1:
    grad_idcs = np.array((0,), dtype=np.int32)
else:
    grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(sols))))


def cols_gradient(n):
    return gradient[grad_idcs[n]]


t_label = r'$t$ [s]'
title_str = f'Comparison for {COMPARISON}'

r2d = 180 / np.pi

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$x_N$ [ft]', r'$x_E$ [ft]',
         r'$V$ [ft$^2$/s$^2$]', r'$\gamma$ [deg]', r'$\psi$ [deg]', r'$m$ [lbm]')
ymult = np.array((1., 1., 1., 1., r2d, r2d, g0))
fig_states = plt.figure()
axes_states = []

for idx, lab in enumerate(ylabs):
    axes_states.append(fig_states.add_subplot(2, 4, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    for jdx, (ivp_sol_dict, sol) in enumerate(zip(ivp_sols_dict, sols)):
        # ax.plot(sol.t, sol.x[idx, :] * ymult[idx], 'k--')
        ax.plot(ivp_sol_dict['t'], ivp_sol_dict['x'][idx, :] * ymult[idx], color=cols_gradient(jdx))

fig_states.suptitle(title_str)
fig_states.tight_layout()

fig_hv = plt.figure()
ax_hv = fig_hv.add_subplot(111)
ax_hv.grid()
ax_hv.set_xlabel(ylabs[3])
ax_hv.set_ylabel(ylabs[0])

for jdx, (ivp_sol_dict, sol) in enumerate(zip(ivp_sols_dict, sols)):
    ax_hv.plot(sol.x[3, :] * ymult[3], sol.x[0, :] * ymult[0], 'k--')
    ax_hv.plot(ivp_sol_dict['x'][3, :] * ymult[3], ivp_sol_dict['x'][0, :] * ymult[0], color=cols_gradient(jdx))

ax_hv.plot(v_interp(v_interp.x), h_interp(h_interp.x), 'k', label='Glide Slope', linewidth=2.)
ax_hv.plot(v_qdyn_max, h_qdyn_max, 'k--', label=r'Max $Q_{\infty}$', linewidth=2.)
ax_hv.set_xlim(left=v_interp(v_interp.x[0]), right=v_interp(v_interp.x[-1]))

fig_hv.tight_layout()

# PLOT OPTIMALITY ALONG h-V
if n_sols >= 3:
    fig_optimality = plt.figure()
    ax_optimality = fig_optimality.add_subplot(111)
    ax_optimality.grid()
    ax_optimality.set_xlabel(ylabs[3])
    ax_optimality.set_ylabel(ylabs[0])
    mappable = ax_optimality.tricontourf(v0_arr, h0_arr, 100*opt_arr,
                                         vmin=0., vmax=100., levels=np.arange(0., 101., 1.))
    ax_optimality.plot(v_interp(v_interp.x), h_interp(h_interp.x), 'k', label='Glide Slope', linewidth=2.)
    ax_optimality.plot(v_qdyn_max, h_qdyn_max, 'k--', label=r'Max $Q_{\infty}$', linewidth=2.)
    ax_optimality.legend()
    ax_optimality.set_title(f'Min. Optimality = {np.min(opt_arr):.2%}')

    ax_optimality.set_xlim(left=np.min(v0_arr), right=np.max(v0_arr))
    ax_optimality.set_ylim(bottom=np.min(h0_arr), top=np.max(h0_arr))

    fig_optimality.colorbar(mappable, label='% Optimal')
    fig_optimality.tight_layout()

plt.show()
