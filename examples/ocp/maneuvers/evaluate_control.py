from typing import Callable, Tuple

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, atm, lut_data
# from glide_slope import get_glide_slope

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARISON = 'max_range'
AOA_LAW = 'energy_climb'  # {weight, max_ld, energy_climb, 0}
ROLL_LAW = 'regulator'  # {0, regulator}
THRUST_LAW = '0'  # {0, min, max}

if COMPARISON == 'max_range':
    with open('sol_set_range.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

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


# Generate glide slope interpolant
e_opt = x_dict['h'] * k_dict['mu'] / (k_dict['Re'] + x_dict['h']) ** 2 + 0.5 * x_dict['v'] ** 2
h_opt = x_dict['h']
if e_opt[-1] > e_opt[0]:
    h_interp = sp.interpolate.PchipInterpolator(
        e_opt, h_opt
    )
else:
    h_interp = sp.interpolate.PchipInterpolator(
        np.flip(e_opt), np.flip(h_opt)
    )

# h_min = 0.
# h_max = np.max(lut_data['h'])
# mach_max = np.max(lut_data['M'])
# e_min = 0.
# e_max = h_max * k_dict['mu'] / (k_dict['Re'] + h_max) ** 2 + 0.5 * (mach_max * atm.speed_of_sound(h_max)) ** 2
# e_vals = np.linspace(e_min, e_max, 1_000)
# h_interp, v_interp, gam_interp, drag_interp = get_glide_slope(
#     k_dict['mu'], k_dict['Re'], x_dict['m'][0], k_dict['s_ref'], k_dict['eta'],
#     e_vals, h_min, h_max, mach_max
# )

# CONTROL LIMITS
alpha_max = 10 * np.pi / 180
alpha_min = -alpha_max
phi_max = np.inf
phi_min = -np.inf
thrust_frac_max = 1.
thrust_frac_min = 0.3

# STATE LIMITS
h_min = 0
h_max = max(lut_data['h']) - 1e3
mach_min = 0.1
mach_max = max(lut_data['M']) - 0.1
gam_max = 85 * np.pi / 180
gam_min = -gam_max
e_min = np.min(e_opt)

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
    _alpha = float(cd0_table(_mach) / (_k_dict['eta'] * cl_alpha_table(_mach))) ** 0.5
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def alpha_weight(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _mach = saturate(_x[3] / atm.speed_of_sound(_x[0]), mach_min, mach_max)
    _qdyn = 0.5 * atm.density(_x[0]) * _x[3] ** 2
    _weight = _x[6] * _k_dict['mu'] / (_x[0] + _k_dict['Re']) ** 2
    _alpha = _weight / float(_qdyn * _k_dict['s_ref'] * cl_alpha_table(_mach))
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def drag_accel(_qdyn, _mach, _weight, _k_dict) -> Tuple[float, float]:
    _ad0 = float(_qdyn * _k_dict['s_ref'] * cd0_table(_mach)) / _weight
    _adl = _k_dict['eta'] * _weight / float(_qdyn * _k_dict['s_ref'] * cl_alpha_table(_mach))
    return _ad0, _adl


def alpha_energy_climb(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    # Conditions at current state
    _mach = saturate(_x[3] / atm.speed_of_sound(_x[0]), mach_min, mach_max)
    _qdyn = 0.5 * atm.density(_x[0]) * _x[3] ** 2
    _g = _k_dict['mu'] / (_x[0] + _k_dict['Re']) ** 2
    _weight = _x[6] * _g
    _ad0, _adl = drag_accel(_qdyn, _mach, _weight, _k_dict)

    # Conditions are glide slope
    _e = 0.5 * _x[3]**2 + _g * _x[0]
    _h_glide = saturate(h_interp(_e), h_min, h_max)
    _g_glide = _k_dict['mu'] / (_x[0] + _k_dict['Re']) ** 2
    _a_glide = atm.speed_of_sound(_h_glide)
    _v_glide = (max(2 * (_e - _h_glide * _g_glide), _a_glide * mach_min)) ** 0.5
    _mach_glide = saturate(_v_glide / _a_glide, mach_min, mach_max)
    _qdyn_glide = 0.5 * atm.density(_h_glide) * _v_glide ** 2
    _weight_glide = _x[6] * _g_glide
    _ad0_glide, _adl_glide = drag_accel(_qdyn_glide, _mach_glide, _weight_glide, _k_dict)

    # Energy climb to achieve glide slope
    _cgam = 1.0
    # _cgam = np.cos(_x[4])
    _radicand = (_ad0 + _adl * _cgam**2 - _cgam * (_ad0_glide + _adl_glide)) / _adl
    _load_factor = np.cos(_x[4]) + np.sign(_h_glide - _x[0]) * (max(_radicand, 0)) ** 0.5

    # Convert Load Factor to AOA
    _alpha = _weight * _load_factor / float(_qdyn * _k_dict['s_ref'] * cl_alpha_table(_mach))
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

    _g = _k_dict['mu'] / (_h + _k_dict['Re'])**2
    _mach = _v / atm.speed_of_sound(_h)
    _qdyn = 0.5 * atm.density(_h) * _v**2
    _cl_alpha = float(cl_alpha_table(_mach))
    _lift = _qdyn * _k_dict['s_ref'] * _cl_alpha * _alpha
    _drag = _qdyn * _k_dict['s_ref'] * (float(cd0_table(_mach)) + _k_dict['eta'] * _cl_alpha * _alpha ** 2)
    _thrust = _f_thrust * float(thrust_table((_mach, _h)))

    _dh = _v * np.sin(_gam)
    _dxn = _v * np.cos(_gam) * np.cos(_psi)
    _dxe = _v * np.cos(_gam) * np.sin(_psi)

    _dv = (_thrust * np.cos(_alpha) - _drag) / _m - _g * np.sin(_gam)
    _dgam = (_thrust * np.sin(_alpha) + _lift) * np.cos(_phi) / (_m * _v) - _g / _v * np.cos(_gam)
    _dpsi = (_thrust * np.sin(_alpha) + _lift) * np.sin(_phi) / (_m * _v * np.cos(_gam))

    _dm = -_thrust / (_k_dict['Isp'] * _k_dict['mu'] / _k_dict['Re']**2)

    return np.array((_dh, _dxn, _dxe, _dv, _dgam, _dpsi, _dm))


def generate_termination_events(_ctrl_law, _p_dict, _k_dict):
    def min_altitude_event(_t: float, _x: np.array) -> float:
        return _x[0] - h_min

    def max_altitude_event(_t: float, _x: np.array) -> float:
        return h_max - _x[0]

    def min_mach_event(_t: float, _x: np.array) -> float:
        _mach = _x[3] / atm.speed_of_sound(_x[0])
        return _mach - mach_min

    def max_mach_event(_t: float, _x: np.array) -> float:
        _mach = _x[3] / atm.speed_of_sound(_x[0])
        return mach_max - _mach

    def min_fpa_event(_t: float, _x: np.array) -> float:
        return _x[4] - gam_min

    def max_fpa_event(_t: float, _x: np.array) -> float:
        return gam_max - _x[4]

    def min_e_event(_t: float, _x: np.array) -> float:
        _e = _k_dict['mu'] / (_k_dict['Re'] + _x[0])**2 * _x[0] + 0.5 * _x[3]**2
        return _e - e_min

    events = [min_altitude_event, max_altitude_event,
              min_mach_event, max_mach_event,
              min_fpa_event, max_fpa_event,
              min_e_event]

    for idx, event in enumerate(events):
        event.terminal = True
        event.direction = 0

    return events


# ---- RUN SIM ---------------------------------------------------------------------------------------------------------

t0 = sol.t[0]
tf = sol.t[-1]

t_span = np.array((t0, tf))
x0 = sol.x[:, 0]

ctrl_law = generate_ctrl_law()
termination_events = generate_termination_events(ctrl_law, p_dict, k_dict)

ivp_sol = sp.integrate.solve_ivp(
    fun=lambda t, x: eom(t, x, ctrl_law(t, x, p_dict, k_dict), p_dict, k_dict),
    t_span=t_span,
    y0=x0,
    events=termination_events
)

t = ivp_sol.t
x = ivp_sol.y

# ---- PLOTTING --------------------------------------------------------------------------------------------------------
t_label = r'$t$ [s]'
title_str = f'Range = {100 * x[1, -1] / sol.x[1, -1]:.2f}% Optimal'

r2d = 180 / np.pi
g0 = k_dict['mu'] / k_dict['Re'] ** 2

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$x_N$ [ft]', r'$x_E$ [ft]',
         r'$V$ [ft$^2$/s$^2$]', r'$\gamma$ [deg]', r'$\psi$ [deg]', r'$m$ [lbm]', r'$E$ [ft/s]')
ymult = np.array((1., 1., 1., 1., r2d, r2d, g0, 1.))
fig_states = plt.figure()
axes_states = []

for idx, (state_opt, state) in enumerate(zip(list(sol.x), x)):
    axes_states.append(fig_states.add_subplot(2, 4, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(t, state * ymult[idx], label=AOA_LAW)
    ax.plot(sol.t, state_opt * ymult[idx], 'k--', label='Max Range')

    if idx == 2:
        ax.legend()

fig_states.suptitle(title_str)
fig_states.tight_layout()

plt.show()
