from typing import Callable

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from airplane2_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, CL0, CLa, alpha_max_ld,\
    CL_max_ld, CD_max_ld, dens_fun, atm
from glide_slope import get_glide_slope

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARE_SWEEP = True
AOA_LAW = 'lam_h0'  # {weight, max_ld, energy_climb, lam_h0, interp, 0}

if COMPARE_SWEEP:
    with open('sol_set_range_sweep.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[0]
else:
    with open('sol_set_range.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]
        sols = [sol]

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


glide_dict = get_glide_slope()

# CONTROL LIMITS
alpha_max = 35 * np.pi / 180
alpha_min = -alpha_max
load_max = 3.
load_min = -3.
phi_max = np.inf
phi_min = -np.inf
thrust_frac_max = 1.

# STATE LIMITS
v_min = 10.
h_min = 0.
gam_max = 90 * np.pi / 180
gam_min = -gam_max

limits_dict = {'h_min': h_min, 'v_min': v_min, 'gam_min': gam_min, 'gam_max': gam_max, 'e_min': 0.}

h_e_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['h'])
dh_de_interp = h_e_interp.derivative()
k_h_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['k_h'])
k_gam_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['k_gam'])


# ---- DYNAMICS & CONTROL LAWS -----------------------------------------------------------------------------------------
def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


def generate_constant_ctrl(_const: float) -> Callable:
    def _const_control(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
        return _const
    return _const_control


def alpha_max_ld_fun(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    return alpha_max_ld


def alpha_energy_climb(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    # Conditions at current state
    _h = _x[0]
    _v = _x[2]
    _gam = _x[3]
    _e = mu/re - mu/(re + _h) + 0.5 * _v**2
    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref

    # Conditions at glide slope
    _h_glide = h_e_interp(_e)
    _dh_de_glide = dh_de_interp(_e)
    _k_h = k_h_interp(_e)
    _k_gam = k_gam_interp(_e)

    _r_glide = re + _h_glide
    _g_glide = mu / _r_glide ** 2
    _v_glide = saturate(2 * (_e + mu/_r_glide - mu/re), 0., np.inf)**0.5
    _qdyn_s_ref_glide = max(0.5 * atm.density(_h_glide) * _v_glide**2 * s_ref, 1.)
    _lift_glide = mass * (_g_glide - _v_glide ** 2 / _r_glide)
    _drag_glide = _qdyn_s_ref_glide * CD0 + CD1 * _lift_glide + CD2/_qdyn_s_ref_glide * _lift_glide**2
    _gam_glide = np.arcsin(saturate(-_drag_glide / mass * _dh_de_glide, -1., 1.))

    _r = re + _h
    _g = mu/_r**2

    _load_ff = _lift_glide / (g0 * mass)
    # _load_ff = CL_max_ld * _qdyn_s_ref / (g0 * mass)

    # _load = _load_ff
    # _load = _load_ff + _k_gam * (_gam_glide - _gam)
    _load = _load_ff + _k_h * (_h_glide - _h) + _k_gam * (_gam_glide - _gam)
    # _load = saturate(_load, load_min, load_max)
    _lift = _load * g0 * mass
    _cl = _lift / _qdyn_s_ref
    _alpha = (_cl - CL0) / CLa
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def alpha_lam_h0(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    # Assuming lam_h = 0, this control law ensures H = 0. Note that two values of alpha satisfy this, so the value
    # which drives h -> h_glide is chosen.
    # TODO - This causes chatter when near h_glide. _c -/-> 0!
    _h = _x[0]
    _v = _x[2]
    _gam = _x[3]
    _r = _h + re
    _g = mu/_r**2
    _e = mu/re - mu/_r + 0.5 * _v**2
    _cgam = np.cos(_gam)
    _tgam = np.tan(_gam)
    _lift0 = mass * (_g - _v**2/_r) * _cgam
    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref

    _h_glide = h_e_interp(_e)
    _r_glide = re + _h_glide
    _g_glide = mu/_r_glide
    _v_glide = saturate(2 * (_e + mu / _r_glide - mu / re), 0., np.inf) ** 0.5
    _qdyn_s_ref_glide = max(0.5 * atm.density(_h_glide) * _v_glide ** 2 * s_ref, 1.)
    _lift_glide = mass * (_g_glide - _v_glide ** 2 / _r_glide)
    _drag_glide = _qdyn_s_ref_glide * CD0 + CD1 * _lift_glide + CD2 / _qdyn_s_ref_glide * _lift_glide ** 2

    _beta = _lift0**2 + _qdyn_s_ref/CD2 * (-_r_glide/_r * _drag_glide * _cgam + CD1 * _lift0 + _qdyn_s_ref * CD0)
    _lift = _lift0 + np.sign(_h_glide - _h) * max(0., _beta)**0.5

    # _beta = (_lift0 + 2 * _r_glide/_r * _drag_glide * _tgam)**2 \
    #     + _qdyn_s_ref/CD2 * (-_r_glide/_r * _drag_glide * _cgam
    #                          + CD1 * (_lift0 - 2 * _r_glide/_r * _drag_glide * _tgam)
    #                          + _qdyn_s_ref * CD0)
    # _lift = _lift0 + 2 * _r_glide/_r * _drag_glide * _tgam + np.sign(_h_glide - _h) * max(0., _beta)**0.5

    _cl = _lift / _qdyn_s_ref
    _alpha = (_cl - CL0) / CLa
    _alpha = saturate(_alpha, alpha_min, alpha_max)

    return _alpha


def generate_ctrl_law(_u_interp=None) -> Callable:
    if AOA_LAW == 'max_ld':
        _aoa_ctrl = alpha_max_ld_fun
    elif AOA_LAW == 'energy_climb':
        _aoa_ctrl = alpha_energy_climb
    elif AOA_LAW == 'lam_h0':
        _aoa_ctrl = alpha_lam_h0
    elif AOA_LAW == 'interp':
        def _aoa_ctrl(_t, _x, _p_dict, _k_dict):
            _h = _x[0]
            _v = _x[2]
            _qdyn_s_ref = 0.5 * atm.density(_h) * _v ** 2 * s_ref
            _load = _u_interp(_t)
            _cl = _load * g0 * mass / _qdyn_s_ref
            _alpha = (_cl - CL0) / CLa
            _alpha = saturate(_alpha, alpha_min, alpha_max)
            return _alpha
    else:
        _aoa_ctrl = generate_constant_ctrl(0.)

    def _ctrl_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
        return np.array((
            _aoa_ctrl(_t, _x, _p_dict, _k_dict),
        ))
    return _ctrl_law


def eom(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _tha = _x[1]
    _v = _x[2]
    _gam = _x[3]

    _alpha = _u[0]

    _r = _h + re
    _g = mu / _r**2
    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
    _cl = CL0 + CLa * _alpha
    _cd = CD0 + CD1 * _cl + CD2 * _cl**2
    _lift = _qdyn_s_ref * _cl
    _drag = _qdyn_s_ref * _cd

    _dh = _v * np.sin(_gam)
    _dtha = _v/_r * np.cos(_gam)
    _dv = - _drag / mass - _g * np.sin(_gam)
    _dgam = _lift / (mass * _v) + (_v/_r - _g/_v) * np.cos(_gam)

    return np.array((_dh, _dtha, _dv, _dgam))


def generate_termination_events(_ctrl_law, _p_dict, _k_dict, _limits_dict):
    def min_altitude_event(_t: float, _x: np.array) -> float:
        return _x[0] - _limits_dict['h_min']

    def min_velocity_event(_t: float, _x: np.array) -> float:
        return _x[2] - _limits_dict['v_min']

    def min_fpa_event(_t: float, _x: np.array) -> float:
        return _x[3] - _limits_dict['gam_min']

    def max_fpa_event(_t: float, _x: np.array) -> float:
        return _limits_dict['gam_max'] - _x[3]

    def min_e_event(_t: float, _x: np.array) -> float:
        _e = mu/re - mu/(re + _x[0]) + 0.5 * _x[2]**2
        return _e - _limits_dict['e_min']

    events = [min_altitude_event, min_velocity_event,
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
ndmult = np.array((k_dict['h_scale'], 1., k_dict['v_scale'], 1.))

print('____ Evaluation ____')
for idx, sol in enumerate(sols):
    for key, val in zip(sol.annotations.states, list(sol.x)):
        x_dict[key] = val
    e_opt = mu/re - mu/(re + x_dict['h_nd'] * k_dict['h_scale']) + 0.5 * (x_dict['v_nd'] * k_dict['v_scale']) ** 2
    u_opt = sp.interpolate.PchipInterpolator(sol.t, sol.u[0, :])

    t0 = sol.t[0]
    tf = sol.t[-1]

    t_span = np.array((t0, np.inf))
    x0 = sol.x[:, 0] * ndmult

    ctrl_law = generate_ctrl_law(u_opt)
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
        'optimality': ivp_sol.y[1, -1] / x_dict['tha'][-1]
    }

    h0_arr[idx] = x_dict['h_nd'][0] * k_dict['h_scale']
    v0_arr[idx] = x_dict['v_nd'][0] * k_dict['v_scale']
    opt_arr[idx] = ivp_sols_dict[idx]['optimality']

    print(f'{opt_arr[idx]:.2%} Optimal at h0 = {h0_arr[idx] / 1e3:.4} kft')

# ---- PLOTTING --------------------------------------------------------------------------------------------------------
gradient = mpl.colormaps['viridis'].colors
# if len(sols) == 1:
#     grad_idcs = np.array((0,), dtype=np.int32)
# else:
#     grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(sols))))
gradient_arr = np.array(gradient).T
idces = np.arange(0, gradient_arr.shape[1], 1)
col0_interp = sp.interpolate.PchipInterpolator(idces, gradient_arr[0, :])
col1_interp = sp.interpolate.PchipInterpolator(idces, gradient_arr[1, :])
col2_interp = sp.interpolate.PchipInterpolator(idces, gradient_arr[2, :])
val_max = 1.


def cols_gradient(n):
    _col1 = float(col0_interp(n/val_max * idces[-1]))
    _col2 = float(col1_interp(n/val_max * idces[-1]))
    _col3 = float(col2_interp(n/val_max * idces[-1]))
    return [_col1, _col2, _col3]


t_label = r'$t$ [s]'
title_str = f'Maximum Range Comparison'

r2d = 180 / np.pi

# PLOT STATES
ylabs = (r'$h$ [ft]', r'$\theta$ [deg]', r'$V$ [ft/s]', r'$\gamma$ [deg]')
ymult = np.array((1., r2d, 1., r2d))
fig_states = plt.figure()
axes_states = []

for idx, lab in enumerate(ylabs):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    for jdx, (ivp_sol_dict, sol) in enumerate(zip(ivp_sols_dict, sols)):
        # ax.plot(sol.t, sol.x[idx, :] * ndmult[idx] * ymult[idx], 'k--')
        ax.plot(ivp_sol_dict['t'], ivp_sol_dict['x'][idx, :] * ymult[idx], color=cols_gradient(min(1., opt_arr[jdx])))

fig_states.suptitle(title_str)
fig_states.tight_layout()

fig_hv = plt.figure()
ax_hv = fig_hv.add_subplot(111)
ax_hv.grid()
ax_hv.set_xlabel(ylabs[2])
ax_hv.set_ylabel(ylabs[0])

for jdx, (ivp_sol_dict, sol) in enumerate(zip(ivp_sols_dict, sols)):
    ax_hv.plot(sol.x[2, :] * ndmult[2] * ymult[2], sol.x[0, :] * ndmult[2] * ymult[0], 'k--')
    ax_hv.plot(ivp_sol_dict['x'][2, :] * ymult[2], ivp_sol_dict['x'][0, :] * ymult[0],
               color=cols_gradient(min(opt_arr[jdx], 1.)))

ax_hv.plot(glide_dict['v'], glide_dict['h'], 'k', label='Glide Slope', linewidth=2.)
ax_hv.set_xlim(left=glide_dict['v'][0], right=glide_dict['v'][-1])

fig_hv.tight_layout()

# PLOT OPTIMALITY ALONG h-V
if n_sols >= 3:
    fig_optimality = plt.figure()
    ax_optimality = fig_optimality.add_subplot(111)
    ax_optimality.grid()
    ax_optimality.set_xlabel(ylabs[2])
    ax_optimality.set_ylabel(ylabs[0])
    mappable = ax_optimality.tricontourf(v0_arr, h0_arr, 100*opt_arr,
                                         vmin=0., vmax=100., levels=np.arange(0., 101., 1.))
    ax_optimality.plot(glide_dict['v'], glide_dict['h'], 'k', label='Glide Slope', linewidth=2.)
    ax_optimality.legend()
    ax_optimality.set_title(f'Min. Optimality = {np.min(opt_arr):.2%}')

    ax_optimality.set_xlim(left=np.min(v0_arr), right=np.max(v0_arr))
    ax_optimality.set_ylim(bottom=np.min(h0_arr), top=np.max(h0_arr))

    fig_optimality.colorbar(mappable, label='% Optimal')
    fig_optimality.tight_layout()

plt.show()
