from typing import Callable

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from airplane2_aero_atm import mu, re, g0, mass, s_ref, CD0_fun, CD1, CD2_fun, CL0, CLa_fun, max_ld_fun, atm,\
    dens_fun, sped_fun
from glide_slope import get_glide_slope, get_costate_feedback

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARE_SWEEP = True
# AOA_LAWS are: {weight, max_ld, approx_costate, energy_climb, lam_h0, interp, 0}
AOA_LAWS = ('approx_costate', 'energy_climb', 'max_ld')
PLOT_PAPER = True

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


e_opt = mu/re - mu/(re + x_dict['h_nd'] * k_dict['h_scale']) + 0.5 * (x_dict['v_nd'] * k_dict['v_scale']) ** 2
glide_dict = get_glide_slope(e_vals=np.flip(e_opt), correct_gam=False)
costate_feedback_dict = get_costate_feedback(glide_dict, return_interp=True)

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
gam_max = np.inf * 90 * np.pi / 180
gam_min = -gam_max

limits_dict = {'h_min': h_min, 'v_min': v_min, 'gam_min': gam_min, 'gam_max': gam_max, 'e_min': 0.}

h_e_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['h'])
dh_de_interp = h_e_interp.derivative()
idx_valid_k_h = np.where(np.logical_not(np.isnan(glide_dict['k_h'])))
k_h_interp = sp.interpolate.PchipInterpolator(glide_dict['E'][idx_valid_k_h], glide_dict['k_h'][idx_valid_k_h])
idx_valid_k_gam = np.where(np.logical_not(np.isnan(glide_dict['k_gam'])))
k_gam_interp = sp.interpolate.PchipInterpolator(glide_dict['E'][idx_valid_k_gam], glide_dict['k_gam'][idx_valid_k_gam])


# ---- DYNAMICS & CONTROL LAWS -----------------------------------------------------------------------------------------
def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


def generate_constant_ctrl(_const: float) -> Callable:
    def _const_control(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
        return _const
    return _const_control


def alpha_max_ld_fun(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _h = _x[0]
    _v = _x[2]
    _mach = _v / atm.speed_of_sound(_h)
    _alpha_max_ld = max_ld_fun(
        _CLa=float(CLa_fun(_mach)), _CD0=float(CD0_fun(_mach)), _CD2=float(CD2_fun(_mach))
    )['alpha']
    return _alpha_max_ld


def alpha_energy_climb(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    # Conditions at current state
    _h = _x[0]
    _v = _x[2]
    _gam = _x[3]
    _e = mu/re - mu/(re + _h) + 0.5 * _v**2
    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
    _mach = _v / atm.speed_of_sound(_h)

    # Conditions at glide slope
    _h_glide = h_e_interp(_e)
    _dh_de_glide = dh_de_interp(_e)
    _k_h = k_h_interp(_e)
    _k_gam = k_gam_interp(_e)

    _r_glide = re + _h_glide
    _g_glide = mu / _r_glide ** 2
    _v_glide = saturate(2 * (_e + mu/_r_glide - mu/re), 0., np.inf)**0.5

    _qdyn_s_ref_glide = max(0.5 * atm.density(_h_glide) * _v_glide**2 * s_ref, 1.)
    _mach_glide = _v_glide / atm.speed_of_sound(_h_glide)
    _lift_glide = mass * (_g_glide - _v_glide ** 2 / _r_glide)
    _CD0_glide = float(CD0_fun(_mach_glide))
    _CD2_glide = float(CD2_fun(_mach_glide))
    _drag_glide = _qdyn_s_ref_glide * _CD0_glide + CD1 * _lift_glide + _CD2_glide/_qdyn_s_ref_glide * _lift_glide**2
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
    _CLa = float(CLa_fun(_mach))
    _alpha = (_cl - CL0) / _CLa
    _alpha = saturate(_alpha, alpha_min, alpha_max)
    return _alpha


def alpha_approx_costate(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    # Conditions at current state
    _h = _x[0]
    _v = _x[2]
    _gam = _x[3]
    _e = mu/re - mu/(re + _h) + 0.5 * _v**2
    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
    _mach = _v / atm.speed_of_sound(_h)
    _CLa = float(CLa_fun(_mach))
    _CD0 = float(CD0_fun(_mach))
    _CD2 = float(CD2_fun(_mach))

    # Conditions at glide slope
    _h_glide = h_e_interp(_e)
    _dh_de_glide = dh_de_interp(_e)
    _k_h = k_h_interp(_e)
    _k_gam = k_gam_interp(_e)

    _r_glide = re + _h_glide
    _g_glide = mu / _r_glide ** 2
    _v_glide = saturate(2 * (_e + mu/_r_glide - mu/re), 0., np.inf)**0.5

    _qdyn_s_ref_glide = max(0.5 * atm.density(_h_glide) * _v_glide**2 * s_ref, 1.)
    _mach_glide = _v_glide / atm.speed_of_sound(_h_glide)
    _lift_glide = mass * (_g_glide - _v_glide ** 2 / _r_glide)
    _CL_glide = _lift_glide / _qdyn_s_ref_glide
    _CD0_glide = float(CD0_fun(_mach_glide))
    _CD2_glide = float(CD2_fun(_mach_glide))
    _drag_glide = _qdyn_s_ref_glide * _CD0_glide + CD1 * _lift_glide + _CD2_glide/_qdyn_s_ref_glide * _lift_glide**2
    # _gam_glide = np.arcsin(saturate(-_drag_glide / mass * _dh_de_glide, -1., 1.))
    _gam_glide = 0.
    # _CLa_glide = float(CLa_fun(_mach_glide))
    # _alpha_glide = (_CL_glide - CL0) / _CLa_glide
    #
    # _k_h = costate_feedback_dict['k_h'](_e)
    # _k_v = costate_feedback_dict['k_v'](_e)
    # _k_gam = costate_feedback_dict['k_gam'](_e)
    # _d_alpha = _k_h * (_h_glide - _h) + _k_v * (_v_glide - _v) + _k_gam * (_gam_glide - _gam)
    # _alpha = _alpha_glide + _d_alpha

    _lam_h_glide = -mass * _g_glide / (_drag_glide * _r_glide)
    _lam_v_glide = - mass * _v_glide / (_drag_glide * _r_glide)
    _lam_gam_glide = - mass * _v_glide**2 / (_drag_glide * _r_glide) * (CD1 + 2 * _CD2_glide * _CL_glide)
    _lam_glide = np.vstack((_lam_h_glide, _lam_v_glide, _lam_gam_glide))

    _dx = np.vstack((_h - _h_glide, _v - _v_glide, _gam - _gam_glide))
    _p = costate_feedback_dict['P'](_e)
    _dlam = _p @ _dx
    _dlam_v = _dlam[1, 0]
    _dlam_gam = _dlam[2, 0]

    # Ensure minimium (lam_v < 0 -> dlam_v < -lam_v_glide)
    dlam_v_max = 0.99 * -_lam_v_glide
    if _dlam_v > dlam_v_max:
        _mult = dlam_v_max / _dlam_v
        _dlam_v = dlam_v_max
        _dlam_gam = _dlam_gam * _mult

    # Optimal Control Law Hu = 0
    _lam_v = _lam_v_glide + _dlam[1, 0]
    _lam_gam = _lam_gam_glide + _dlam[2, 0]

    _CL = (_lam_gam/(_v * _lam_v) - CD1) / (2 * _CD2)
    #
    # if _CL < 0:
    #     print('CL < 0')

    _alpha = (_CL - CL0) / _CLa
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
    _mach = _v / atm.speed_of_sound(_h)

    _h_glide = h_e_interp(_e)
    _dh_de_glide = dh_de_interp(_e)

    _r_glide = re + _h_glide
    _g_glide = mu / _r_glide ** 2
    _v_glide = saturate(2 * (_e + mu / _r_glide - mu / re), 0., np.inf) ** 0.5

    _qdyn_s_ref_glide = max(0.5 * atm.density(_h_glide) * _v_glide ** 2 * s_ref, 1.)
    _mach_glide = _v_glide / atm.speed_of_sound(_h_glide)
    _lift_glide = mass * (_g_glide - _v_glide ** 2 / _r_glide)
    _CD0_glide = float(CD0_fun(_mach_glide))
    _CD2_glide = float(CD2_fun(_mach_glide))
    _drag_glide = _qdyn_s_ref_glide * _CD0_glide + CD1 * _lift_glide + _CD2_glide / _qdyn_s_ref_glide * _lift_glide ** 2
    _gam_glide = np.arcsin(saturate(-_drag_glide / mass * _dh_de_glide, -1., 1.))

    _CLa = float(CLa_fun(_mach))
    _CD0 = float(CD0_fun(_mach))
    _CD2 = float(CD2_fun(_mach))
    _beta = _lift0**2 + _qdyn_s_ref/_CD2 * (-_r_glide/_r * _drag_glide * _cgam + CD1 * _lift0 + _qdyn_s_ref * _CD0)
    _lift = _lift0 + np.sign(_h_glide - _h) * max(0., _beta)**0.5

    # _beta = (_lift0 + 2 * _r_glide/_r * _drag_glide * _tgam)**2 \
    #     + _qdyn_s_ref/CD2 * (-_r_glide/_r * _drag_glide * _cgam
    #                          + CD1 * (_lift0 - 2 * _r_glide/_r * _drag_glide * _tgam)
    #                          + _qdyn_s_ref * CD0)
    # _lift = _lift0 + 2 * _r_glide/_r * _drag_glide * _tgam + np.sign(_h_glide - _h) * max(0., _beta)**0.5

    _cl = _lift / _qdyn_s_ref
    _alpha = (_cl - CL0) / _CLa
    _alpha = saturate(_alpha, alpha_min, alpha_max)

    return _alpha


def generate_ctrl_law(_aoa_law, _u_interp=None) -> Callable:
    if _aoa_law == 'max_ld':
        _aoa_ctrl = alpha_max_ld_fun
    elif _aoa_law == 'energy_climb':
        _aoa_ctrl = alpha_energy_climb
    elif _aoa_law == 'approx_costate':
        _aoa_ctrl = alpha_approx_costate
    elif _aoa_law == 'lam_h0':
        _aoa_ctrl = alpha_lam_h0
    elif _aoa_law == 'interp':
        def _aoa_ctrl(_t, _x, _p_dict, _k_dict):
            _h = _x[0]
            _v = _x[2]
            _qdyn_s_ref = 0.5 * atm.density(_h) * _v ** 2 * s_ref
            _mach = _v / atm.speed_of_sound(_h)
            _load = _u_interp(_t)
            _cl = _load * g0 * mass / _qdyn_s_ref
            _alpha = (_cl - CL0) / float(CLa_fun(_mach))
            # _alpha = saturate(_alpha, alpha_min, alpha_max)
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

    _mach = _v / atm.speed_of_sound(_h)
    CLa = float(CLa_fun(_mach))
    CD0 = float(CD0_fun(_mach))
    CD2 = float(CD2_fun(_mach))

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
opt_arr = [{}] * n_sols
ndmult = np.array((k_dict['h_scale'], 1., k_dict['v_scale'], 1.))

print('____ Evaluation ____')
for idx, sol in enumerate(sols):
    for key, val in zip(sol.annotations.states, list(sol.x)):
        x_dict[key] = val
    e_opt = mu/re - mu/(re + x_dict['h_nd'] * k_dict['h_scale']) + 0.5 * (x_dict['v_nd'] * k_dict['v_scale']) ** 2
    lift_nd_opt = sol.u[0, :]
    u_opt = sp.interpolate.PchipInterpolator(sol.t, lift_nd_opt)

    t0 = sol.t[0]
    tf = sol.t[-1]

    t_span = np.array((t0, np.inf))
    x0 = sol.x[:, 0] * ndmult
    limits_dict['e_min'] = np.min(e_opt)

    # States
    h_opt = sol.x[0, :] * k_dict['h_scale']
    gam_opt = sol.x[3, :]
    v_opt = sol.x[2, :] * k_dict['v_scale']
    g_opt = mu / (re + h_opt)**2

    x_opt = np.vstack((e_opt, h_opt, gam_opt))

    # Convert Costates to E-h-gam
    # H = lam_h d(h)/dt + lam_V d(V)/dt + lam_gam d(gam)/dt
    # d(E)/dt = g d(h)/dt + V d(V)/dt
    # ->
    # lam_E    = lam_V / V
    # lam_h'   = lam_h - (g/V) lam_V
    # lam_gam' = lam_gam
    lam_E_opt = (sol.lam[2, :] / k_dict['v_scale']) / v_opt
    lam_h_opt = (sol.lam[0, :] / k_dict['h_scale']) - g_opt * lam_E_opt
    lam_gam_opt = sol.lam[3, :]
    lam_opt = np.vstack((lam_E_opt, lam_h_opt, lam_gam_opt))

    # Glide slope values
    x_glide = np.vstack((glide_dict['E'], glide_dict['h'], glide_dict['gam']))
    lam_glide = np.vstack((glide_dict['lam_E'], glide_dict['lam_h'], glide_dict['lam_gam']))
    lift_nd_glide = glide_dict['u']

    h0_arr[idx] = x_dict['h_nd'][0] * k_dict['h_scale']
    v0_arr[idx] = x_dict['v_nd'][0] * k_dict['v_scale']

    ivp_sols_dict[idx] = {}
    for ctrl_law_type in AOA_LAWS:
        ctrl_law = generate_ctrl_law(ctrl_law_type, u_opt)
        termination_events = generate_termination_events(ctrl_law, p_dict, k_dict, limits_dict)

        ivp_sol = sp.integrate.solve_ivp(
            fun=lambda t, x: eom(t, x, ctrl_law(t, x, p_dict, k_dict), p_dict, k_dict),
            t_span=t_span,
            y0=x0,
            events=termination_events,
            rtol=1e-6, atol=1e-10
        )

        # Calculate Control
        alpha_sol = np.empty(shape=(ivp_sol.t.shape[0],))
        for jdx, (t, x) in enumerate(zip(ivp_sol.t, ivp_sol.y.T)):
            alpha_sol[jdx] = ctrl_law(t, x, p_dict, k_dict)

        # Wrap FPA
        _fpa_unwrapped = ivp_sol.y[3, :]
        _fpa = (_fpa_unwrapped + np.pi) % (2 * np.pi) - np.pi
        ivp_sol.y[3, :] = _fpa

        # Calculate State
        h = ivp_sol.y[0, :]
        v = ivp_sol.y[2, :]
        gam = ivp_sol.y[3, :]
        r = re + h
        g = mu / r ** 2
        e = mu / re - mu / r + 0.5 * v**2
        x = np.vstack((e, h, gam))

        # Calculate Alternate Control
        qdyn_s_ref = 0.5 * np.asarray(dens_fun(h)).flatten() * v**2 * s_ref
        mach = v / np.asarray(sped_fun(h)).flatten()

        CL = CL0 + np.asarray(CLa_fun(mach)).flatten() * alpha_sol
        lift = qdyn_s_ref * CL
        lift_nd = lift / (g0 * mass)

        ivp_sols_dict[idx][ctrl_law_type] = {
            't': ivp_sol.t,
            'x': x,
            'optimality': ivp_sol.y[1, -1] / x_dict['tha'][-1],
            'lift_nd': lift_nd,
            'alpha': alpha_sol,
            'lam_opt': lam_opt,
            'x_opt': x_opt,
            'lift_nd_opt': lift_nd_opt,
            'x_glide': x_glide,
            'lam_glide': lam_glide,
            'lift_nd_glide': lift_nd_glide,
            'thaf_opt': x_dict['tha'][-1],
            'thaf': ivp_sol.y[1, -1]
        }
        opt_arr[idx][ctrl_law_type] = ivp_sols_dict[idx][ctrl_law_type]['optimality']
        print(ctrl_law_type + f' is {opt_arr[idx][ctrl_law_type]:.2%} Optimal at h0 = {h0_arr[idx] / 1e3:.4} kft')

# ---- PLOTTING --------------------------------------------------------------------------------------------------------
col = plt.rcParams['axes.prop_cycle'].by_key()['color']
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


r2d = 180 / np.pi

# PLOT STATES
ylabs = (r'$h$ [kft]', r'$\gamma$ [deg]',
         r'$\lambda_E$ [s$^2$/kft$^2$]', r'$\lambda_h$ [1/kft]', r'$\lambda_\gamma$ [-]',
         r'$n$ [g]')
ymult = np.array((1e-3, r2d, 1e6, 1e3, 1., 1.))
xlab = r'$E$ [kft$^2$/m$^2$]'
xmult = 1e-6

plt_order = (5, 2, 0, 3, 1, 4)

figs = []
axes_list = []
for idx, ivp_sol_dict in enumerate(ivp_sols_dict):
    figs.append(plt.figure())
    fig = figs[-1]

    fig_name = 'airplane2_case' + str(idx + 1)

    y_arr_opt = np.vstack((
        ivp_sols_dict[idx][AOA_LAWS[0]]['x_opt'][1:, :],
        ivp_sols_dict[idx][AOA_LAWS[0]]['lam_opt'],
        ivp_sols_dict[idx][AOA_LAWS[0]]['lift_nd_opt'],
    ))
    ydata_opt = list(y_arr_opt)
    xdata_opt = ivp_sols_dict[idx][AOA_LAWS[0]]['x_opt'][0, :]

    y_arr_glide = np.vstack((
        ivp_sols_dict[idx][AOA_LAWS[0]]['x_glide'][1:, :],
        ivp_sols_dict[idx][AOA_LAWS[0]]['lam_glide'],
        ivp_sols_dict[idx][AOA_LAWS[0]]['lift_nd_glide'],
    ))
    ydata_glide = list(y_arr_glide)
    xdata_glide = ivp_sols_dict[idx][AOA_LAWS[0]]['x_glide'][0, :]

    ydata_eval_list = []
    xdata_eval_list = []

    for ctrl_law_type in AOA_LAWS:
        y_arr = np.vstack((
            ivp_sols_dict[idx][ctrl_law_type]['x'][1:, :],
            ivp_sols_dict[idx][ctrl_law_type]['lift_nd'],
        ))
        ydata_eval = list(y_arr)
        for jdx in (2, 3, 4):
            ydata_eval.insert(jdx, None)

        xdata_eval = ivp_sols_dict[idx][ctrl_law_type]['x'][0, :]

        ydata_eval_list.append(ydata_eval)
        xdata_eval_list.append(xdata_eval)

    axes_list.append([])
    for jdx, lab in enumerate(ylabs):
        axes_list[idx].append(fig.add_subplot(3, 2, jdx + 1))
        plt_idx = plt_order[jdx]
        ax = axes_list[idx][-1]
        ax.grid()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylabs[plt_idx])
        ax.invert_xaxis()

        ax.plot(xdata_glide * xmult, ydata_glide[plt_idx] * ymult[plt_idx], 'k-', label='Glide Slope')
        ax.plot(xdata_opt * xmult, ydata_opt[plt_idx] * ymult[plt_idx], color=col[0], label=f'Optimal')

        for eval_idx, x in enumerate(xdata_eval_list):
            if ydata_eval_list[eval_idx][plt_idx] is not None:
                ax.plot(
                    x * xmult, ydata_eval_list[eval_idx][plt_idx] * ymult[plt_idx],
                    color=col[1+eval_idx], label=AOA_LAWS[eval_idx]
                )

        # if jdx == 0:
        #     ax.legend()

    fig.tight_layout()
    fig.savefig(
        fig_name + '.eps',
        format='eps',
        bbox_inches='tight'
    )

plt.show()
