from typing import Callable

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from airplane2_aero_atm import mu, re, g0, mass, s_ref, CD0_fun, CD1, CD2_fun, CL0, CLa_fun, max_ld_fun, atm,\
    dens_fun, sped_fun, gam_qdyn0

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARE_SWEEP = True
# AOA_LAWS are: {weight, max_ld, approx_costate, energy_climb, lam_h0, interp, 0}
AOA_LAWS = ('fpa_feedback', 'max_ld')
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

g = g0

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


def alpha_fpa_feedback(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _h = _x[0]
    _v = _x[2]
    _gam = _x[3]

    _gam_ss = gam_qdyn0(_h, _v)

    _mach = _v / atm.speed_of_sound(_h)
    _CLa = float(CLa_fun(_mach))
    _CD0 = float(CD0_fun(_mach))
    _CD2 = float(CD2_fun(_mach))
    _max_ld_dict = max_ld_fun(_CLa=_CLa, _CD0=_CD0, _CD2=_CD2)
    _alpha_max_ld = _max_ld_dict['alpha']
    _CD_max_ld = _max_ld_dict['CD']
    _CDuu = 2 * _CD2

    _k_gamma = 2 * (_CD_max_ld / _CDuu)**0.5 / _CLa
    _alpha = _alpha_max_ld + _k_gamma * (_gam_ss - _gam)
    return _alpha


def generate_ctrl_law(_aoa_law, _u_interp=None) -> Callable:
    if _aoa_law == 'max_ld':
        _aoa_ctrl = alpha_max_ld_fun
    elif _aoa_law == 'fpa_feedback':
        _aoa_ctrl = alpha_fpa_feedback
    elif _aoa_law == 'interp':
        def _aoa_ctrl(_t, _x, _p_dict, _k_dict):
            _h = _x[0]
            _v = _x[2]
            _mach = _v / atm.speed_of_sound(_h)
            _cl = _u_interp(_t)
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
    _xd = _x[1]
    _v = _x[2]
    _gam = _x[3]

    _alpha = _u[0]

    _mach = _v / atm.speed_of_sound(_h)
    CLa = float(CLa_fun(_mach))
    CD0 = float(CD0_fun(_mach))
    CD2 = float(CD2_fun(_mach))

    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
    _cl = CL0 + CLa * _alpha
    _cd = CD0 + CD1 * _cl + CD2 * _cl**2
    _lift = _qdyn_s_ref * _cl
    _drag = _qdyn_s_ref * _cd

    _dh = _v * np.sin(_gam)
    _dxd = _v * np.cos(_gam)
    _dv = - _drag / mass - g * np.sin(_gam)
    _dgam = _lift / (mass * _v) - g/_v * np.cos(_gam)

    return np.array((_dh, _dxd, _dv, _dgam))


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
ndmult = np.array((k_dict['h_scale'], k_dict['xd_scale'], k_dict['v_scale'], 1.))

print('____ Evaluation ____')
for idx, sol in enumerate(sols):
    for key, val in zip(sol.annotations.states, list(sol.x)):
        x_dict[key] = val
    e_opt = mu/re - mu/(re + x_dict['h_nd'] * k_dict['h_scale']) + 0.5 * (x_dict['v_nd'] * k_dict['v_scale']) ** 2
    CL_opt = sol.u[0, :]
    u_opt = sp.interpolate.PchipInterpolator(sol.t, CL_opt)

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
    mach_opt = v_opt / np.asarray(sped_fun(h_opt)).flatten()

    x_opt = np.vstack((e_opt, h_opt, gam_opt, mach_opt))

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
        mach = v / np.asarray(sped_fun(h)).flatten()
        e = g*h + 0.5 * v**2
        x = np.vstack((e, h, gam, mach))

        # Calculate Alternate Control
        qdyn_s_ref = 0.5 * np.asarray(dens_fun(h)).flatten() * v**2 * s_ref
        CL = CL0 + np.asarray(CLa_fun(mach)).flatten() * alpha_sol
        lift = qdyn_s_ref * CL
        lift_nd = lift / (g0 * mass)

        ivp_sols_dict[idx][ctrl_law_type] = {
            't': ivp_sol.t,
            'x': x,
            'optimality': ivp_sol.y[1, -1] / (x_dict['xd_nd'][-1] * k_dict['xd_scale']),
            'lift_nd': lift_nd,
            'CL': CL,
            'alpha': alpha_sol,
            'x_opt': x_opt,
            'xdf_opt': x_dict['xd_nd'][-1] * k_dict['xd_scale'],
            'xdf': ivp_sol.y[1, -1],
            'CL_opt': CL_opt
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
ylabs = (r'$h$ [1,000 ft]', r'$\gamma$ [deg]', r'$M$', r'$C_L$')
ymult = np.array((1e-3, r2d, 1., 1.))
xlab = r'$E$ [$10^6$ ft$^2$/m$^2$]'
xmult = 1e-6

plt_order = (0, 2, 1, 3)

figs = []
axes_list = []
for idx, ivp_sol_dict in enumerate(ivp_sols_dict):
    figs.append(plt.figure())
    fig = figs[-1]

    fig_name = 'airplane2_case' + str(idx + 1)

    y_arr_opt = np.vstack((
        ivp_sols_dict[idx][AOA_LAWS[0]]['x_opt'][1:, :],
        ivp_sols_dict[idx][AOA_LAWS[0]]['CL_opt'],
    ))
    ydata_opt = list(y_arr_opt)
    xdata_opt = ivp_sols_dict[idx][AOA_LAWS[0]]['x_opt'][0, :]

    ydata_eval_list = []
    xdata_eval_list = []

    for ctrl_law_type in AOA_LAWS:
        y_arr = np.vstack((
            ivp_sols_dict[idx][ctrl_law_type]['x'][1:, :],
            ivp_sols_dict[idx][ctrl_law_type]['CL'],
        ))
        ydata_eval = list(y_arr)

        xdata_eval = ivp_sols_dict[idx][ctrl_law_type]['x'][0, :]

        ydata_eval_list.append(ydata_eval)
        xdata_eval_list.append(xdata_eval)

    axes_list.append([])
    for jdx, lab in enumerate(ylabs):
        axes_list[idx].append(fig.add_subplot(2, 2, jdx + 1))
        plt_idx = plt_order[jdx]
        ax = axes_list[idx][-1]
        ax.grid()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylabs[plt_idx])
        ax.invert_xaxis()

        ax.plot(xdata_opt * xmult, ydata_opt[plt_idx] * ymult[plt_idx], color=col[0], label=f'Optimal')

        for eval_idx, x in enumerate(xdata_eval_list):
            if ydata_eval_list[eval_idx][plt_idx] is not None:
                ax.plot(
                    x * xmult, ydata_eval_list[eval_idx][plt_idx] * ymult[plt_idx],
                    color=col[1+eval_idx], label=AOA_LAWS[eval_idx]
                )

    fig.tight_layout()
    fig.savefig(
        fig_name + '.eps',
        format='eps',
        bbox_inches='tight'
    )

plt.show()
