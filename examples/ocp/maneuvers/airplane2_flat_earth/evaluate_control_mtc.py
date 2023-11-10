from typing import Callable

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from airplane2_aero_atm import g, s_ref, CD0_fun, CD1, CD2_fun, CL0, CLa_fun, max_ld_fun, atm,\
    dens_fun, sped_fun, gam_qdyn0, thrust_fun, Isp
from ardema_mae import find_climb_path, obj_fun_es, zero_fun_es, grad_fun_es, G_es_fun, GE_es_fun, newton_search_x_hd

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARE_SWEEP = True
# CTRL_LAWS are: {interp, aenoc, d2hdt2}
CTRL_LAWS = ('aenoc',)

with open('sol_set_mtc_xf.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[0]
    sols = [sol, sols[-1]]

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

# CONTROL LIMITS
alpha_max = 35 * np.pi / 180
alpha_min = -alpha_max
load_max = 6.
load_min = -6.
phi_max = np.inf
phi_min = -np.inf
thrust_frac_max = 1.

# STATE LIMITS
v_min = 10.
h_min = 0.
gam_max = np.inf * 90 * np.pi / 180
gam_min = -gam_max

limits_dict = {'h_min': h_min, 'v_min': v_min, 'gam_min': gam_min, 'gam_max': gam_max, 'e_max': 0.}

# Energy State Dict (for control law)
h_vals = sol.x[0, :] * k_dict['h_scale']
v_vals = sol.x[1, :] * k_dict['v_scale']
mass_vals = sol.x[3, :] * k_dict['m_scale']
mass0 = mass_vals[0]
e_vals = g * h_vals + 0.5 * v_vals**2
h_es_vals = np.nan * np.empty(e_vals.shape)
gam_es_vals = np.nan * np.empty(e_vals.shape)
lift_es_vals = np.nan * np.empty(e_vals.shape)
h_guess0 = h_vals[0]
gam_guess0 = 0.
lift_guess0 = mass0 * g

# Get initial values for h (Newton's Method)
for idx, (m_val, e_val) in enumerate(zip(mass_vals, e_vals)):
    x, success = newton_search_x_hd(m_val, e_val, h_guess0, gam_guess0, lift_guess0)

    if not success:
        print(f'Newton Search Failed for Initial Guess at idx {idx}')
        # Try other initial conditions for h
        h_min = 0.
        h_max = 40e3
        h_vals = np.linspace(0., 0.9*e_val/g, 100)
        for h_val in h_vals:
            x, success = newton_search_x_hd(m_val, e_val, h_val, gam_guess0, lift_guess0)
            if success:
                break
    if not success:
        print(f'Newton Search Failed for All Guesses at idx {idx}')
    else:
        h_guess0 = x[0]
        gam_guess0 = x[1]
        lift_guess0 = x[2]
        h_es_vals[idx] = x[0]
        gam_es_vals[idx] = x[1]
        lift_es_vals[idx] = x[2]

h_es_guess_interp = sp.interpolate.PchipInterpolator(e_vals, h_es_vals)
v_es_vals = (2 * (e_vals - g * h_es_vals))**0.5
mach_es_vals = v_es_vals / np.asarray(sped_fun(h_es_vals)).flatten()
_gain_h_es = 1.


# ---- DYNAMICS & CONTROL LAWS -----------------------------------------------------------------------------------------
def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


def generate_constant_ctrl(_const: float) -> Callable:
    def _const_control(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
        return _const
    return _const_control


def zeroth_order_ae_ctrl_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _v = _x[1]
    _gam = _x[2]
    _mass = _x[3]
    _e = g * _h + 0.5 * _v**2

    # if _e > 4.46e6:
    #     print(_e*1e-6)

    # Solve for neighboring costate value to achieve required terminal conditions
    # Solutions have form:
    # s = sig +- j w [Complex eigenvalue]
    # v = vs +- j vw [Complex eigenvector]
    # [dx(dE); dlam(dE)] = exp(sig * dE) [vs vw] [cos(w dE) sin(w dE); -sin(w dE) cos(w dE)] [c1; c2]
    # With known final perturbation (gamf free) dhf = hf_cmd - hf_outer, dlam_gamf = 0
    _hf = _k_dict['h_f']
    _vf = _k_dict['v_f']
    _ef = g * _hf + 0.5 * _vf**2
    _ego = max(_ef - _e, 0.)
    _hf_guess = h_es_guess_interp(_ef)
    _energy_climb_dictf = find_climb_path(_mass, _ef, _hf_guess)
    eig_Gf, eig_vec_Gf = np.linalg.eig(_energy_climb_dictf['GE'])
    idx_stablef = np.where(np.real(eig_Gf) > 0)  # Req. stability BACKWARDS in energy -> positive eigenvalues
    eig_stablef = eig_Gf[idx_stablef]
    eig_vec_stablef = eig_vec_Gf[:, idx_stablef[0]]

    # Initial Conditions (Ego = 0)
    Vsf_Ego0 = np.hstack((np.real(eig_vec_stablef[:, 0].reshape(-1, 1)), np.imag(eig_vec_stablef[:, 0].reshape(-1, 1))))
    dhf_Ego0 = _hf - _energy_climb_dictf['h']
    dlam_gamf_Ego0 = 0.  # gamf free
    zf_known_Ego0 = np.vstack((dhf_Ego0, dlam_gamf_Ego0))
    Vsf_Ego0_known = Vsf_Ego0[(0, 3), :]  # dhf, dlam_gamf known [gamf free]
    cf = np.linalg.solve(Vsf_Ego0_known, zf_known_Ego0)
    _rotation_Ego = np.imag(eig_stablef[0]) * _ego

    # Saturate rotation to the value the rotation does not cause z to oscillate (i.e. find the rotation angles where
    # each element of z = 0 and choose the smallest)
    for _row in list(Vsf_Ego0):
        _rotation_Ego_row_max = np.arctan2(
            (_row[0] * cf[0, 0] + _row[1] * cf[1, 0]), (_row[1] * cf[0, 0] - _row[0] * cf[1, 0])
        )
        _rotation_Ego_row_max = _rotation_Ego_row_max % (2 * np.pi)  # Wrap to positive angle
        if _rotation_Ego > _rotation_Ego_row_max:
            _rotation_Ego = _rotation_Ego_row_max

    _c_dE = np.cos(_rotation_Ego)
    _s_dE = np.sin(_rotation_Ego)
    _DCM = np.vstack(((_c_dE, _s_dE), (-_s_dE, _c_dE)))
    Vsf = Vsf_Ego0 @ _DCM
    zf = np.exp(-np.real(eig_stablef[0])*_ego) * Vsf @ cf
    dhf = zf[0, 0]
    dgamf = zf[1, 0]
    dlam_gamf = 0.
    # dlam_gamf = zf[-1, 0]

    # # Saturate dhf, dgamf based on direction of terminal constraint (i.e. remove oscillations)
    # _sign_dhf = np.sign(dhf_Ego0)
    # dhf = _sign_dhf * max(dhf * _sign_dhf, 0.)
    # dgamf = _sign_dhf * max(dgamf * _sign_dhf, 0.)
    # dlam_gamf = -_sign_dhf * max(dlam_gamf * -_sign_dhf, 0.)  # sign(lam_gam) opposite to sin(lift)

    # Solve for neighboring costate value to damp out perturbation
    _h_guess = float(h_es_guess_interp(_e))
    _energy_climb_dict = find_climb_path(_mass, _e, _h_guess)
    eig_G, eig_vec_G = np.linalg.eig(_energy_climb_dict['G'])
    idx_stable = np.where(np.real(eig_G) < 0)
    eig_vec_stable = eig_vec_G[:, idx_stable[0]]
    Vs = np.hstack((np.real(eig_vec_stable[:, 0].reshape(-1, 1)), np.imag(eig_vec_stable[:, 0].reshape(-1, 1))))

    Vs_known = Vs[(0, 1), :]  # h0, gam0 fixed
    Vs_unknown = Vs[(2, 3), :]  # lam_h0, lam_gam0 unknown
    dh0 = _h - _energy_climb_dict['h']
    dgam0 = _gam - _energy_climb_dict['gam']
    # dh0 = _h - (_energy_climb_dict['h'] + dhf)  # Include dhf in offset
    # dgam0 = _gam - (_energy_climb_dict['gam'] + dgamf)
    z0_known = np.vstack((dh0, dgam0))
    z0_unknown = Vs_unknown @ np.linalg.solve(Vs_known, z0_known)
    dlam_h = z0_unknown[0, 0]
    dlam_gam = z0_unknown[1, 0]

    # Use the perturbed costate to estimate the optimal control
    _lam_gam = _energy_climb_dict['lam_gam'] + dlam_gam + dlam_gamf
    _lam_e = _energy_climb_dict['lam_E']
    _mach = _v / atm.speed_of_sound(_h)
    _CD2 = float(CD2_fun(_mach))
    _rho = atm.density(_h)
    _lift = _lam_gam/_lam_e * (_rho * s_ref) / (4 * _CD2)

    # Ensure lift results in positive T - D
    _thrust_max = float(thrust_fun(_mach, _h))
    _qdyn_s_ref = 0.5 * _rho * _v**2 * s_ref
    _CD0 = float(CD0_fun(_mach))
    _drag0 = _qdyn_s_ref * _CD0
    _lift_mag = abs(_lift)
    _lift_mag_max = (_qdyn_s_ref/_CD2 * (_thrust_max - _drag0))**0.5
    if _thrust_max < _drag0:
        # Choose min. drag value if it is impossible to achieve T > D
        _lift = 0.
    elif _lift_mag > _lift_mag_max:
        # If unconstrained lift causes T < D (loss of energy), constrain it.
        _lift = np.sign(_lift) * _lift_mag_max

    _load = _lift / (_mass * g)

    _thrust_frac = 1.
    # # Use max thrust until E = Ef. Then dE/dt = 0
    # if _e < _ef:
    #     _thrust_frac = 1.
    # else:
    #     _drag = _qdyn_s_ref * _CD0 + _CD2 / _qdyn_s_ref * _lift ** 2
    #     _thrust_frac = _drag / _thrust_max

    return np.asarray((_load, _thrust_frac))


# def climb_estimator_ctrl_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
#     _h = _x[0]
#     _v = _x[1]
#     _gam = _x[2]
#     _mass = _x[3]
#     # _h_es = _x[4]
#
#     _e = g * _h + 0.5 * _v**2
#     _mach = _v / atm.speed_of_sound(_h)
#     _CD0 = float(CD0_fun(_mach))
#     _CD2 = float(CD2_fun(_mach))
#     _rho = atm.density(_h)
#     _qdyn_s_ref = 0.5 * _rho * _v ** 2 * s_ref
#     _drag0 = _qdyn_s_ref * _CD0
#
#     _h_guess = float(h_es_guess_interp(_e))
#     _outer_dict = find_climb_path(_mass, _e, _h_guess)
#     _h_es = _outer_dict['h']
#
#     # Propagation of Energy State Estimate
#     _F = float(zero_fun_es(_mass, _e, _h_es))
#     _G = float(grad_fun_es(_mass, _e, _h_es))
#     _dhes_dt_ff = float(dhdt_es_fun(_mass, _e, _h_es))
#     _dhes_dt = -_gain_h_es/_G * _F + _dhes_dt_ff  # Gradient descent + FF term to account for d(E)/dt, d(m)/dt =/= 0
#
#     # Calculate d2(h)/dt2 due to energy state (feed forward)
#     _d2hes_dt2 = float(d2hdt2_es_fun(_mass, _e, _h_es))
#
#     # Calculate displacement from equilibrium state (for feedback)
#     # z = z0 + zf
#     # d2(dh)/dt2 = Gh G z
#     # First, calculate zf from h_esf, gam_es
#     zf = np.zeros((4, 1))
#     # Second, calculate z0 from h_es, gam_es
#     _gam_es = gam_es_fun(_mass, _e, _h_es)
#     _dx0 = np.vstack((_h_es, _gam_es))
#
#     _G = float(G_es_fun(_mass, _e, _h_es))
#     eig_G, eig_vec_G = np.linalg.eig(_G)
#     idx_stable = np.where(np.real(eig_G) < 0)
#     eig_vec_stable = eig_vec_G[:, idx_stable[0]]
#     Vs = np.hstack((np.real(eig_vec_stable[:, 0].reshape(-1, 1)), np.imag(eig_vec_stable[:, 0].reshape(-1, 1))))
#
#     Vs_x = Vs[(0, 1), :]  # h0, gam0 fixed -> use to determine constants for z0
#     z0 = Vs @ np.linalg.solve(Vs_x, _dx0)
#     # Third, calculate d2(dh)/dt2 from z = z0 + zf
#     z = z0 + zf
#     d2dhdt2 = _G[0, :] @ _G @ z
#     d2hdt2 = _d2hes_dt2 + d2dhdt2
#
#     # Calculate control input in order to achieve d2(h)/dt2
#     # d2(h)/dt2 = -D/m sin(gam) - g sin(gam)**2 + L/m cos(gam) - g cos(gam)**2
#     # For gam ~= 0:
#     # d2(h)/dt2 = L/m - g
#     # Hence the lift is:
#     _lift = _mass * (d2hdt2 + g)
#
#     # Ensure lift results in positive T - D
#     _thrust_max = float(thrust_fun(_mach, _h))
#     _lift_mag = abs(_lift)
#     _lift_mag_max = (_qdyn_s_ref/_CD2 * (_thrust_max - _drag0))**0.5
#     if _thrust_max < _drag0:
#         # Choose min. drag value if it is impossible to achieve T > D
#         _lift = 0.
#     elif _lift_mag > _lift_mag_max:
#         # If unconstrained lift causes T < D (loss of energy), constrain it.
#         _lift = np.sign(_lift) * _lift_mag_max
#
#     _thrust_frac = 1.
#     # # Use max thrust until E = Ef. Then dE/dt = 0
#     # if _e < _ef:
#     #     _thrust_frac = 1.
#     # else:
#     #     _drag = _qdyn_s_ref * _CD0 + _CD2 / _qdyn_s_ref * _lift ** 2
#     #     _thrust_frac = _drag / _thrust_max
#
#     _load = _lift / (_mass * g)
#
#     if hasattr(_load, '__len__') or hasattr(_thrust_frac, '__len__'):
#         print('Oops!')
#
#     return np.asarray((_load, _thrust_frac))


def generate_ctrl_law(_ctrl_law, _u_interp=None) -> Callable:
    if _ctrl_law == 'interp':
        def _ctrl_fun(_t, _x, _p_dict, _k_dict):
            _load = _u_interp(_t)
            # _load = saturate(_load, load_min, load_max)
            _thrust_frac = 1.
            return np.asarray((_load, _thrust_frac))
    elif _ctrl_law == 'aenoc':
        _ctrl_fun = zeroth_order_ae_ctrl_law
    # elif _ctrl_law == 'd2hdt2':
    #     _ctrl_fun = climb_estimator_ctrl_law
    else:
        _ctrl_fun = generate_constant_ctrl(0.)
    return _ctrl_fun


def eom(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _v = _x[1]
    _gam = _x[2]
    _mass = _x[3]

    _load = _u[0]
    _thrust_frac = _u[1]
    _lift = _load * _mass * g

    _mach = _v / atm.speed_of_sound(_h)
    CD0 = float(CD0_fun(_mach))
    CD2 = float(CD2_fun(_mach))

    _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
    _cl = _lift / _qdyn_s_ref
    _cd = CD0 + CD1 * _cl + CD2 * _cl**2
    _drag = _qdyn_s_ref * _cd
    _thrust = _thrust_frac * float(thrust_fun(_mach, _h))

    _dh = _v * np.sin(_gam)
    _dv = (_thrust - _drag) / _mass - g * np.sin(_gam)
    _dgam = _lift / (_mass * _v) - g/_v * np.cos(_gam)
    _dm = -_thrust / Isp

    if hasattr(_dh, '__len__') or hasattr(_dv, '__len__') or hasattr(_dgam, '__len__') or hasattr(_dm, '__len__'):
        print('Ragged array!')
        zeroth_order_ae_ctrl_law(_t, _x, _p_dict, _k_dict)

    return np.array((_dh, _dv, _dgam, _dm))


def generate_termination_events(_ctrl_law, _p_dict, _k_dict, _limits_dict):
    def min_altitude_event(_t: float, _x: np.array) -> float:
        return _x[0] - _limits_dict['h_min']

    def min_velocity_event(_t: float, _x: np.array) -> float:
        return _x[1] - _limits_dict['v_min']

    def min_fpa_event(_t: float, _x: np.array) -> float:
        return _x[2] - _limits_dict['gam_min']

    def max_fpa_event(_t: float, _x: np.array) -> float:
        return _limits_dict['gam_max'] - _x[2]

    def max_e_event(_t: float, _x: np.array) -> float:
        _e = g * _x[0] + 0.5 * _x[1]**2
        return _limits_dict['e_max'] - _e

    events = [min_altitude_event, min_velocity_event,
              min_fpa_event, max_fpa_event,
              max_e_event]

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
ndmult = np.array((k_dict['h_scale'], k_dict['v_scale'], 1., k_dict['m_scale']))

print('____ Evaluation ____')
for idx, sol in enumerate(sols):
    for key, val in zip(sol.annotations.states, list(sol.x)):
        x_dict[key] = val
    for key, val in zip(sol.annotations.constants, sol.k):
        k_dict[key] = val
    for key, val in zip(sol.annotations.parameters, sol.p):
        p_dict[key] = val
    e_opt = g * x_dict['h_nd'] * k_dict['h_scale'] + 0.5 * (x_dict['v_nd'] * k_dict['v_scale']) ** 2
    load_opt = sol.u[0, :]
    u_opt = sp.interpolate.PchipInterpolator(sol.t, load_opt)

    t0 = sol.t[0]
    tf = sol.t[-1]

    t_span = np.array((t0, np.inf))
    x0 = sol.x[:, 0] * ndmult
    limits_dict['e_max'] = np.max(e_opt)

    # States
    h_opt = sol.x[0, :] * k_dict['h_scale']
    gam_opt = sol.x[2, :]
    v_opt = sol.x[1, :] * k_dict['v_scale']
    mach_opt = v_opt / np.asarray(sped_fun(h_opt)).flatten()

    x_opt = np.vstack((e_opt, h_opt, gam_opt, mach_opt))

    h0_arr[idx] = x_dict['h_nd'][0] * k_dict['h_scale']
    v0_arr[idx] = x_dict['v_nd'][0] * k_dict['v_scale']

    ivp_sols_dict[idx] = {}
    for ctrl_law_type in CTRL_LAWS:
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
        load_sol = np.empty(shape=(ivp_sol.t.shape[0],))
        for jdx, (t, x) in enumerate(zip(ivp_sol.t, ivp_sol.y.T)):
            load_sol[jdx] = ctrl_law(t, x, p_dict, k_dict)[0]

        # Wrap FPA
        _fpa_unwrapped = ivp_sol.y[2, :]
        _fpa = (_fpa_unwrapped + np.pi) % (2 * np.pi) - np.pi
        ivp_sol.y[2, :] = _fpa

        # Calculate State
        h = ivp_sol.y[0, :]
        v = ivp_sol.y[1, :]
        gam = ivp_sol.y[2, :]
        mass = ivp_sol.y[3, :]
        mach = v / np.asarray(sped_fun(h)).flatten()
        e = g*h + 0.5 * v**2
        x = np.vstack((e, h, gam, mach))

        # Calculate Alternate Control
        qdyn_s_ref = 0.5 * np.asarray(dens_fun(h)).flatten() * v**2 * s_ref
        CL = load_sol * g * mass / qdyn_s_ref
        CLa = np.asarray(CLa_fun(mach)).flatten()
        alpha_sol = (CL - CL0) / CLa

        ivp_sols_dict[idx][ctrl_law_type] = {
            't': ivp_sol.t,
            'x': x,
            'x_opt': x_opt,
            'optimality': ivp_sol.t[-1] / sol.t[-1],
            'lift_nd': load_sol,
            'CL': CL,
            'alpha': alpha_sol,
            'tf_opt': sol.t[-1],
            'tf': ivp_sol.t[-1],
            'lift_nd_opt': load_opt
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
ylabs = (r'$h$ [1,000 ft]', r'$\gamma$ [deg]', r'$M$', r'$L$ [g]')
ymult = np.array((1e-3, r2d, 1., 1.))
xlab = r'$E$ [$10^6$ ft$^2$/m$^2$]'
xmult = 1e-6

plt_order = (0, 2, 1, 3)

figs = []
axes_list = []
for idx, ivp_sol_dict in enumerate(ivp_sols_dict):
    figs.append(plt.figure())
    fig = figs[-1]

    fig_name = 'airplane2_mtc_case' + str(idx + 1)

    y_arr_opt = np.vstack((
        ivp_sols_dict[idx][CTRL_LAWS[0]]['x_opt'][1:, :],
        ivp_sols_dict[idx][CTRL_LAWS[0]]['lift_nd_opt'],
    ))
    ydata_opt = list(y_arr_opt)
    xdata_opt = ivp_sols_dict[idx][CTRL_LAWS[0]]['x_opt'][0, :]

    y_arr_ec = np.vstack((
        h_es_vals, gam_es_vals, mach_es_vals, lift_es_vals/(mass_vals * g)
    ))
    ydata_ec = list(y_arr_ec)
    xdata_ec = e_vals

    ydata_eval_list = []
    xdata_eval_list = []

    for ctrl_law_type in CTRL_LAWS:
        y_arr = np.vstack((
            ivp_sols_dict[idx][ctrl_law_type]['x'][1:, :],
            ivp_sols_dict[idx][ctrl_law_type]['lift_nd'],
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
        # ax.invert_xaxis()

        ax.plot(xdata_opt * xmult, ydata_opt[plt_idx] * ymult[plt_idx], color=col[0], label=f'Optimal')

        for eval_idx, x in enumerate(xdata_eval_list):
            if ydata_eval_list[eval_idx][plt_idx] is not None:
                ax.plot(
                    x * xmult, ydata_eval_list[eval_idx][plt_idx] * ymult[plt_idx],
                    color=col[1+eval_idx], label=CTRL_LAWS[eval_idx]
                )

        ax.plot(xdata_ec * xmult, ydata_ec[plt_idx] * ymult[plt_idx], 'k--', label=f'Energy Climb')

    fig.tight_layout()
    fig.savefig(
        fig_name + '.eps',
        format='eps',
        bbox_inches='tight'
    )

plt.show()
