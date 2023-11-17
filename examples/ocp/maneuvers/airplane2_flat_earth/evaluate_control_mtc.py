from typing import Callable

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

from airplane2_aero_atm import g, s_ref, CD0_fun, CD1, CD2_fun, CL0, CLa_fun, max_ld_fun, atm,\
    dens_fun, sped_fun, gam_qdyn0, thrust_fun, Isp
from ardema_mae import find_climb_path, obj_fun_es, zero_fun_es, grad_fun_es, G_es_fun, GE_es_fun,\
    newton_seach_multiple_start, obj_fun_full

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARE_SWEEP = True
# CTRL_LAWS are: {interp, aenoc, dgamdt}
CTRL_LAWS = ('dgamdt',)
# CTRL_LAWS = []

with open('sol_set_mtc_xf.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[0]
    sol_tec = sols[-1]
    sols = [sol, sol_tec]
    # sol_tec = sol

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

limits_dict = {'h_max': 0., 'h_min': h_min, 'v_min': v_min, 'gam_min': gam_min, 'gam_max': gam_max, 'e_max': 0.}

# Energy State Dict (for control law)
h_vals = sol.x[0, :] * k_dict['h_scale']
v_vals = sol.x[1, :] * k_dict['v_scale']
mass_vals = sol.x[3, :] * k_dict['m_scale']
mass0 = mass_vals[0]
e_vals = g * h_vals + 0.5 * v_vals**2
h_es_vals = np.nan * np.empty(e_vals.shape)
gam_es_vals = np.nan * np.empty(e_vals.shape)
lift_es_vals = np.nan * np.empty(e_vals.shape)
lam_gam_es_vals = np.nan * np.empty(e_vals.shape)
h_guess0 = h_vals[0]
gam_guess0 = 0.
lam_gam_guess0 = None

# Get initial values for h (Newton's Method)
for idx, (m_val, e_val) in enumerate(zip(mass_vals, e_vals)):
    x, lift_val, success = newton_seach_multiple_start(m_val, e_val, h_guess0, gam_guess0, lam_gam_guess0)

    if not success:
        print(f'Newton Search Failed for All Guesses at idx {idx}')
    else:
        h_guess0 = x[0]
        gam_guess0 = x[1]
        lam_gam_guess0 = x[2]
        h_es_vals[idx] = x[0]
        gam_es_vals[idx] = x[1]
        lam_gam_es_vals[idx] = x[2]
        lift_es_vals[idx] = lift_val

h_es_guess_interp = sp.interpolate.PchipInterpolator(e_vals, h_es_vals)
gam_es_guess_interp = sp.interpolate.PchipInterpolator(e_vals, gam_es_vals)
lam_gam_es_guess_interp = sp.interpolate.PchipInterpolator(e_vals, lam_gam_es_vals)
lift_es_guess_interp = sp.interpolate.PchipInterpolator(e_vals, lift_es_vals)
v_es_vals = (2 * (e_vals - g * h_es_vals))**0.5
mach_es_vals = v_es_vals / np.asarray(sped_fun(h_es_vals)).flatten()
_gain_h_es = 1.

# Linearization of terminal boundary layer
hf_vals = sol_tec.x[0, :] * k_dict['h_scale']
vf_vals = sol_tec.x[1, :] * k_dict['v_scale']
gamf_vals = sol_tec.x[2, :]
massf_vals = sol_tec.x[3, :] * k_dict['m_scale']
ef_vals = g * hf_vals + 0.5 * vf_vals**2

hf_es_vals = np.empty(ef_vals.shape)
gamf_es_vals = np.empty(ef_vals.shape)
liftf_es_vals = np.empty(ef_vals.shape)
lam_gamf_es_vals = np.empty(ef_vals.shape)

for idx, (m_val, e_val) in enumerate(zip(massf_vals, ef_vals)):
    x, lift_val, success = newton_seach_multiple_start(m_val, e_val, h_guess0, gam_guess0, lam_gam_guess0)

    if not success:
        print(f'Newton Search Failed for All Guesses at idx {idx}')
    else:
        h_guess0 = x[0]
        gam_guess0 = x[1]
        lam_gam_guess0 = x[2]
        hf_es_vals[idx] = x[0]
        gamf_es_vals[idx] = x[1]
        lam_gamf_es_vals[idx] = x[2]
        liftf_es_vals[idx] = lift_val

vf_es_vals = (2*(ef_vals - g * hf_vals))**0.5
dgamf_es_vals = (liftf_es_vals - massf_vals * g * np.cos(gamf_es_vals))/(massf_vals * vf_es_vals)

GEf = np.array(GE_es_fun(massf_vals[-1], ef_vals[-1], hf_es_vals[-1], gamf_es_vals[-1]))
Gf = np.array(G_es_fun(massf_vals[-1], ef_vals[-1], hf_es_vals[-1], gamf_es_vals[-1]))
eig_Gf, eig_vec_Gf = np.linalg.eig(GEf)
idx_stablef = np.where(np.real(eig_Gf) > 0)  # Req. stability BACKWARDS in energy -> positive eigenvalues
eig_stablef = eig_Gf[idx_stablef]
eig_vec_stablef = eig_vec_Gf[:, idx_stablef[0]]

Vsf_Ego0 = np.hstack((np.real(eig_vec_stablef[:, 0].reshape(-1, 1)), np.imag(eig_vec_stablef[:, 0].reshape(-1, 1))))
dhf_Ego0 = hf_vals[-1] - hf_es_vals[-1]
dlam_gamf_Ego0 = 0. - lam_gamf_es_vals[-1]  # gamf free
zf_known_Ego0 = np.vstack((dhf_Ego0, dlam_gamf_Ego0))
Vsf_Ego0_known = Vsf_Ego0[(0, 3), :]  # dhf, dlam_gamf known [gamf free]
cf = np.linalg.solve(Vsf_Ego0_known, zf_known_Ego0)

# # lam_gamf corresponds to d(gam)/dt|f = 0. Solve iteratively to find that solution rather than relying on linear
# # approximation of lam_gamf
# gamf = gamf_es_vals[-1]
# cf = np.nan * np.empty((2, 1))
# for idx in range(1000):
#     gamf_last = gamf
#     ddgamdtf_last = -g/vf_vals[-1] * np.cos(gamf) - dgamf_es_vals[-1]
#     cf = np.linalg.solve(np.vstack((Vsf_Ego0[0, :], Gf[1, :] @ Vsf_Ego0)), np.vstack((dhf_Ego0, ddgamdtf_last)))
#     gamf = gamf_es_vals[-1] + float(Vsf_Ego0[1, :] @ cf)
#     ddgamdtf_err = (-g / vf_vals[-1] * np.cos(gamf) - dgamf_es_vals[-1]) - float(Gf[1, :] @ Vsf_Ego0 @ cf)
#
#     if abs(gamf - gamf_last) < 1e-8 and abs(ddgamdtf_err) < 1e-8:
#         break

_dEf_vals = np.minimum(ef_vals - ef_vals[-1], 0.)

hf_tec_vals = np.empty(ef_vals.shape)
gamf_tec_vals = np.empty(ef_vals.shape)
dgamf_tec_vals = np.empty(ef_vals.shape)

for idx, _dEf_val in enumerate(_dEf_vals):
    _rotation_Ego = np.imag(eig_stablef[0]) * _dEf_val

    # # Saturate rotation to the value the rotation does not cause z to oscillate (i.e. find the rotation angles where
    # # each element of z = 0 and choose the smallest)
    # for _row in list(Vsf_Ego0):
    #     _rotation_Ego_row_max = np.arctan2(
    #         (_row[0] * cf[0, 0] + _row[1] * cf[1, 0]), (_row[1] * cf[0, 0] - _row[0] * cf[1, 0])
    #     )
    #     _rotation_Ego_row_max = _rotation_Ego_row_max % (2 * np.pi)  # Wrap to positive angle
    #     if _rotation_Ego > _rotation_Ego_row_max:
    #         _rotation_Ego = _rotation_Ego_row_max

    _c_dE = np.cos(_rotation_Ego)
    _s_dE = np.sin(_rotation_Ego)
    _DCM = np.vstack(((_c_dE, _s_dE), (-_s_dE, _c_dE)))
    Vsf = Vsf_Ego0 @ _DCM
    zf = np.exp(np.real(eig_stablef[0]) * _dEf_val) * Vsf @ cf

    hf_tec_vals[idx] = zf[0, 0] + hf_es_vals[idx]
    gamf_tec_vals[idx] = zf[1, 0] + gamf_es_vals[idx]
    dgamf_tec_vals[idx] = float(Gf[1, :] @ zf) + dgamf_es_vals[idx]

vf_tec_vals = (2 * (ef_vals - g * hf_tec_vals))**0.5
machf_tec_vals = vf_tec_vals / np.asarray(sped_fun(hf_tec_vals)).flatten()
liftf_tec_vals = massf_vals * (g * np.cos(gamf_tec_vals) + vf_tec_vals * dgamf_tec_vals)

# hf_ec_vals = np.empty(ef_vals.shape)
# gamf_ec_vals = np.empty(ef_vals.shape)
# dgamf_ec_vals = np.empty(ef_vals.shape)
#
# # Full Linearization (include stable and unstable modes, but at present energy rather than Ef)
# for idx, ef_val in enumerate(ef_vals):
#     GEf = np.array(GE_es_fun(massf_vals[idx], ef_val, hf_es_vals[idx], gamf_es_vals[idx]))
#     Gf = np.array(G_es_fun(massf_vals[idx], ef_val, hf_es_vals[idx], gamf_es_vals[idx]))
#
#     eig_GEf, eig_vec_GEf = np.linalg.eig(GEf)
#     idx_stablef = np.where(np.real(eig_GEf) < 0)  # Stable modes to suppress path error
#     idx_unstablef = np.where(np.real(eig_GEf) > 0)  # Unstable modes to drive dy -> dyf
#     eig_stablef = eig_GEf[idx_stablef]
#     eig_unstablef = eig_GEf[idx_unstablef]
#     eig_vec_stablef = eig_vec_GEf[:, idx_stablef[0]]
#     eig_vec_unstablef = eig_vec_GEf[:, idx_unstablef[0]]
#
#     ego = np.maximum(ef_vals[-1] - ef_val, 100.)  # When Ego = 0, matrix inversion becomes ill conditioned
#
#     theta_egos = np.imag(eig_stablef[0]) * ego
#     DCM_egos = np.vstack(((np.cos(theta_egos), np.sin(theta_egos)), (-np.sin(theta_egos), np.cos(theta_egos))))
#     Vs0 = np.hstack((np.real(eig_vec_stablef[:, 0].reshape(-1, 1)), np.imag(eig_vec_stablef[:, 0].reshape(-1, 1))))
#     Vsf = np.exp(np.real(eig_stablef[0]) * ego) * Vs0 @ DCM_egos
#
#     theta_egou = np.imag(eig_unstablef[0]) * ego
#     DCM_egou = np.vstack(((np.cos(theta_egou), np.sin(theta_egou)), (-np.sin(theta_egou), np.cos(theta_egou))))
#     Vu0 = np.hstack((np.real(eig_vec_unstablef[:, 0].reshape(-1, 1)), np.imag(eig_vec_unstablef[:, 0].reshape(-1, 1))))
#     Vuf = np.exp(np.real(eig_unstablef[0]) * ego) * Vu0 @ DCM_egou
#
#     V0 = np.hstack((Vs0, Vu0))
#     Vf = np.hstack((Vsf, Vuf))
#
#     # Calculate free coefficients in order to fix initial/final state
#     dh0 = hf_vals[idx] - hf_es_vals[idx]
#     dgam0 = gamf_vals[idx] - gamf_es_vals[idx]
#     dhf = hf_vals[-1] - hf_es_vals[idx]
#     dlam_gamf = 0.
#
#     idx_fixed0 = (0, 1)  # Fix dh0, dgam0 [because we can't change the initial state :)]
#     idx_fixedf = (0, 3)  # Fix dhf, dlamgamf [i.e. final FPA free -> dlamgamf = 0]
#
#     V_fixed = np.vstack((V0[idx_fixed0, :], Vf[idx_fixedf, :]))
#     z_fixed = np.vstack((dh0, dgam0, dhf, dlam_gamf))
#     coeffs = np.linalg.solve(V_fixed, z_fixed)
#     z0 = V0 @ coeffs
#
#     hf_ec_vals[idx] = z0[0, 0] + hf_es_vals[idx]
#     gamf_ec_vals[idx] = z0[1, 0] + gamf_es_vals[idx]
#     dgamf_ec_vals[idx] = float(Gf[1, :] @ z0) + dgamf_es_vals[idx]
#
# vf_ec_vals = (2 * (ef_vals - g * hf_ec_vals))**0.5
# machf_ec_vals = vf_ec_vals / np.asarray(sped_fun(hf_ec_vals)).flatten()
# liftf_ec_vals = massf_vals * (g * np.cos(gamf_ec_vals) + vf_ec_vals * dgamf_ec_vals)


# ---- DYNAMICS & CONTROL LAWS -----------------------------------------------------------------------------------------
def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


def generate_constant_ctrl(_const: float) -> Callable:
    def _const_control(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> float:
        return _const
    return _const_control


def climb_estimator_ctrl_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _v = _x[1]
    _gam = _x[2]
    _mass = _x[3]
    # _h_es_guess = _x[4]
    # _h_esf_guess = _x[5]

    _e = g * _h + 0.5 * _v**2
    _mach = _v / atm.speed_of_sound(_h)
    _CD0 = float(CD0_fun(_mach))
    _CD2 = float(CD2_fun(_mach))
    _rho = atm.density(_h)
    _qdyn_s_ref = 0.5 * _rho * _v ** 2 * s_ref
    _drag0 = _qdyn_s_ref * _CD0

    _h_es_guess = float(h_es_guess_interp(_e))
    _gam_es_guess = float(gam_es_guess_interp(_e))
    _lam_gam_es_guess = float(lam_gam_es_guess_interp(_e))
    _lift_es = float(lift_es_guess_interp(_e))
    # _x_es, _lift_es, _success = newton_seach_multiple_start(
    #     _mass, _e, _h=_h_es_guess, _gam=_gam_es_guess, _lam_gam=_lam_gam_es_guess
    # )
    #
    # # Propagation of Energy State Estimate
    # if _success:
    #     _h_es = _x_es[0]
    #     _gam_es = _x_es[1]
    #     _dhes_dt = -_gain_h_es * (_h_es - _h_es_guess)
    # else:
    #     _h_es = _h_es_guess
    #     _gam_es = _gam_es_guess
    #     _dhes_dt = 0.
    _h_es = _h_es_guess
    _gam_es = _gam_es_guess
    _dhes_dt = 0.

    # _v_es = _v
    _v_es = (2*(_e - g * _h_es))**0.5
    _dgames_dt = (_lift_es - _mass * g * np.cos(_gam_es)) / (_mass * _v_es)

    # Modify path to include offset for terminal criteria --------------------------------------------------------------
    _hf = k_dict['h_f']
    _vf = k_dict['v_f']
    _ef = g * _hf + 0.5 * _vf**2
    _h_esf_guess = float(h_es_guess_interp(_ef))
    _gam_esf_guess = float(gam_es_guess_interp(_ef))
    _lam_gam_esf_guess = float(lam_gam_es_guess_interp(_ef))
    # _x_esf, _lift_esf, _success = newton_seach_multiple_start(
    #     _mass, _ef, _h=_h_esf_guess, _gam=_gam_esf_guess, _lam_gam=_lam_gam_esf_guess
    # )
    #
    # # Propagation of Terminal Energy State Estimate
    # if _success:
    #     _h_esf = _x_esf[0]
    #     _gam_esf = _x_esf[1]
    #     _lam_gam_esf = _x_esf[2]
    #     _dhesf_dt = -_gain_h_es * (_h_esf - _h_esf_guess)
    # else:
    #     _h_esf = _h_esf_guess
    #     _gam_esf = _gam_esf_guess
    #     _lam_gam_esf = _lam_gam_esf_guess
    #     _dhesf_dt = 0.
    _h_esf = _h_esf_guess
    _gam_esf = _gam_esf_guess
    _lam_gam_esf = _lam_gam_esf_guess
    _dhesf_dt = 0.

    _dhf = _hf - _h_esf
    _dlamgamf = 0. - _lam_gam_esf
    _def = min(_e - _ef, 0.)

    GEf = np.array(GE_es_fun(_mass, _ef, _h_esf, _gam_esf))
    Gf = np.array(G_es_fun(_mass, _ef, _h_esf, _gam_esf))
    eig_Gf, eig_vec_Gf = np.linalg.eig(GEf)
    idx_stablef = np.where(np.real(eig_Gf) > 0)  # Req. stability BACKWARDS in energy -> positive eigenvalues
    eig_stablef = eig_Gf[idx_stablef]
    eig_vec_stablef = eig_vec_Gf[:, idx_stablef[0]]

    Vsf_Ego0 = np.hstack((np.real(eig_vec_stablef[:, 0].reshape(-1, 1)), np.imag(eig_vec_stablef[:, 0].reshape(-1, 1))))
    zf_known_Ego0 = np.vstack((_dhf, _dlamgamf))
    Vsf_Ego0_known = Vsf_Ego0[(0, 3), :]  # dhf, dlam_gamf known [gamf free]
    cf = np.linalg.solve(Vsf_Ego0_known, zf_known_Ego0)

    _theta_de = np.imag(eig_stablef[0]) * _def
    _c_dE = np.cos(_theta_de)
    _s_dE = np.sin(_theta_de)
    _DCM = np.vstack(((_c_dE, _s_dE), (-_s_dE, _c_dE)))
    zf = np.exp(np.real(eig_stablef[0]) * _def) * Vsf_Ego0 @ _DCM @ cf

    _h_ref = _h_es + zf[0, 0]
    _gam_ref = _gam_es + zf[1, 0]
    _dgam_dt_ref = _dgames_dt + float(Gf[1, :] @ zf)

    # TODO - dh/dt and dgam/dt reference are not simultaneously feasible.
    #
    # _v_ref = (2 * (_e - g * _h_ref))**0.5
    # _dh_dt_ref = _v_es * np.sin(_gam_es) + float(Gf[0, :] @ zf)
    # _v_ref = _dh_dt_ref / np.sin(_gam_ref)
    # _h_ref = (_e - 0.5 * _v_ref**2)/g
    # _gam_ref =

    # Calculate displacement from equilibrium state (for feedback) -----------------------------------------------------
    # d(dgam0)/dt = G0gam z0
    # Second, calculate z0 from h_es, gam_es
    _dx0 = np.vstack((_h - _h_ref, _gam - _gam_ref))

    _G = np.asarray(G_es_fun(_mass, _e, _h_es, _gam_es))
    _eig_G, _eig_vec_G = np.linalg.eig(_G)
    _idx_stable = np.where(np.real(_eig_G) < 0)
    _eig_vec_stable = _eig_vec_G[:, _idx_stable[0]]
    _Vs = np.hstack((np.real(_eig_vec_stable[:, 0].reshape(-1, 1)), np.imag(_eig_vec_stable[:, 0].reshape(-1, 1))))
    _Vs_x = _Vs[(0, 1), :]  # h0, gam0 fixed -> use to determine constants for z0
    _z0 = _Vs @ np.linalg.solve(_Vs_x, _dx0)
    # Third, calculate d(dgam)/dt from z0
    _ddgamdt0 = float(_G[1, :] @ _z0)

    # Calculate control input in order to achieve d(gam)/dt
    _dgamdt = _dgam_dt_ref + _ddgamdt0
    # Lift = W cos(gam) + mV d(gam)/dt
    _lift = _mass * (g * np.cos(_gam) + _v * _dgamdt)

    # Ensure lift results in positive T - D
    _thrust_max = float(thrust_fun(_mach, _h))
    _lift_mag = abs(_lift)
    _lift_mag_max = (_qdyn_s_ref/_CD2 * (_thrust_max - _drag0))**0.5
    if _thrust_max < _drag0:
        # Choose min. drag value if it is impossible to achieve T > D
        _lift = 0.
    elif _lift_mag > _lift_mag_max:
        # If unconstrained lift causes T < D (loss of energy), constrain it.
        _lift = np.sign(_lift) * _lift_mag_max

    _thrust_frac = 1.
    # # Use max thrust until E = Ef. Then dE/dt = 0
    # if _e < _ef:
    #     _thrust_frac = 1.
    # else:
    #     _drag = _qdyn_s_ref * _CD0 + _CD2 / _qdyn_s_ref * _lift ** 2
    #     _thrust_frac = _drag / _thrust_max

    _load = _lift / (_mass * g)

    if hasattr(_load, '__len__') or hasattr(_thrust_frac, '__len__'):
        print('Oops!')

    return np.asarray((_load, _thrust_frac))


# def backstepping_ctrl_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
#     _h = _x[0]
#     _v = _x[1]
#     _gam = _x[2]
#     _mass = _x[3]
#
#     _e = g * _h + 0.5 * _v**2
#     _mach = _v / atm.speed_of_sound(_h)
#     _CD0 = float(CD0_fun(_mach))
#     _CD2 = float(CD2_fun(_mach))
#     _rho = atm.density(_h)
#     _qdyn_s_ref = 0.5 * _rho * _v ** 2 * s_ref
#     _drag0 = _qdyn_s_ref * _CD0
#     _drag2_inv = _qdyn_s_ref / _CD2
#     _thrust = float(thrust_fun(_mach, _h))
#     _weight = _mass * g
#
#     # Outer Loop (find h* at current E)
#     _h_es_guess = float(h_es_guess_interp(_e))
#     _dict = find_climb_path(_mass, _e, _h_es_guess)
#     _h_es = _dict['h']
#     _v_es = _dict['V']
#     _thrust_es = _dict['T']
#     _drag_es = _dict['D']
#
#     # First Inner Loop (find gam* at current h)
#     _gam_es2 = _drag2_inv/_weight**2 * (_thrust - _drag0 - _v_es/_v * (_thrust_es - _drag_es)) - 1
#     _gam_es = np.sign(_h_es - _h) * saturate(_gam_es2, 0., 1.)**0.5
#
#     # Second Inner Loop (find L* at current gam)
#     _beta =
#
#     if not _dict['success']:
#         print('Oops! Climb path not found! :/')
#
#     _thrust_frac = 1.
#     _load = _lift / (_mass * g)
#
#     if hasattr(_load, '__len__') or hasattr(_thrust_frac, '__len__'):
#         print('Oops! u wrong shape! :/')
#
#     return np.asarray((_load, _thrust_frac))


def generate_ctrl_law(_ctrl_law, _u_interp=None) -> Callable:
    if _ctrl_law == 'interp':
        def _ctrl_fun(_t, _x, _p_dict, _k_dict):
            _load = _u_interp(_t)
            # _load = saturate(_load, load_min, load_max)
            _thrust_frac = 1.
            return np.asarray((_load, _thrust_frac))
    elif _ctrl_law == 'dgamdt':
        _ctrl_fun = climb_estimator_ctrl_law
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

    return np.array((_dh, _dv, _dgam, _dm))


def generate_termination_events(_ctrl_law, _p_dict, _k_dict, _limits_dict):
    def min_altitude_event(_t: float, _x: np.array) -> float:
        return _x[0] - _limits_dict['h_min']

    # def max_altitude_event(_t: float, _x: np.array) -> float:
    #     return _limits_dict['h_max'] - _x[0]

    def min_velocity_event(_t: float, _x: np.array) -> float:
        return _x[1] - _limits_dict['v_min']

    def min_fpa_event(_t: float, _x: np.array) -> float:
        return _x[2] - _limits_dict['gam_min']

    def max_fpa_event(_t: float, _x: np.array) -> float:
        return _limits_dict['gam_max'] - _x[2]

    def max_e_event(_t: float, _x: np.array) -> float:
        _e = g * _x[0] + 0.5 * _x[1]**2
        return _limits_dict['e_max'] - _e

    events = [min_altitude_event, min_velocity_event, min_fpa_event, max_fpa_event, max_e_event]
    # events = [min_altitude_event, max_altitude_event, min_velocity_event, min_fpa_event, max_fpa_event, max_e_event]

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
    h_opt = x_dict['h_nd'] * k_dict['h_scale']
    e_opt = g * h_opt + 0.5 * (x_dict['v_nd'] * k_dict['v_scale']) ** 2
    load_opt = sol.u[0, :]
    u_opt = sp.interpolate.PchipInterpolator(sol.t, load_opt)

    t0 = sol.t[0]
    tf = sol.t[-1]

    t_span = np.array((t0, 2 * tf))
    x0 = sol.x[:, 0] * ndmult
    limits_dict['e_max'] = np.max(e_opt)
    limits_dict['h_max'] = np.max(h_opt)

    # States
    h_opt = sol.x[0, :] * k_dict['h_scale']
    gam_opt = sol.x[2, :]
    v_opt = sol.x[1, :] * k_dict['v_scale']
    mass_opt = sol.x[2, :] * k_dict['m_scale']
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

    y_arr_tec = np.vstack((
        hf_tec_vals, gamf_tec_vals, machf_tec_vals, liftf_tec_vals/(massf_vals * g)
    ))
    ydata_tec = list(y_arr_tec)
    xdata_tec = ef_vals

    # y_arr_ec = np.vstack((
    #     hf_ec_vals, gamf_ec_vals, machf_ec_vals, liftf_ec_vals/(massf_vals * g)
    # ))
    # ydata_ec = list(y_arr_ec)
    # xdata_ec = ef_vals


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

        ax.plot(xdata_tec * xmult, ydata_tec[plt_idx] * ymult[plt_idx], '--', label=f'Energy Climb')
        ax.plot(xdata_ec * xmult, ydata_ec[plt_idx] * ymult[plt_idx], 'k--', label=f'Energy Climb')

    fig.tight_layout()
    fig.savefig(
        fig_name + '.eps',
        format='eps',
        bbox_inches='tight'
    )

plt.show()
