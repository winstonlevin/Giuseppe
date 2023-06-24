import os
import sys
from typing import Optional

import numpy as np
import scipy as sp
import casadi as ca

from lookup_tables import cl_alpha_table, cd0_table, atm, dens_fun, sped_fun


# For consistency, use LUT values for speed of sound and density
def speed_of_sound(_h):
    _a = atm.speed_of_sound(_h)
    return _a


def density(_h):
    _rho = atm.density(_h)
    return _rho


# Find Glide Slope (assuming small FPA)
# (1) Determine V to minimize Drag, holding energy constant and holding L = W
#     E, V  -> solve for h, mach, qdyn
#     L = W -> solve for alpha, drag
#     search min(D) using V as control variable.
# (2) Use the glide slope (dh/dV) to determine gamma:
#     dh/dt = V sin(gam) = dh/dV * dV/dt -> solve for gamma
def drag_n1(_h, _e, _mu, _Re, _m, _s_ref, _eta):
    _g0 = _mu / _Re ** 2
    _g = _mu / (_Re + _h)**2
    _v = (2 * (_e - _g * _h)) ** 0.5
    _mach = _v / atm.speed_of_sound(_h)
    _cd0 = float(cd0_table(_mach))
    _cla = float(cl_alpha_table(_mach))
    _weight = _m * _g0
    _qdyn = 0.5 * atm.density(_h) * _v**2

    _alpha = _weight / (_qdyn * _s_ref * _cla)
    _drag = _qdyn * _s_ref * (_cd0 + _eta * _cla * _alpha ** 2)
    return _drag


def fpa_glide(_h, _e, _dh_dv, _mu, _Re, _m, _s_ref, _eta):
    _g = _mu / (_Re + _h) ** 2
    _v = (2 * (_e - _g * _h)) ** 0.5
    _drag = drag_n1(_h, _e, _mu, _Re, _m, _s_ref, _eta)
    _sin_gamma = -_drag * _dh_dv / (_m * (_v + _dh_dv * _g))
    _sin_gamma_sat = min(1, max(-1, _sin_gamma))
    _gamma = np.arcsin(_sin_gamma_sat)
    return _gamma


def alpha_n1(_v, _h, _g0, _m, _s_ref):
    _rho = density(_h)
    _qdyn = 0.5 * _rho * _v**2
    _mach = _v / atm.speed_of_sound(_h)
    _cla = float(cl_alpha_table(_mach))
    _weight = _m * _g0
    _alpha = _weight / (_qdyn * _s_ref * _cla)
    return _alpha


def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


# TODO -- Validate Hamiltonian approach
def get_glide_slope(_mu, _Re, _m, _s_ref, _eta,
                    _e_vals: Optional[np.array] = None,
                    _h_min: float = 0., _h_max: float = 100e3,
                    _mach_min: float = 0.25, _mach_max: float = 0.9):

    # DERIVE GLIDE SLOPE FROM HAMILTONIAN
    # States
    _e_sym = ca.MX.sym('e', 1)
    _h_sym = ca.MX.sym('h', 1)
    _gam_sym = ca.MX.sym('gam', 1)
    _lam_e_sym = ca.MX.sym('lam_e', 1)
    _lam_h_sym = ca.MX.sym('lam_h', 1)
    _lam_gam_sym = ca.MX.sym('lam_gam', 1)

    # Control (Load Factor)
    _u_sym = ca.MX.sym('u', 1)

    # Expressions
    _g_sym = _mu / (_h_sym + _Re) ** 2
    _v_sym = (2 * (_e_sym - _g_sym * _h_sym)) ** 0.5
    _qdyn_sym = 0.5 * dens_fun(_h_sym) * _v_sym ** 2
    _mach_sym = _v_sym / sped_fun(_h_sym)
    _weight_sym = _m * _g_sym
    _ad0_sym = _qdyn_sym * _s_ref * cd0_table(_mach_sym) / _weight_sym
    _adl_sym = _eta * _weight_sym / (_qdyn_sym * _s_ref * cl_alpha_table(_mach_sym))

    # Dynamics
    _dh_dt_sym = _v_sym * ca.sin(_gam_sym)
    _dxn_dt_sym = _v_sym * ca.cos(_gam_sym)
    _de_dt_sym = - _g_sym * _v_sym * (_ad0_sym + _adl_sym * _u_sym ** 2)
    _dgam_dt_sym = _g_sym/_v_sym * (_u_sym - ca.cos(_gam_sym))

    # Hamiltonian
    _path_cost = -_dxn_dt_sym
    _hamiltonian = _path_cost + _lam_e_sym * _de_dt_sym + _lam_h_sym * _dh_dt_sym + _lam_gam_sym * _dgam_dt_sym

    # First-Order Outer Solution
    _lam_e_opt = -1 / (_g_sym * (_ad0_sym + _adl_sym))
    _lam_gam_opt = 2 * _lam_e_opt * _v_sym ** 2 * _adl_sym
    _lam_h_opt = 0

    _dham_dh = ca.jacobian(_hamiltonian, _h_sym)
    _dham_dh_opt = _dham_dh
    _hamiltonian_opt = _hamiltonian
    for _original_arg, _new_arg in zip(
            (_gam_sym, _u_sym, _lam_e_sym, _lam_gam_sym, _lam_h_sym),
            (0, 0, _lam_e_opt, _lam_gam_opt, _lam_h_opt)
    ):
        _dham_dh_opt = ca.substitute(_dham_dh_opt, _original_arg, _new_arg)
        _hamiltonian_opt = ca.substitute(_hamiltonian_opt, _original_arg, _new_arg)

    _dham_dh_fun = ca.Function('dH_dh', (_h_sym, _e_sym), (_dham_dh_opt,), ('h', 'E'), ('dH_dh',))
    _hamiltonian_opt_fun = ca.Function('H', (_h_sym, _e_sym), (_hamiltonian_opt,), ('h', 'E'), ('H',))
    _mach_fun = ca.Function('M', (_h_sym, _e_sym), (_mach_sym,), ('h', 'E'), ('M',))

    _g_max = _mu / (_Re + _h_min) ** 2
    _g_min = _mu / (_Re + _h_max) ** 2

    if _e_vals is None:
        _e_min = _g_max * _h_min + 0.5 * (_mach_min * atm.speed_of_sound(_h_min)) ** 2
        _e_max = _g_min * _h_max + 0.5 * (_mach_max * atm.speed_of_sound(_h_max)) ** 2
        _e_vals = np.linspace(_e_min, _e_max, 1_000)

    _e_direction = np.sign(_e_vals[-1] - _e_vals[0])
    if _e_direction < 0:
        _e_vals = np.flip(_e_vals)
        _e_direction = 1

    _v_vals = np.empty(_e_vals.shape)
    _h_vals = np.empty(_e_vals.shape)
    _drag_vals = np.empty(_e_vals.shape)

    _h_guess = _h_min

    # The glide slope occurs where the del(Hamiltonian)/delh = 0 for the zeroth-order asymptotic expansion
    for idx in range(len(_v_vals)):
        _e_i = _e_vals[idx]

        _h_max_i = min((_e_i - 0.5 * (_mach_min * atm.speed_of_sound(_h_guess))**2) / _g_max, _h_max)
        _h_min_i = max((_e_i - 0.5 * (_mach_max * atm.speed_of_sound(_h_guess))**2) / _g_min, _h_min)

        if _h_min_i > _h_max_i:
            _h_i = _h_max_i
        else:
            _nlp_dict = {'x': _h_sym, 'f': _dham_dh_fun(_h_sym, _e_i)**2, 'g': ca.vcat((_h_sym, _mach_fun(_h_sym, _e_i)))}
            _nlp_solver = ca.nlpsol('Hh0', 'ipopt', _nlp_dict)

            block_print()
            _nlp_sol = _nlp_solver(x0=_h_guess, lbg=(_h_min, _mach_min), ubg=(_h_max_i, _mach_max))
            enable_print()
            # _nlp_dict = {'x': _h_sym, 'f': _h_sym, 'g': ca.vcat((_h_sym, _mach_fun(_h_sym, _e_i), _dham_dh_fun(_h_sym, _e_i)))}
            # _nlp_solver = ca.nlpsol('Hh0', 'ipopt', _nlp_dict)
            #
            # block_print()
            # _nlp_sol = _nlp_solver(x0=_h_guess, lbg=(_h_min, _mach_min, 0), ubg=(_h_max_i, _mach_max, 0))
            # enable_print()
            _h_i = float(min(max(_h_min, _nlp_sol['x']), _h_max_i))

        _h_vals[idx] = _h_i
        _g_i = _mu / (_Re + _h_i) ** 2
        _a_i = atm.speed_of_sound(_h_i)
        _v_i = min(max(_mach_min * _a_i, (2 * (_e_i - _g_i * _h_vals[idx])) ** 0.5), _mach_max * _a_i)
        _v_vals[idx] = _v_i
        _e_i_corrected = _g_i * _h_i + 0.5 * _v_i ** 2
        _e_vals[idx] = _e_i_corrected
        _drag_vals[idx] = drag_n1(_h_i, _e_i_corrected, _mu, _Re, _m, _s_ref, _eta)
        _h_guess = _h_i

    # Remove values decreasing values (these are artifacts of the temperature interpolant)
    _valid_idces = []
    idx_last = len(_e_vals) - 1
    for idx, (_e_val, _h_val, _v_val) in enumerate(zip(_e_vals, _h_vals, _v_vals)):
        if idx < idx_last:
            if np.sign(_e_vals[idx + 1] - _e_val) == _e_direction \
                    and np.sign(_v_vals[idx + 1] - _v_val) != -_e_direction \
                    and np.sign(_h_vals[idx + 1] - _h_val) != -_e_direction:
                _valid_idces.append(idx)
        else:
            if np.sign(_e_val - _e_vals[idx - 1]) == _e_direction \
                    and np.sign(_v_val - _v_vals[idx - 1]) != -_e_direction \
                    and np.sign(_h_val - _h_vals[idx - 1]) != -_e_direction:
                _valid_idces.append(idx)

    _e_vals = _e_vals[_valid_idces]
    _v_vals = _v_vals[_valid_idces]
    _h_vals = _h_vals[_valid_idces]
    _drag_vals = _drag_vals[_valid_idces]

    h_interp = sp.interpolate.pchip(_e_vals, _h_vals)
    v_interp = sp.interpolate.pchip(_e_vals, _v_vals)
    drag_interp = sp.interpolate.pchip(_e_vals, _drag_vals)

    _gam_vals = np.empty(_e_vals.shape)

    idx_last = len(_gam_vals) - 1

    for idx, (_v_val, _e_val, _h_val) in enumerate(zip(_v_vals, _e_vals, _h_vals)):
        if idx < idx_last:
            _v_pert = _v_vals[idx + 1]
            _h_pert = _h_vals[idx + 1]
        else:
            _v_pert = _v_vals[idx - 1]
            _h_pert = _v_vals[idx - 1]
        _dh_dv = (_h_val - _h_pert) / (_v_val - _v_pert)
        _gam_vals[idx] = fpa_glide(_h_val, _e_val, _dh_dv, _mu, _Re, _m, _s_ref, _eta)

    gam_interp = sp.interpolate.pchip(_e_vals, _gam_vals)

    return h_interp, v_interp, gam_interp, drag_interp
