import numpy as np
import scipy as sp
from typing import Optional

from lookup_tables import cl_alpha_table, cd0_table, atm


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
def drag_n1(_v, _e, _g0, _m, _s_ref, _eta):
    _h = (_e - 0.5 * _v**2) / _g0
    _mach = _v / atm.speed_of_sound(_h)
    _cd0 = float(cd0_table(_mach))
    _cla = float(cl_alpha_table(_mach))
    _weight = _m * _g0
    _qdyn = 0.5 * atm.density(_h) * _v**2

    _alpha = _weight / (_qdyn * _s_ref * _cla)
    _drag = _qdyn * _s_ref * (_cd0 + _eta * _cla * _alpha ** 2)
    return _drag


def fpa_glide(_v, _e, _dh_dv, _g0, _m, _s_ref, _eta):
    _drag = drag_n1(_v, _e, _g0, _m, _s_ref, _eta)
    _gamma = - np.arcsin(_drag * _dh_dv / (_m * (_v + _dh_dv * _g0)))
    return _gamma


def alpha_n1(_v, _h, _g0, _m, _s_ref):
    _rho = density(_h)
    _qdyn = 0.5 * _rho * _v**2
    _mach = _v / atm.speed_of_sound(_h)
    _cla = float(cl_alpha_table(_mach))
    _weight = _m * _g0
    _alpha = _weight / (_qdyn * _s_ref * _cla)
    return _alpha


def get_glide_slope(_g0, _m, _s_ref, _eta, _e_vals: Optional[np.array] = None):
    if _e_vals is None:
        _e_min = float(0.5 * (0.6 * speed_of_sound(0.)) ** 2)
        _e_max = float(0.5 * (3.0 * speed_of_sound(0.)) ** 2)
        _e_vals = np.linspace(_e_min, _e_max, 1_000)

    _v_vals = np.empty(_e_vals.shape)
    _h_vals = np.empty(_e_vals.shape)
    _drag_vals = np.empty(_e_vals.shape)

    _v_guess = (2 * _e_vals[0]) ** 0.5

    for idx in range(len(_v_vals)):
        _e_i = _e_vals[idx]
        _sol = sp.optimize.minimize(lambda v_i: drag_n1(v_i, _e_i, _g0, _m, _s_ref, _eta), _v_guess)
        _v_vals[idx] = _sol.x[0]
        _drag_vals[idx] = drag_n1(_v_vals[idx], _e_vals[idx], _g0, _m, _s_ref, _eta)
        _v_guess = _sol.x[0]
        _h_vals[idx] = (_e_i - 0.5 * _v_vals[idx] ** 2) / _g0

    # Remove values decreasing values (these are artifacts of the temperature interpolant)
    _valid_idces = np.where(np.logical_and(np.diff(_v_vals) > 0, np.diff(_h_vals) > 0))
    _e_vals = _e_vals[_valid_idces]
    _v_vals = _v_vals[_valid_idces]
    _h_vals = _h_vals[_valid_idces]
    _drag_vals = _drag_vals[_valid_idces]

    h_interp = sp.interpolate.pchip(_e_vals, _h_vals)
    v_interp = sp.interpolate.pchip(_e_vals, _v_vals)
    drag_interp = sp.interpolate.pchip(_e_vals, _drag_vals)
    h_interp_v = sp.interpolate.pchip(_v_vals, _h_vals)

    _gam_vals = np.empty(_e_vals.shape)

    for idx, (_v_val, _e_val, _h_val) in enumerate(zip(_v_vals, _e_vals, _h_vals)):
        _v_pert = _v_val + 10
        _dh_dv = (h_interp_v(_v_val) - h_interp_v(_v_pert)) / (_v_val - _v_pert)
        _gam_vals[idx] = fpa_glide(_v_val, _e_val, _dh_dv, _g0, _m, _s_ref, _eta)

    gam_interp = sp.interpolate.pchip(_e_vals, _gam_vals)

    return h_interp, v_interp, gam_interp, drag_interp
