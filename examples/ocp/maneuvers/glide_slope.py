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


# TODO -- Refactor to calculate h*, V* via Hamiltonian: d(H)/dh = 0 @ glide slope for L = -dx/dt, x = [E, h, gam]'
def get_glide_slope(_mu, _Re, _m, _s_ref, _eta,
                    _e_vals: Optional[np.array] = None,
                    _h_min: float = 0., _h_max: float = np.inf, _mach_max: float = np.inf):
    _g_max = _mu / (_Re + _h_min) ** 2

    if _e_vals is None:
        _e_min = _g_max * _h_min

        if not np.isinf(_h_max) and not np.isinf(_mach_max):
            _e_max = _mu / (_Re + _h_max) ** 2 * _h_max + 0.5 * (_mach_max * atm.speed_of_sound(_h_max)) ** 2
        elif not np.isinf(_h_max):
            _e_max = _mu / (_Re + _h_max) ** 2 * _h_max + 0.5 * (1.0 * atm.speed_of_sound(_h_max)) ** 2
        elif not np.isinf(_mach_max):
            _e_max = _mu / (_Re + 1e4) ** 2 * 1e4 + 0.5 * (_mach_max * atm.speed_of_sound(1e4)) ** 2
        else:
            _e_max = _mu / (_Re + 1e4) ** 2 * 1e4 + 0.5 * (1.0 * atm.speed_of_sound(1e4)) ** 2

        _e_vals = np.linspace(_e_min, _e_max, 1_000)

    _e_direction = np.sign(_e_vals[-1] - _e_vals[0])
    if _e_direction < 0:
        _e_vals = np.flip(_e_vals)
        _e_direction = 1

    _v_vals = np.empty(_e_vals.shape)
    _h_vals = np.empty(_e_vals.shape)
    _drag_vals = np.empty(_e_vals.shape)

    _h_guess = 0.5 * _e_vals[0] / _g_max

    for idx in range(len(_v_vals)):
        _e_i = _e_vals[idx]
        _sol = sp.optimize.minimize(lambda _hi: drag_n1(_hi, _e_i, _mu, _Re, _m, _s_ref, _eta), _h_guess)
        _h_vals[idx] = max(_sol.x[0], _h_min)
        _g_i = _mu / (_Re + _h_vals[idx]) ** 2
        _a_i = atm.speed_of_sound(_h_vals[idx])
        _v_vals[idx] = min((2 * (_e_i - _g_i * _h_vals[idx])) ** 0.5, _a_i * _mach_max)
        _e_i_corrected = _g_i * _h_vals[idx] + 0.5 * _v_vals[idx] ** 2
        _e_vals[idx] = _e_i_corrected
        _drag_vals[idx] = drag_n1(_h_vals[idx], _e_i_corrected, _mu, _Re, _m, _s_ref, _eta)
        _h_guess = _h_vals[idx]

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
