import os
import sys
from typing import Optional

import numpy as np
import scipy as sp
import casadi as ca

from lookup_tables import cl_alpha_fun, cd0_fun, atm, dens_fun, sped_fun


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
    _cd0 = float(cd0_fun(_mach))
    _cla = float(cl_alpha_fun(_mach))
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
    _cla = float(cl_alpha_fun(_mach))
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
    _ad0_sym = _qdyn_sym * _s_ref * cd0_fun(_mach_sym) / _weight_sym
    _adl_sym = _eta * _weight_sym / (_qdyn_sym * _s_ref * cl_alpha_fun(_mach_sym))

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
    _d2ham_dh2 = ca.jacobian(_dham_dh, _h_sym)

    _dham_dh_opt = _dham_dh
    _d2ham_dh2_opt = _d2ham_dh2
    _hamiltonian_opt = _hamiltonian
    for _original_arg, _new_arg in zip(
            (_gam_sym, _u_sym, _lam_e_sym, _lam_gam_sym, _lam_h_sym),
            (0, 1., _lam_e_opt, _lam_gam_opt, _lam_h_opt)
    ):
        _dham_dh_opt = ca.substitute(_dham_dh_opt, _original_arg, _new_arg)
        _d2ham_dh2_opt = ca.substitute(_d2ham_dh2_opt, _original_arg, _new_arg)
        _hamiltonian_opt = ca.substitute(_hamiltonian_opt, _original_arg, _new_arg)

    _dham_dh_fun = ca.Function('dH_dh', (_h_sym, _e_sym), (_dham_dh_opt,), ('h', 'E'), ('dH_dh',))
    _d2ham_dh2_fun = ca.Function('d2H_dh2', (_h_sym, _e_sym), (_d2ham_dh2_opt,), ('h', 'E'), ('d2H_dh2',))
    _hamiltonian_opt_fun = ca.Function('H', (_h_sym, _e_sym), (_hamiltonian_opt,), ('h', 'E'), ('H',))
    _mach_fun = ca.Function('M', (_h_sym, _e_sym), (_mach_sym,), ('h', 'E'), ('M',))

    _g_max = _mu / (_Re + _h_min) ** 2
    _g_min = _mu / (_Re + _h_max) ** 2

    if _e_vals is None:
        # Space based on altitude range (note: steps not linear w.r.t. h)
        _n_e_vals = int(np.ceil((_h_max - _h_min)/100.))

        _e_min = _g_max * _h_min + 0.5 * (_mach_min * atm.speed_of_sound(_h_min)) ** 2
        _e_max = _g_min * _h_max + 0.5 * (_mach_max * atm.speed_of_sound(_h_max)) ** 2
        _e_vals = np.linspace(_e_min, _e_max, _n_e_vals)

    _e_direction = np.sign(_e_vals[-1] - _e_vals[0])
    if _e_direction < 0:
        _e_vals = np.flip(_e_vals)
        _e_direction = 1

    _v_vals = np.empty(_e_vals.shape)
    _h_vals = np.empty(_e_vals.shape)
    _drag_vals = np.empty(_e_vals.shape)

    _h_guess = _h_min

    # The glide slope occurs where the del(Hamiltonian)/delh = 0 for the zeroth-order asymptotic expansion
    idx = 0

    for idx in range(len(_v_vals)):
        _e_i = _e_vals[idx]

        _h_min_i = _h_guess  # altitude should be monotonically increasing

        # Glide slope occures where (in asymptotic expansion) the the Hamiltonian is stationary w.r.t. altitude
        _fsolve_sol = sp.optimize.fsolve(
            func=lambda _h_trial: np.asarray(_dham_dh_fun(_h_trial, _e_i)).flatten(),
            x0=np.asarray((_h_guess,)),
            fprime=lambda _h_trial: np.asarray(_d2ham_dh2_fun(_h_trial, _e_i)).flatten()
        )

        # Bound altitude
        _h_i = max(min(float(_fsolve_sol[0]), _h_max), _h_min_i)

        _g_i = _mu / (_Re + _h_i) ** 2
        _a_i = atm.speed_of_sound(_h_i)
        _v_i = (2 * (_e_i - _g_i * _h_i)) ** 0.5
        _mach_i = _v_i / _a_i

        if _mach_i > 3.0:
            print(f'Mach = {_mach_i}')

        # Ensure velocity is within bounds, i.e. applying bounds does not change energy
        if _mach_i < _mach_min or _mach_i > _mach_max:
            break

        # Assign values if valid
        _h_vals[idx] = _h_i
        _v_vals[idx] = _v_i
        _drag_vals[idx] = drag_n1(_h_i, _e_i, _mu, _Re, _m, _s_ref, _eta)
        _h_guess = _h_i

    # Remove invalid values where energy exceeds altitude/Mach bounds
    _e_vals = _e_vals[:idx]
    _h_vals = _h_vals[:idx]
    _v_vals = _v_vals[:idx]
    _drag_vals = _drag_vals[:idx]

    _h_interp = sp.interpolate.pchip(_e_vals, _h_vals)
    _v_interp = sp.interpolate.pchip(_e_vals, _v_vals)
    _drag_interp = sp.interpolate.pchip(_e_vals, _drag_vals)

    _gam_vals = np.empty(_e_vals.shape)

    idx_last = len(_gam_vals) - 1

    for idx, (_v_val, _e_val, _h_val) in enumerate(zip(_v_vals, _e_vals, _h_vals)):
        if idx < idx_last:
            _v_pert = _v_vals[idx + 1]
            _h_pert = _h_vals[idx + 1]
        else:
            _v_pert = _v_vals[idx - 1]
            _h_pert = _h_vals[idx - 1]
        _dh_dv = (_h_val - _h_pert) / (_v_val - _v_pert)
        _gam_vals[idx] = fpa_glide(_h_val, _e_val, _dh_dv, _mu, _Re, _m, _s_ref, _eta)

    _gam_interp = sp.interpolate.pchip(_e_vals, _gam_vals)

    return _h_interp, _v_interp, _gam_interp, _drag_interp


if __name__=='__main__':
    from matplotlib import pyplot as plt
    import pickle

    with open('sol_set_range.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

    k_dict = {}
    x_dict = {}
    lam_dict = {}
    u_dict = {}

    for key, val in zip(sol.annotations.constants, sol.k):
        k_dict[key] = val
    for key, x_val, lam_val in zip(sol.annotations.states, list(sol.x), list(sol.lam)):
        x_dict[key] = x_val
        lam_dict[key] = lam_val
    for key, val in zip(sol.annotations.controls, list(sol.u)):
        u_dict[key] = val

    h_min = 0.
    h_max = 130e3
    g_max = k_dict['mu'] / (k_dict['Re'] + h_min) ** 2
    g_min = k_dict['mu'] / (k_dict['Re'] + h_max) ** 2
    mach_min = 0.25
    mach_max = 3.0

    h_interp, v_interp, gam_interp, drag_interp = get_glide_slope(
        k_dict['mu'], k_dict['Re'], x_dict['m'][0], k_dict['s_ref'], k_dict['eta'],
        _h_min=h_min, _h_max=h_max, _mach_min=mach_min, _mach_max=mach_max)

    e_vals = h_interp.x
    h_vals = h_interp(e_vals)
    v_vals = v_interp(e_vals)
    mach_vals = v_vals / np.asarray(sped_fun(h_vals)).flatten()
    gam_vals = gam_interp(e_vals)
    drag_vals = drag_interp(e_vals)
    g_vals = k_dict['mu'] / (k_dict['Re'] + h_vals) ** 2
    weight_vals = x_dict['m'][0] * g_vals

    # Plot Interpolants
    e_lab = r'$E$ [ft$^2$/s$^2$]'
    mach_lab = 'Mach'
    fig = plt.figure()

    ax_h = fig.add_subplot(221)
    ax_h.grid()
    ax_h.set_xlabel(e_lab)
    ax_h.set_ylabel(r'$h$ [1,000 ft]')
    ax_h.plot(e_vals, h_vals * 1e-3)

    ax_hv = fig.add_subplot(222)
    ax_hv.grid()
    ax_hv.set_xlabel(mach_lab)
    ax_hv.set_ylabel(r'$h$ [1,000 ft]')
    ax_hv.plot(mach_vals, h_vals * 1e-3)
    ax_hv.plot(np.array((mach_min, mach_max)),
               np.array((h_min, h_min)) * 1e-3,
               'k--')
    ax_hv.plot(np.array((mach_max, mach_max)),
               np.array((h_min, h_max)) * 1e-3,
               'k--')
    ax_hv.plot(np.array((mach_max, mach_min)),
               np.array((h_max, h_max)) * 1e-3,
               'k--')
    ax_hv.plot(np.array((mach_min, mach_min)),
               np.array((h_max, h_min)) * 1e-3,
               'k--')

    ax_gam = fig.add_subplot(223)
    ax_gam.grid()
    ax_gam.set_xlabel(mach_lab)
    ax_gam.set_ylabel(r'$\gamma$ [deg]')
    ax_gam.plot(mach_vals, gam_vals * 180/np.pi)

    ax_drag = fig.add_subplot(224)
    ax_drag.grid()
    ax_drag.set_xlabel(mach_lab)
    ax_drag.set_ylabel(r'$D$ [g]')
    ax_drag.plot(mach_vals, drag_vals / weight_vals)

    fig.tight_layout()

    plt.show()
