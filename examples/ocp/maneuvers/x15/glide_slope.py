import os
import sys
from typing import Optional

import numpy as np
import scipy as sp
import casadi as ca

from x15_aero_model import cd0_fun, cdl_fun, cla_fun, s_ref, weight_empty
from x15_atmosphere import atm, dens_fun, sped_fun, mu, Re, g0


# Find Glide Slope (assuming small FPA)
# (1) Determine V to minimize Drag, holding energy constant and holding L = W
#     E, V  -> solve for h, mach, qdyn
#     L = W -> solve for alpha, drag
#     search min(D) using V as control variable.
# (2) Use the glide slope (dh/dV) to determine gamma:
#     dh/dt = V sin(gam) = dh/dV * dV/dt -> solve for gamma
def drag_n1(_h, _e, _m):
    _g = mu / (Re + _h)**2
    _v = (2 * (_e - _g * _h)) ** 0.5
    _mach = _v / atm.speed_of_sound(_h)
    _cd0 = float(cd0_fun(_mach))
    _cla = float(cla_fun(_mach))
    _cdl = float(cdl_fun(_mach))
    _weight = _m * g0
    _qdyn = 0.5 * atm.density(_h) * _v**2

    _alpha = _weight / (_qdyn * s_ref * _cla)
    _drag = _qdyn * s_ref * (_cd0 + _cdl * (_cla * _alpha) ** 2)
    return _drag


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_glide_slope(_m, _e_vals: Optional[np.array] = None, _h_guess0: Optional[float] = None,
                    _h_min: float = 0., _h_max: float = 100e3,
                    _mach_min: float = 0.25, _mach_max: float = 0.9, independent_var: Optional[str] = 'e'):

    # DERIVE GLIDE SLOPE FROM HAMILTONIAN
    # States
    _e_sym = ca.SX.sym('e', 1)
    _h_sym = ca.SX.sym('h', 1)
    _gam_sym = ca.SX.sym('gam', 1)
    _lam_e_sym = ca.SX.sym('lam_e', 1)
    _lam_h_sym = ca.SX.sym('lam_h', 1)
    _lam_gam_sym = ca.SX.sym('lam_gam', 1)

    # Control (Load Factor)
    _u_sym = ca.SX.sym('u', 1)

    # Expressions
    _g_sym = mu / (_h_sym + Re) ** 2
    _v_sym = (2 * (_e_sym - _g_sym * _h_sym)) ** 0.5
    _qdyn_sym = 0.5 * dens_fun(_h_sym) * _v_sym ** 2
    _mach_sym = _v_sym / sped_fun(_h_sym)
    _weight_sym = _m * _g_sym
    _ad0_sym = _qdyn_sym * s_ref * cd0_fun(_mach_sym) / _weight_sym
    _adl_sym = cdl_fun(_mach_sym) * _weight_sym / (_qdyn_sym * s_ref)

    # Dynamics
    _dh_dt_sym = _v_sym * ca.sin(_gam_sym)
    _dxn_dt_sym = _v_sym * ca.cos(_gam_sym)
    _de_dt_sym = - _g_sym * _v_sym * (_ad0_sym + _adl_sym * _u_sym ** 2)
    _dgam_dt_sym = _g_sym/_v_sym * (_u_sym - ca.cos(_gam_sym))

    # Hamiltonian
    _path_cost = -_dxn_dt_sym
    _hamiltonian = _path_cost + _lam_e_sym * _de_dt_sym + _lam_h_sym * _dh_dt_sym + _lam_gam_sym * _dgam_dt_sym

    # Zeroth-Order Outer Solution
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

    _g_max = mu / (Re + _h_min) ** 2
    _g_min = mu / (Re + _h_max) ** 2

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

    if _h_guess0 is not None:
        _h_guess = _h_guess0
    else:
        _h_guess = _h_min

    # The glide slope occurs where the del(Hamiltonian)/delh = 0 for the zeroth-order asymptotic expansion
    idx0 = 0
    idxf = 0

    for idx in range(len(_e_vals)):
        _e_i = _e_vals[idx]

        # Glide slope occurs where (in asymptotic expansion) the the Hamiltonian is stationary w.r.t. altitude
        _fsolve_sol = sp.optimize.fsolve(
            func=lambda _h_trial: np.asarray(_dham_dh_fun(_h_trial, _e_i)).flatten(),
            x0=np.asarray((_h_guess,)),
            fprime=lambda _h_trial: np.asarray(_d2ham_dh2_fun(_h_trial, _e_i)).flatten()
        )

        # Altitude should be monotonically increasing
        _h_min_i = _h_guess
        _h_i = max(min(float(_fsolve_sol[0]), _h_max), _h_min_i)

        # Altitude should produce nonnegative velocity
        _g_i = mu / (Re + _h_i) ** 2
        _h_max_i = min(_e_i / _g_i, _h_max)
        _h_i = min(_h_i, _h_max_i)

        _g_i = mu / (Re + _h_i) ** 2
        _a_i = atm.speed_of_sound(_h_i)
        _v_i = (2 * (_e_i - _g_i * _h_i)) ** 0.5
        _mach_i = _v_i / _a_i

        # Ensure velocity is within bounds, i.e. applying bounds does not change energy
        if _mach_i < _mach_min:
            idx0 = idx + 1
            continue
        elif _mach_i > _mach_max:
            break
        else:
            idxf = idx

        # Assign values if valid
        _h_vals[idx] = _h_i
        _v_vals[idx] = _v_i
        _drag_vals[idx] = drag_n1(_h_i, _e_i, _m)
        _h_guess = _h_i

    # Remove invalid values where energy exceeds altitude/Mach bounds
    _e_vals = _e_vals[idx0:idxf+1]
    _h_vals = _h_vals[idx0:idxf+1]
    _v_vals = _v_vals[idx0:idxf+1]
    _drag_vals = _drag_vals[idx0:idxf+1]

    _gam_vals = np.empty(_e_vals.shape)

    idx_last = len(_gam_vals) - 1

    for idx, (_h_val, _e_val, _drag_val) in enumerate(zip(_h_vals, _e_vals, _drag_vals)):
        if idx == 0:  # Forward difference
            _dh_dE = (_h_vals[idx + 1] - _h_val) / (_e_vals[idx + 1] - _e_val)
        elif idx == idx_last:  # Backward difference
            _dh_dE = (_h_val - _h_vals[idx - 1]) / (_e_val - _e_vals[idx - 1])
        else:  # Central difference
            _dh_dE = (_h_vals[idx + 1] - _h_vals[idx - 1]) / (_e_vals[idx + 1] - _e_vals[idx - 1])

        _gam_vals[idx] = - np.sin(_dh_dE * _drag_val / _m)

    independent_var_lower = independent_var.lower()
    if independent_var_lower in ('v', 'velocity'):
        _h_interp = sp.interpolate.pchip(_v_vals, _h_vals)
        _v_interp = sp.interpolate.pchip(_e_vals, _v_vals)
        _drag_interp = sp.interpolate.pchip(_v_vals, _drag_vals)
        _gam_interp = sp.interpolate.pchip(_v_vals, _gam_vals)
    elif independent_var_lower in ('m', 'mach'):
        _mach_vals = _v_vals / np.asarray(sped_fun(_h_vals)).flatten()

        _h_interp = sp.interpolate.pchip(_mach_vals, _h_vals)
        _v_interp = sp.interpolate.pchip(_mach_vals, _v_vals)
        _drag_interp = sp.interpolate.pchip(_mach_vals, _drag_vals)
        _gam_interp = sp.interpolate.pchip(_mach_vals, _gam_vals)
    elif independent_var in ('h', 'altitude'):
        # Indices prior to this one are not monotonically increasing
        _last_idx_h0 = max(np.where(_h_vals > _h_vals[0])[0][0] - 1, 0)

        _h_interp = sp.interpolate.pchip(_e_vals[_last_idx_h0:], _h_vals[_last_idx_h0:])
        _v_interp = sp.interpolate.pchip(_h_vals[_last_idx_h0:], _v_vals[_last_idx_h0:])
        _drag_interp = sp.interpolate.pchip(_h_vals[_last_idx_h0:], _drag_vals[_last_idx_h0:])
        _gam_interp = sp.interpolate.pchip(_h_vals[_last_idx_h0:], _gam_vals[_last_idx_h0:])
    else:  # Default to Energy
        _h_interp = sp.interpolate.pchip(_e_vals, _h_vals)
        _v_interp = sp.interpolate.pchip(_e_vals, _v_vals)
        _drag_interp = sp.interpolate.pchip(_e_vals, _drag_vals)
        _gam_interp = sp.interpolate.pchip(_e_vals, _gam_vals)

    return _h_interp, _v_interp, _gam_interp, _drag_interp


def get_glide_slope_neighboring_feedback(_m, _h_interp):
    # DERIVE GLIDE SLOPE FROM HAMILTONIAN
    # States
    _e_sym = ca.SX.sym('e', 1)
    _h_sym = ca.SX.sym('h', 1)
    _gam_sym = ca.SX.sym('gam', 1)
    _lam_e_sym = ca.SX.sym('lam_e', 1)
    _lam_h_sym = ca.SX.sym('lam_h', 1)
    _lam_gam_sym = ca.SX.sym('lam_gam', 1)

    # Control (Load Factor)
    _u_sym = ca.SX.sym('u', 1)

    # Expressions
    _g_sym = mu / (_h_sym + Re) ** 2
    _v_sym = (2 * (_e_sym - _g_sym * _h_sym)) ** 0.5
    _qdyn_sym = 0.5 * dens_fun(_h_sym) * _v_sym ** 2
    _mach_sym = _v_sym / sped_fun(_h_sym)
    _weight_sym = _m * _g_sym
    _ad0_sym = _qdyn_sym * s_ref * cd0_fun(_mach_sym) / _weight_sym
    _adl_sym = cdl_fun(_mach_sym) * _weight_sym / (_qdyn_sym * s_ref)

    # Dynamics
    _dh_dt_sym = _v_sym * ca.sin(_gam_sym)
    _dxn_dt_sym = _v_sym * ca.cos(_gam_sym)
    _de_dt_sym = - _g_sym * _v_sym * (_ad0_sym + _adl_sym * _u_sym ** 2)
    _dgam_dt_sym = _g_sym/_v_sym * (_u_sym - ca.cos(_gam_sym))

    # Hamiltonian
    _path_cost = -_dxn_dt_sym
    _hamiltonian = _path_cost + _lam_e_sym * _de_dt_sym + _lam_h_sym * _dh_dt_sym + _lam_gam_sym * _dgam_dt_sym

    # ARE Parameters
    _x_sym = ca.vcat((_h_sym, _gam_sym))
    _ham_x = ca.jacobian(_hamiltonian, _x_sym)
    _ham_u = ca.jacobian(_hamiltonian, _u_sym)

    _ham_xx = ca.jacobian(_ham_x, _x_sym)
    _ham_xu = ca.jacobian(_ham_x, _u_sym)
    _ham_uu = ca.jacobian(_ham_u, _u_sym)

    _f = ca.vcat((_dh_dt_sym, _dgam_dt_sym))
    _f_x = ca.jacobian(_f, _x_sym)
    _f_u = ca.jacobian(_f, _u_sym)

    A_sym = _f_x
    B_sym = _f_u
    Q_sym = _ham_xx
    N_sym = _ham_xu
    R_sym = _ham_uu

    # Zeroth-Order Outer Solution
    _lam_e_opt = -1 / (_g_sym * (_ad0_sym + _adl_sym))
    _lam_gam_opt = 2 * _lam_e_opt * _v_sym ** 2 * _adl_sym
    _lam_h_opt = 0

    A_sym_opt = A_sym
    B_sym_opt = B_sym
    Q_sym_opt = Q_sym
    N_sym_opt = N_sym
    R_sym_opt = R_sym

    for _original_arg, _new_arg in zip(
            (_gam_sym, _u_sym, _lam_e_sym, _lam_gam_sym, _lam_h_sym),
            (0, 1., _lam_e_opt, _lam_gam_opt, _lam_h_opt)
    ):
        A_sym_opt = ca.substitute(A_sym_opt, _original_arg, _new_arg)
        B_sym_opt = ca.substitute(B_sym_opt, _original_arg, _new_arg)
        Q_sym_opt = ca.substitute(Q_sym_opt, _original_arg, _new_arg)
        N_sym_opt = ca.substitute(N_sym_opt, _original_arg, _new_arg)
        R_sym_opt = ca.substitute(R_sym_opt, _original_arg, _new_arg)


    A_fun = ca.Function('A', (_h_sym, _e_sym), (A_sym_opt,), ('h', 'E'), ('A',))
    B_fun = ca.Function('B', (_h_sym, _e_sym), (B_sym_opt,), ('h', 'E'), ('B',))
    Q_fun = ca.Function('Q', (_h_sym, _e_sym), (Q_sym_opt,), ('h', 'E'), ('Q',))
    N_fun = ca.Function('N', (_h_sym, _e_sym), (N_sym_opt,), ('h', 'E'), ('N',))
    R_fun = ca.Function('R', (_h_sym, _e_sym), (R_sym_opt,), ('h', 'E'), ('R',))

    _e_vals = _h_interp.x
    _h_vals = _h_interp(_e_vals)

    _k_h_vals = np.empty(_e_vals.shape)
    _k_gam_vals = np.empty(_e_vals.shape)

    for idx, (_h_val, _e_val) in enumerate(zip(_h_vals, _e_vals)):
        _a = np.asarray(A_fun(_h_val, _e_val))
        _b = np.asarray(B_fun(_h_val, _e_val))
        _q = np.asarray(Q_fun(_h_val, _e_val))
        _r = np.asarray(R_fun(_h_val, _e_val))
        _n = np.asarray(N_fun(_h_val, _e_val))
        _p = sp.linalg.solve_continuous_are(
            a=_a,
            b=_b,
            q=_q,
            r=_r,
            s=_n
        )
        _k = np.linalg.solve(_r, _b.T @ _p + _n.T)
        _k_h_vals[idx] = _k[0, 0]
        _k_gam_vals[idx] = _k[0, 1]

    _k_h_interp = sp.interpolate.pchip(_e_vals, _k_h_vals)
    _k_gam_interp = sp.interpolate.pchip(_e_vals, _k_gam_vals)

    return _k_h_interp, _k_gam_interp


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    h_min = 0.
    h_max = 130e3
    g_max = mu / (Re + h_min) ** 2
    g_min = mu / (Re + h_max) ** 2
    mach_min = 0.25
    mach_max = 7.0

    mass = weight_empty / g0

    h_interp, v_interp, gam_interp, drag_interp = get_glide_slope(
        mass, _h_min=h_min, _h_max=h_max, _mach_min=mach_min, _mach_max=mach_max
    )

    k_h_interp, k_gam_interp = get_glide_slope_neighboring_feedback(mass, h_interp)

    e_vals = h_interp.x
    h_vals = h_interp(e_vals)
    v_vals = v_interp(e_vals)
    k_h_vals = k_h_interp(e_vals)
    k_gam_vals = k_gam_interp(e_vals)
    mach_vals = v_vals / np.asarray(sped_fun(h_vals)).flatten()
    gam_vals = gam_interp(e_vals)
    drag_vals = drag_interp(e_vals)
    g_vals = mu / (Re + h_vals) ** 2
    weight_vals = mass * g_vals

    # Plot Interpolants
    e_lab = r'$E$ [ft$^2$/s$^2$]'
    mach_lab = 'Mach'
    fig_glideslope = plt.figure()

    ax_h = fig_glideslope.add_subplot(221)
    ax_h.grid()
    ax_h.set_xlabel(e_lab)
    ax_h.set_ylabel(r'$h$ [1,000 ft]')
    ax_h.plot(e_vals, h_vals * 1e-3)

    ax_hv = fig_glideslope.add_subplot(222)
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

    ax_gam = fig_glideslope.add_subplot(223)
    ax_gam.grid()
    ax_gam.set_xlabel(mach_lab)
    ax_gam.set_ylabel(r'$\gamma$ [deg]')
    ax_gam.plot(mach_vals, gam_vals * 180/np.pi)

    ax_drag = fig_glideslope.add_subplot(224)
    ax_drag.grid()
    ax_drag.set_xlabel(mach_lab)
    ax_drag.set_ylabel(r'$D$ [g]')
    ax_drag.plot(mach_vals, drag_vals / weight_vals)

    fig_glideslope.tight_layout()

    fig_gains = plt.figure()

    ax_k_h = fig_gains.add_subplot(211)
    ax_k_h.grid()
    ax_k_h.set_ylabel(r'$k_h$')
    ax_k_h.plot(mach_vals, k_h_vals)

    ax_k_gam = fig_gains.add_subplot(212)
    ax_k_gam.grid()
    ax_k_gam.set_xlabel(mach_lab)
    ax_k_gam.set_ylabel(r'$k_\gamma$')
    ax_k_gam.plot(mach_vals, k_gam_vals)

    fig_gains.tight_layout()

    plt.show()
