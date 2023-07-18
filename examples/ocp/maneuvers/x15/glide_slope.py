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
    _weight = _m * _g
    _qdyn = 0.5 * atm.density(_h) * _v**2

    _cl = _weight / (_qdyn * s_ref)
    _drag = _qdyn * s_ref * (_cd0 + _cdl * _cl ** 2)
    return _drag


h_SX = ca.SX.sym('h')
dens_expr = dens_fun(h_SX)
drho_dh_expr = ca.jacobian(dens_expr, h_SX)
drho_dh_fun = ca.Function('drho_dh', (h_SX,), (drho_dh_expr,), ('h',), ('drho_dh',))


def drho_dh(_h):
    _drho_dh = np.asarray(drho_dh_fun(_h)).flatten()
    if len(_drho_dh) == 1:
        _drho_dh = _drho_dh[0]
    return _drho_dh


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_glide_slope(_m, _e_vals: Optional[np.array] = None,
                    _h_guess0: Optional[float] = None, _gam_guess0: Optional[float] = None,
                    _h_min: float = 0., _h_max: float = 100e3,
                    _gam_min: float = -80.*np.pi/180., _gam_max: float = 0.,
                    _mach_min: float = 0.25, _mach_max: float = 0.9, independent_var: Optional[str] = 'e',
                    _use_qdyn_expansion: bool = False):

    # DERIVE GLIDE SLOPE FROM HAMILTONIAN
    # States
    _e_sym = ca.SX.sym('e', 1)
    _h_sym = ca.SX.sym('h', 1)
    _gam_sym = ca.SX.sym('gam', 1)
    _lam_e_sym = ca.SX.sym('lam_e', 1)
    _lam_h_sym = ca.SX.sym('lam_h', 1)
    _lam_qdyn_sym = ca.SX.sym('lam_qdyn', 1)
    _lam_gam_sym = ca.SX.sym('lam_gam', 1)

    # Control (Load Factor)
    _u_sym = ca.SX.sym('u', 1)

    # Expressions
    _g_sym = mu / (_h_sym + Re) ** 2
    _v_sym = (2 * (_e_sym - _g_sym * _h_sym)) ** 0.5
    _rho_sym = dens_fun(_h_sym)
    _qdyn_sym = 0.5 * _rho_sym * _v_sym ** 2
    _mach_sym = _v_sym / sped_fun(_h_sym)
    _weight_sym = _m * _g_sym
    _ad0_sym = _qdyn_sym * s_ref * cd0_fun(_mach_sym) / _weight_sym
    _adl_sym = cdl_fun(_mach_sym) * _weight_sym / (_qdyn_sym * s_ref)
    _ad_sym = _ad0_sym + _adl_sym * _u_sym ** 2

    def d_dqdyn(_arg):
        return ca.jacobian(_arg, _h_sym) * ca.jacobian(_qdyn_sym, _h_sym) \
               + ca.jacobian(_arg, _e_sym) * ca.jacobian(_qdyn_sym, _e_sym)

    # Dynamics
    _dh_dt_sym = _v_sym * ca.sin(_gam_sym)
    _dv_dt_sym = -_g_sym * (_ad_sym + ca.sin(_gam_sym))
    _dg_dt_sym = ca.jacobian(_g_sym, _h_sym)[0] * _dh_dt_sym
    _dxn_dt_sym = _v_sym * ca.cos(_gam_sym)
    _de_dt_sym = - _g_sym * _v_sym * _ad_sym + _h_sym * _dg_dt_sym
    _dgam_dt_sym = _g_sym/_v_sym * (_u_sym - ca.cos(_gam_sym))
    _dqdyn_dt_sym = ca.jacobian(_qdyn_sym, _h_sym) * _dh_dt_sym + ca.jacobian(_qdyn_sym, _e_sym) * _de_dt_sym
    _drho_dh_sym = ca.jacobian(_rho_sym, _h_sym)

    # Hamiltonian
    _path_cost = -_dxn_dt_sym
    _hamiltonian_h = _path_cost + _lam_e_sym * _de_dt_sym + _lam_h_sym * _dh_dt_sym + _lam_gam_sym * _dgam_dt_sym
    _hamiltonian_qdyn = _path_cost \
        + _lam_e_sym * _de_dt_sym + _lam_qdyn_sym * _dqdyn_dt_sym + _lam_gam_sym * _dgam_dt_sym

    if _use_qdyn_expansion:
        # ZEROTH-ORDER OUTER SOLUTION
        # From dgamdt = 0 -> u = cos(gam)
        _u_opt = ca.cos(_gam_sym)

        # From H = L + lam_e * dEdt = 0 (dQdyndt = dgamdt = 0 @ glide slope)
        _lam_e_opt = -_path_cost / _de_dt_sym

        # From Hu = 0
        _tmp1 = ca.jacobian(_path_cost + _lam_e_sym * _de_dt_sym + _lam_qdyn_sym * _dqdyn_dt_sym, _u_sym)
        _tmp2 = ca.jacobian(_dgam_dt_sym, _u_sym)
        _lam_gam_opt = _tmp1 / _tmp2

        # From Hgam = 0 [Hu = 0 used to factor out lam_gam]
        _tmp1 = ca.jacobian(_dgam_dt_sym, _gam_sym) * ca.jacobian(_path_cost + _lam_e_sym * _de_dt_sym, _u_sym)
        _tmp2 = ca.jacobian(_dgam_dt_sym, _u_sym) * ca.jacobian(_path_cost + _lam_e_sym * _de_dt_sym, _gam_sym)
        _tmp3 = ca.jacobian(_dgam_dt_sym, _u_sym) * ca.jacobian(_dqdyn_dt_sym, _gam_sym)
        _tmp4 = ca.jacobian(_dgam_dt_sym, _gam_sym) * ca.jacobian(_dqdyn_dt_sym, _u_sym)
        _lam_qdyn_opt = (_tmp1 - _tmp2) / (_tmp3 - _tmp4)

        # The remaining equations are:
        # d(H)/dQdyn = 0, d(Qdyn)/dt = 0 -> h*, gam*
        _dqdyn_dt_opt = _dqdyn_dt_sym
        _dham_dqdyn = d_dqdyn(_hamiltonian_qdyn)
        _dham_dqdyn_opt = _dham_dqdyn

        for _original_arg, _new_arg in zip(
                (_lam_qdyn_sym, _lam_gam_sym, _lam_e_sym, _u_sym),
                (_lam_qdyn_opt, _lam_gam_opt, _lam_e_opt, _u_opt)
        ):
            _dqdyn_dt_opt = ca.substitute(_dqdyn_dt_opt, _original_arg, _new_arg)
            _dham_dqdyn_opt = ca.substitute(_dham_dqdyn_opt, _original_arg, _new_arg)

        _zero_expr = ca.vcat((_dqdyn_dt_opt, _dham_dqdyn_opt))
        _zero_jac_expr = ca.jacobian(_zero_expr, ca.vcat((_h_sym, _gam_sym)))

        _zero_fun = ca.Function('Z', (_h_sym, _gam_sym, _e_sym), (_zero_expr,), ('h', 'gam', 'E'), ('Z',))
        _zero_jac = ca.Function('JZ', (_h_sym, _gam_sym, _e_sym), (_zero_jac_expr,), ('h', 'gam', 'E'), ('JZ',))
        _guess_gam = True
    else:
        # Zeroth-Order Outer Solution
        _lam_e_opt = -1 / (_g_sym * (_ad0_sym + _adl_sym))
        _lam_gam_opt = 2 * _lam_e_opt * _v_sym ** 2 * _adl_sym
        _lam_h_opt = 0

        _dham_dh = ca.jacobian(_hamiltonian_h, _h_sym)
        _d2ham_dh2 = ca.jacobian(_dham_dh, _h_sym)

        _dham_dh_opt = _dham_dh
        _d2ham_dh2_opt = _d2ham_dh2
        _hamiltonian_opt = _hamiltonian_h
        for _original_arg, _new_arg in zip(
                (_gam_sym, _u_sym, _lam_e_sym, _lam_gam_sym, _lam_h_sym),
                (0, 1., _lam_e_opt, _lam_gam_opt, _lam_h_opt)
        ):
            _dham_dh_opt = ca.substitute(_dham_dh_opt, _original_arg, _new_arg)
            _d2ham_dh2_opt = ca.substitute(_d2ham_dh2_opt, _original_arg, _new_arg)
            _hamiltonian_opt = ca.substitute(_hamiltonian_opt, _original_arg, _new_arg)

        _zero_fun = ca.Function('dH_dh', (_h_sym, _e_sym), (_dham_dh_opt,), ('h', 'E'), ('dH_dh',))
        _zero_jac = ca.Function('d2H_dh2', (_h_sym, _e_sym), (_d2ham_dh2_opt,), ('h', 'E'), ('d2H_dh2',))
        _guess_gam = False
        # _hamiltonian_opt_fun = ca.Function('H', (_h_sym, _e_sym), (_hamiltonian_opt,), ('h', 'E'), ('H',))
    _mach_fun = ca.Function('M', (_h_sym, _e_sym), (_mach_sym,), ('h', 'E'), ('M',))


    # Initial guess for gamma generated from L = W, d(Qdyn)/dt = 0
    _dh_dE_expr = 1 / (_g_sym - 0.5 * (_drho_dh_sym / _rho_sym) * _v_sym**2)
    _gam_n1_expr = - np.arcsin(_dh_dE_expr * (_ad0_sym + _adl_sym) * _g_sym)
    _gam_qdyn0_fun_n1 = ca.Function('gam', (_h_sym, _e_sym),
                                    (_gam_n1_expr,),
                                    ('h', 'E'), ('gam',))

    # # Direct Solution (minimize drag w.r.t. h)
    # _cl_n1 = _weight_sym / (_qdyn_sym * s_ref)
    # _cd_n1 = cd0_fun(_mach_sym) + cdl_fun(_mach_sym) * _cl_n1**2
    # _drag_n1 = (_qdyn_sym * s_ref) * _cd_n1
    # _ddrag_dh = ca.jacobian(_drag_n1, _h_sym)
    # _d2drag_dh2 = ca.jacobian(_ddrag_dh, _h_sym)
    #
    # _drag_n1_fun = ca.Function('D', (_h_sym, _e_sym), (_drag_n1,), ('h', 'E'), ('D',))
    # _ddrag_dh_fun = ca.Function('Dh', (_h_sym, _e_sym), (_ddrag_dh,), ('h', 'E'), ('Dh',))
    # _d2drag_dh2_fun = ca.Function('Dhh', (_h_sym, _e_sym), (_d2drag_dh2,), ('h', 'E'), ('Dhh',))

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
    _gam_vals = np.empty(_e_vals.shape)
    _drag_vals = np.empty(_e_vals.shape)

    if _h_guess0 is not None:
        _h_guess = _h_guess0
    else:
        _h_guess = _h_min

    if _gam_guess0 is not None:
        _gam_guess = _gam_guess0
    else:
        # Guess where dynamic pressure is unchanging at initial altitude guess w/ L = W
        _gam_guess = float(_gam_qdyn0_fun_n1(0, _e_vals[0]))

    # The glide slope occurs where the del(Hamiltonian)/delh = 0 for the zeroth-order asymptotic expansion
    idx0 = 0
    idxf = 0

    for idx in range(len(_e_vals)):
        _e_i = _e_vals[idx]

        # Altitude should be monotonically increasing
        _h_min_i = _h_guess

        # # Glide slope occurs where (in energy state) Drag is minimize @ constant energy
        # _min_sol = sp.optimize.minimize(fun=lambda _h_trial: np.asarray(_drag_n1_fun(_h_trial, _e_i)).flatten(),
        #                                 jac=lambda _h_trial: np.asarray(_ddrag_dh_fun(_h_trial, _e_i)).flatten(),
        #                                 hess=lambda _h_trial: np.asarray(_d2drag_dh2_fun(_h_trial, _e_i)).flatten(),
        #                                 x0=np.asarray((_h_guess,)),
        #                                 bounds=((_h_min_i, _h_max),),
        #                                 method='trust-constr',
        #                                 tol=1e-10
        #                                 )
        # _h_i = _min_sol.x[0]

        # Glide slope occurs where (in asymptotic expansion) the the Hamiltonian is stationary w.r.t. altitude
        if _guess_gam:
            _fsolve_sol = sp.optimize.fsolve(
                func=lambda _x_trial: np.asarray(_zero_fun(_x_trial[0], _x_trial[1], _e_i)).flatten(),
                x0=np.asarray((_h_guess, _gam_guess)),
                fprime=lambda _x_trial: np.asarray(_zero_jac(_x_trial[0], _x_trial[1], _e_i))
            )
            if _fsolve_sol[0] < _h_min:
                _h_i = _h_min
                _gam_i = 0.
            else:
                _h_i = max(min(float(_fsolve_sol[0]), _h_max), _h_min_i)
                _gam_i = max(min(float(_fsolve_sol[1]), _gam_max), _gam_min)
        else:
            _fsolve_sol = sp.optimize.fsolve(
                func=lambda _h_trial: np.asarray(_zero_fun(_h_trial, _e_i)).flatten(),
                x0=np.asarray((_h_guess,)),
                fprime=lambda _h_trial: np.asarray(_zero_jac(_h_trial, _e_i)).flatten()
            )
            _h_i = max(min(float(_fsolve_sol[0]), _h_max), _h_min_i)
            _gam_i = _gam_guess

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
        _gam_vals[idx] = _gam_i
        _drag_vals[idx] = drag_n1(_h_i, _e_i, _m)

        # Prepare guess for next iteration
        _h_guess = _h_i
        _gam_guess = _gam_i

    # Remove invalid values where energy exceeds altitude/Mach bounds
    _e_vals = _e_vals[idx0:idxf+1]
    _h_vals = _h_vals[idx0:idxf+1]
    _v_vals = _v_vals[idx0:idxf+1]
    _gam_vals = _gam_vals[idx0:idxf+1]
    _drag_vals = _drag_vals[idx0:idxf+1]

    if not _guess_gam:
        idx_last = len(_gam_vals) - 1
        for idx, (_h_val, _e_val, _drag_val) in enumerate(zip(_h_vals, _e_vals, _drag_vals)):
            # # Model
            # _g_val = mu / (Re + _h_val) ** 2
            # _v_val2 = 2*(_e_val - _g_val * _h_val)
            # _rho_val = atm.density(_h_val)
            # _drho_dh_val = drho_dh(_h_val)
            #
            # _dh_dE = 1 / (_g_val - 0.5*(_drho_dh_val / _rho_val) * _v_val2)

            # Discretized
            if idx == 0:  # Forward difference
                _dh_dE = (_h_vals[idx + 1] - _h_val) / (_e_vals[idx + 1] - _e_val)
            elif idx == idx_last:  # Backward difference
                _dh_dE = (_h_val - _h_vals[idx - 1]) / (_e_val - _e_vals[idx - 1])
            else:  # Central difference
                _dh_dE = (_h_vals[idx + 1] - _h_vals[idx - 1]) / (_e_vals[idx + 1] - _e_vals[idx - 1])

            _gam_vals[idx] = - np.arcsin(_dh_dE * _drag_val / _m)

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


def get_glide_slope_neighboring_feedback(_m, _h_interp, _use_qdyn_expansion: bool = False):
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

    # Solve with asymptotic expansion in E, h, gam
    h_interp, v_interp, gam_interp, drag_interp = get_glide_slope(
        mass, _h_min=h_min, _h_max=h_max, _mach_min=mach_min, _mach_max=mach_max, _use_qdyn_expansion=False
    )
    k_h_interp, k_gam_interp = get_glide_slope_neighboring_feedback(
        mass, h_interp, _use_qdyn_expansion=False
    )

    # Solve with asymptotic expansion in E, Qdyn, gam
    h_interp_qdyn, v_interp_qdyn, gam_interp_qdyn, drag_interp_qdyn = get_glide_slope(
        mass, _h_min=h_min, _h_max=h_max, _mach_min=mach_min, _mach_max=mach_max, _use_qdyn_expansion=True
    )
    k_qdyn_interp_qdyn, k_gam_interp_qdyn = get_glide_slope_neighboring_feedback(
        mass, h_interp_qdyn, _use_qdyn_expansion=True
    )

    e_vals = h_interp.x
    h_vals = h_interp(e_vals)
    h_vals_qdyn = h_interp_qdyn(e_vals)
    v_vals = v_interp(e_vals)
    v_vals_qdyn = v_interp_qdyn(e_vals)
    k_h_vals = k_h_interp(e_vals)
    k_qdyn_vals = k_qdyn_interp_qdyn(e_vals)
    k_gam_vals = k_gam_interp(e_vals)
    k_gam_vals_qdyn = k_gam_interp_qdyn(e_vals)
    mach_vals = v_vals / np.asarray(sped_fun(h_vals)).flatten()
    mach_vals_qdyn = v_vals_qdyn / np.asarray(sped_fun(h_vals_qdyn)).flatten()
    gam_vals = gam_interp(e_vals)
    gam_vals_qdyn = gam_interp_qdyn(e_vals)
    drag_vals = drag_interp(e_vals)
    drag_vals_qdyn = drag_interp_qdyn(e_vals)
    g_vals = mu / (Re + h_vals) ** 2
    g_vals_qdyn = mu / (Re + h_vals_qdyn) ** 2
    weight_vals = mass * g_vals
    weight_vals_qdyn = mass * g_vals_qdyn

    # Compare to max(L/D)
    def get_ld(_mach):
        _cd0 = np.asarray(cd0_fun(_mach)).flatten()
        _cdl = np.asarray(cdl_fun(_mach)).flatten()
        _cla = np.asarray(cla_fun(_mach)).flatten()

        _cl_ld = (_cd0 / _cdl) ** 0.5
        _max_ld = _cl_ld / (_cd0 + _cdl * _cl_ld ** 2)
        return _max_ld

    max_ld_vals = get_ld(mach_vals)
    max_ld_vals_qdyn = get_ld(mach_vals_qdyn)
    ld_vals = weight_vals / drag_vals
    ld_vals_qdyn = weight_vals_qdyn / drag_vals_qdyn

    # Plot Interpolants
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    e_lab = r'$E$ [ft$^2$/s$^2$]'
    mach_lab = 'Mach'
    fig_glideslope = plt.figure()

    ax_h = fig_glideslope.add_subplot(221)
    ax_h.grid()
    ax_h.set_xlabel(e_lab)
    ax_h.set_ylabel(r'$h$ [1,000 ft]')
    ax_h.plot(e_vals, h_vals_qdyn * 1e-3, color=colors[1], label='Qdyn')
    ax_h.plot(e_vals, h_vals * 1e-3, color=colors[0], label='h')
    ax_h.legend()

    ax_hv = fig_glideslope.add_subplot(222)
    ax_hv.grid()
    ax_hv.set_xlabel(mach_lab)
    ax_hv.set_ylabel(r'$h$ [1,000 ft]')
    ax_hv.plot(mach_vals_qdyn, h_vals_qdyn * 1e-3, color=colors[1], label='Qdyn')
    ax_hv.plot(mach_vals, h_vals * 1e-3, color=colors[0], label='h')
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
    ax_gam.plot(mach_vals_qdyn, gam_vals_qdyn * 180/np.pi, color=colors[1], label='Qdyn')
    ax_gam.plot(mach_vals, gam_vals * 180 / np.pi, color=colors[0], label='h')

    ax_drag = fig_glideslope.add_subplot(224)
    ax_drag.grid()
    ax_drag.set_xlabel(mach_lab)
    ax_drag.set_ylabel(r'$D$ [g]')
    ax_drag.plot(mach_vals_qdyn, drag_vals_qdyn / weight_vals_qdyn, color=colors[1], label='Qdyn')
    ax_drag.plot(mach_vals, drag_vals / weight_vals, color=colors[0], label='h')

    fig_glideslope.tight_layout()

    fig_gains = plt.figure()

    ax_k_h = fig_gains.add_subplot(311)
    ax_k_h.grid()
    ax_k_h.set_ylabel(r'$k_h$')
    ax_k_h.plot(mach_vals, k_h_vals, color=colors[0])

    ax_k_gam = fig_gains.add_subplot(312)
    ax_k_gam.grid()
    ax_k_gam.set_ylabel(r'$k_\gamma$')
    ax_k_gam.plot(mach_vals, k_gam_vals_qdyn, color=colors[1], label='Qdyn')
    ax_k_gam.plot(mach_vals, k_gam_vals, color=colors[0], label='h')

    ax_k_qdyn = fig_gains.add_subplot(313)
    ax_k_qdyn.grid()
    ax_k_qdyn.set_xlabel(mach_lab)
    ax_k_qdyn.set_ylabel(r'$Q_\infty$')
    ax_k_qdyn.plot(mach_vals, k_qdyn_vals, color=colors[1])

    fig_gains.tight_layout()

    fig_ld = plt.figure()
    ax_ld = fig_ld.add_subplot(111)
    ax_ld.grid()
    ax_ld.set_xlabel(mach_lab)
    ax_ld.set_ylabel('L/D')
    ax_ld.plot(mach_vals_qdyn, ld_vals_qdyn, color=colors[1], label='Qdyn')
    ax_ld.plot(mach_vals, ld_vals, color=colors[0], label='h')
    ax_ld.plot(mach_vals, max_ld_vals, 'k--', label='Max L/D')
    ax_ld.legend()

    fig_ld.tight_layout()

    plt.show()
