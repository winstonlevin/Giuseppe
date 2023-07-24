from typing import Optional

import numpy as np
import scipy as sp
import casadi as ca

from x15_aero_model import cd0_fun, cdl_fun, cla_fun, s_ref
from x15_atmosphere import atm, dens_fun, sped_fun, mu, Re, g0

# TODO - remove references to glide slope
from glide_slope import get_glide_slope


def derive_expansion_equations(_glide_slope_dict: dict, _use_qdyn_expansion: bool = False):
    # States
    _m_sym = ca.SX.sym('m', 1)
    _e_sym = ca.SX.sym('e', 1)
    _h_sym = ca.SX.sym('h', 1)
    _gam_sym = ca.SX.sym('gam', 1)
    _lam_e_sym = ca.SX.sym('lam_e', 1)
    _lam_gam_sym = ca.SX.sym('lam_gam', 1)

    # Control (Load Factor)
    _u_sym = ca.SX.sym('u', 1)

    # Expressions
    _g_sym = mu / (_h_sym + Re) ** 2
    _v_sym = (2 * (_e_sym - _g_sym * _h_sym)) ** 0.5
    _rho_sym = dens_fun(_h_sym)
    _qdyn_sym = 0.5 * _rho_sym * _v_sym ** 2
    _mach_sym = _v_sym / sped_fun(_h_sym)
    _weight_sym = _m_sym * _g_sym
    _ad0_sym = _qdyn_sym * s_ref * cd0_fun(_mach_sym) / _weight_sym
    _adl_sym = cdl_fun(_mach_sym) * _weight_sym / (_qdyn_sym * s_ref)
    _ad_sym = _ad0_sym + _adl_sym * _u_sym ** 2

    # Dynamics
    _dh_dt_sym = _v_sym * ca.sin(_gam_sym)
    _dv_dt_sym = -_g_sym * (_ad_sym + ca.sin(_gam_sym))
    _dg_dt_sym = ca.jacobian(_g_sym, _h_sym)[0] * _dh_dt_sym
    _dxn_dt_sym = _v_sym * ca.cos(_gam_sym)
    _de_dt_sym = - _g_sym * _v_sym * _ad_sym + _h_sym * _dg_dt_sym
    _dgam_dt_sym = _g_sym/_v_sym * (_u_sym - ca.cos(_gam_sym))
    _dqdyn_dt_sym = ca.jacobian(_qdyn_sym, _h_sym) * _dh_dt_sym + ca.jacobian(_qdyn_sym, _e_sym) * _de_dt_sym
    _drho_dh_sym = ca.jacobian(_rho_sym, _h_sym)

    if _use_qdyn_expansion:
        _lam_vert_name = 'lam_Qdyn'
        _dvert_dt_name = 'dQdyn_dt'
        _dham_dvert_name = 'H_Qdyn'
        _dvert_dt_sym = _dqdyn_dt_sym
        _lam_vert_sym = ca.SX.sym('lam_Qdyn', 1)

        def d_dvert(_arg):
            return ca.jacobian(_arg, _h_sym) * ca.jacobian(_qdyn_sym, _h_sym) \
                   + ca.jacobian(_arg, _e_sym) * ca.jacobian(_qdyn_sym, _e_sym)

        def _d_dx(_arg):
            if _arg.shape[0] == 1:  # Only 1 row, append vertically
                return ca.vcat((d_dvert(_arg), ca.jacobian(_arg, _gam_sym)))
            else:  # Multiple rows, append horizontally
                return ca.hcat((d_dvert(_arg), ca.jacobian(_arg, _gam_sym)))
    else:
        _lam_vert_name = 'lam_h'
        _dvert_dt_name = 'dh_dt'
        _dham_dvert_name = 'H_h'
        _dvert_dt_sym = _dh_dt_sym

        def d_dvert(_arg):
            return ca.jacobian(_arg, _h_sym)

        def _d_dx(_arg):
            return ca.jacobian(_arg, ca.vcat((_h_sym, _gam_sym)))

    _lam_vert_sym = ca.SX.sym(_lam_vert_name, 1)

    # Hamiltonian
    _path_cost = -_dxn_dt_sym
    _hamiltonian = _path_cost + _lam_e_sym * _de_dt_sym + _lam_vert_sym * _dvert_dt_sym + _lam_gam_sym * _dgam_dt_sym

    _u_glide = ca.cos(_gam_sym)
    _lam_e_glide = ca.substitute(-_path_cost / _de_dt_sym, _u_sym, _u_glide)

    # From Hgam = 0 [Hu = 0 used to factor out lam_gam]
    _tmp1 = ca.jacobian(_dgam_dt_sym, _gam_sym) * ca.jacobian(_path_cost + _lam_e_sym * _de_dt_sym, _u_sym)
    _tmp2 = ca.jacobian(_dgam_dt_sym, _u_sym) * ca.jacobian(_path_cost + _lam_e_sym * _de_dt_sym, _gam_sym)
    _tmp3 = ca.jacobian(_dgam_dt_sym, _u_sym) * ca.jacobian(_dvert_dt_sym, _gam_sym)
    _tmp4 = ca.jacobian(_dgam_dt_sym, _gam_sym) * ca.jacobian(_dvert_dt_sym, _u_sym)
    _lam_vert_inner = (_tmp1 - _tmp2) / (_tmp3 - _tmp4)

    # From Hu = 0
    _tmp1 = ca.jacobian(_path_cost + _lam_e_sym * _de_dt_sym + _lam_vert_sym * _dvert_dt_sym, _u_sym)
    _tmp2 = ca.jacobian(_dgam_dt_sym, _u_sym)
    _lam_gam_opt = _tmp1 / _tmp2

    _lam_e_glide_func = ca.Function('lam_E', (_m_sym, _e_sym, _h_sym, _gam_sym),
                                   (_lam_e_glide,), ('m', 'E', 'h', 'gam'), ('lam_E',))
    _lam_vert_inner_fun = ca.Function(
        _lam_vert_name, (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _u_sym), (_lam_vert_inner,),
        ('m', 'E', 'h', 'gam', 'lam_E', 'u'), (_lam_vert_name,)
    )
    _lam_gam_opt_fun = ca.Function(
        'lam_gam', (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _u_sym), (_lam_gam_opt,),
        ('m', 'E', 'h', 'gam', 'lam_E', _lam_vert_name, 'u'), ('lam_gam',)
    )
    _ham_fun = ca.Function(
        'H', (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _lam_gam_sym, _u_sym), (_hamiltonian,),
        ('m', 'E', 'h', 'gam', 'lam_E', _lam_vert_name, 'lam_gam', 'u'), ('H',)
    )
    _dham_dvert_fun = ca.Function(
        _dham_dvert_name, (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _lam_gam_sym, _u_sym),
        (d_dvert(_hamiltonian),), ('m', 'E', 'h', 'gam', 'lam_E', _lam_vert_name, 'lam_gam', 'u'), (_dham_dvert_name,)
    )
    _dvert_dt_fun = ca.Function(
        _dvert_dt_name, (_m_sym, _e_sym, _h_sym, _gam_sym, _u_sym), (_dvert_dt_sym,),
        ('m', 'E', 'h', 'gam', 'u'), (_dvert_dt_name,)
    )

    # OUTER SOLUTION ---------------------------------------------------------------------------------------------------
    # Here, consider E dynamics assuming vert, gam can be changed instantly.
    # Use: m, E
    # Intermediately solve for: h_glide, gam_glide
    # Solve for: lam_E
    _u_outer = _u_glide
    _lam_e_outer = _lam_e_glide_func(_m_sym, _e_sym, _h_sym, _gam_sym)
    _lam_vert_outer = _lam_vert_inner_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_outer, _u_outer)
    _lam_gam_outer = _lam_gam_opt_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_outer, _lam_vert_outer, _u_outer)
    _ham_vert_outer = _dham_dvert_fun(
        _m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_outer, _lam_vert_outer, _lam_gam_outer, _u_outer
    )
    _dvert_dt_outer = _dvert_dt_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _u_outer)

    _f1_expr = ca.vcat((_ham_vert_outer, _dvert_dt_outer))
    _f1_jac_expr = _d_dx(_f1_expr)  # Jac w.r.t. vert, gam

    _lam_e_func = _lam_e_glide_func
    _f1_func = ca.Function('f1', (_m_sym, _e_sym, _h_sym, _gam_sym), (_f1_expr,), ('m', 'E', 'h', 'gam'), ('f1',))
    _f1_jac = ca.Function('f1_jac', (_m_sym, _e_sym, _h_sym, _gam_sym),
                          (_f1_jac_expr,), ('m', 'E', 'h', 'gam'), ('f1_jac',))

    # FIRST INNER SOLUTION ---------------------------------------------------------------------------------------------
    # Here, consider vert dynamics assuming E fixed and gam can be changed instantly.
    # Use: m, E, h, lam_E (from outer solution)
    # Intermediately solve for: gam_inner
    # Solve for: lam_vert.
    _u_inner_1 = _u_glide
    _lam_vert_inner_1 = _lam_vert_inner_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _u_inner_1)
    _lam_gam_inner_1 = _lam_gam_opt_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_inner_1, _u_inner_1)
    _f2_expr = _ham_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_inner_1, _lam_gam_inner_1, _u_inner_1)
    _f2_jac_expr = ca.jacobian(_f2_expr, _gam_sym)

    _lam_vert_func = ca.Function(_lam_vert_name, (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym),
                                 (_lam_vert_inner_1,), ('m', 'E', 'h', 'gam', 'lam_E'), (_lam_vert_name,))
    _f2_func = ca.Function('f2', (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym),
                           (_f2_expr,), ('m', 'E', 'h', 'gam', 'lam_E'), ('f2',))
    _f2_jac = ca.Function('f2_jac', (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym),
                          (_f2_jac_expr,), ('m', 'E', 'h', 'gam', 'lam_E'), ('f2_jac',))

    # SECOND INNER SOLUTION --------------------------------------------------------------------------------------------
    # Here, consider gam dynamics assuming E, vert fixed.
    # Use: m, E, h, gam, lam_E (from outer solution), lam_vert (from first inner solution)
    # Solve for: the optimal control u*.
    _lam_gam_inner_2 = _lam_gam_opt_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _u_sym)
    _f3_expr = _ham_fun(_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _lam_gam_inner_2, _u_sym)
    _f3_jac_expr = ca.jacobian(_f3_expr, _u_sym)

    _f3_func = ca.Function('f3', (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _u_sym),
                           (_f3_expr,), ('m', 'E', 'h', 'gam', 'lam_E', _lam_vert_name, 'u'), ('f2',))
    _f3_jac = ca.Function('f3_jac', (_m_sym, _e_sym, _h_sym, _gam_sym, _lam_e_sym, _lam_vert_sym, _u_sym),
                          (_f3_jac_expr,), ('m', 'E', 'h', 'gam', 'lam_E', _lam_vert_name, 'u'), ('f2_jac',))

    _expansion_dict = {
        'f1': _f1_func,
        'f1_jac': _f1_jac,
        'lam_E': _lam_e_glide_func,
        'f2': _f2_func,
        'f2_jac': _f2_jac,
        _lam_vert_name: _lam_vert_func,
        'f3': _f3_func,
        'f3_jac': _f3_jac
    }

    return _expansion_dict


def solve_newton(_x, _fun: ca.Function, _jac: ca.Function, _max_iter: int, _fun_tol: float):
    _f_prev = _fun(_x)
    _f_norm_prev = np.linalg.norm(_f_prev)

    for idx in range(_max_iter):
        _x_step = -ca.solve(_jac(_x), _f_prev)
        _x_trial = _x + _x_step
        _f = _fun(_x_trial)
        _f_norm = np.linalg.norm(_f)

        _damp = 1.
        while _damp > 0.5**4 and _f_norm > _f_norm_prev:
            _damp *= 0.5
            _x_trial = _x + _damp * _x_step
            _f = _fun(_x_trial)
            _f_norm = np.linalg.norm(_f)

        _x = _x_trial
        _f_prev = _f
        _f_norm_prev = _f_norm

        if np.linalg.norm(_f) < _fun_tol:
            break

    return _x


def get_solution_interpolant(
        _m: float,
        _expansion_dict: Optional[dict] = None, _use_qdyn_expansion: bool = False,
        _e_vals: Optional[np.array] = None, _h_vals: Optional[np.array] = None,
        _h_guess0: Optional[float] = None, _gam_guess0: Optional[float] = None,
        _h_min: float = 0., _h_max: float = 100e3,
        _gam_min: float = -80.*np.pi/180., _gam_max: float = 80.,
        _mach_min: float = 0.25, _mach_max: float = 0.9
):
    if _expansion_dict is None:
        _expansion_dict = derive_expansion_equations(_use_qdyn_expansion=_use_qdyn_expansion)

    if _e_vals is None:
        # Space based on altitude range (note: steps not linear w.r.t. h)
        _n_e_vals = int(np.ceil((_h_max - _h_min)/100.))

        _g_max = mu / (Re + _h_min) ** 2
        _g_min = mu / (Re + _h_max) ** 2
        _e_min = _g_max * _h_min + 0.5 * (_mach_min * atm.speed_of_sound(_h_min)) ** 2
        _e_max = _g_min * _h_max + 0.5 * (_mach_max * atm.speed_of_sound(_h_max)) ** 2
        _e_vals = np.linspace(_e_min, _e_max, _n_e_vals)

    _h_vals = np.empty(_e_vals.shape)
    _gam_vals = np.empty(_e_vals.shape)
    _lam_e_vals = np.empty(_e_vals.shape)

    if _h_guess0 is not None:
        _h_guess = _h_guess0
    else:
        _h_guess = _h_min

    if _gam_guess0 is not None:
        _gam_guess = _gam_guess0
    else:
        _gam_guess = 0.
        # _gam_guess = None

    # OUTER SOLUTION ---------------------------------------------------------------------------------------------------
    # The glide slope occurs where the del(Hamiltonian)/delh = 0 for the zeroth-order asymptotic expansion
    idx0 = 0
    idxf = 0

    for idx in range(len(_e_vals)):
        _e_i = _e_vals[idx]

        # Altitude should be monotonically increasing
        _h_min_i = _h_guess

        # if _gam_guess is None:
        #     _fsolve_sol = sp.optimize.fsolve(
        #         func=lambda _gam_trial:
        #         np.asarray(_expansion_dict['f1'](_m, _e_i, _h_guess, _gam_trial[0])[1]).flatten(),
        #         x0=np.asarray((0.,)),
        #         fprime=lambda _gam_trial:
        #         np.asarray(_expansion_dict['f1_jac'](_m, _e_i, _h_guess, _gam_trial[0])[1, 1])
        #     )
        #     _gam_guess = _fsolve_sol[0]

        # First, solve for h holding gam constant
        _fsolve_sol = sp.optimize.fsolve(
            func=lambda _h_trial:
            np.asarray(_expansion_dict['f1'](_m, _e_i, _h_trial[0], _gam_guess)[0]).flatten(),
            x0=np.asarray((_h_guess,)),
            fprime=lambda _h_trial:
            np.asarray(_expansion_dict['f1_jac'](_m, _e_i, _h_trial[0], _gam_guess)[0, 0])
        )
        _h_guess = _fsolve_sol[0]

        # Glide slope solved from f1
        _fsolve_sol = sp.optimize.fsolve(
            func=lambda _x_trial: np.asarray(_expansion_dict['f1'](_m, _e_i, _x_trial[0], _x_trial[1])).flatten(),
            x0=np.asarray((_h_guess, _gam_guess)),
            fprime=lambda _x_trial: np.asarray(_expansion_dict['f1_jac'](_m, _e_i, _x_trial[0], _x_trial[1]))
        )

        if _fsolve_sol[0] < _h_min:
            _h_i = _h_min
            _gam_i = 0.
        else:
            _h_i = max(min(float(_fsolve_sol[0]), _h_max), _h_min_i)
            _gam_i = max(min(float(_fsolve_sol[1]), _gam_max), _gam_min)

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
        _gam_vals[idx] = _gam_i
        _lam_e_vals[idx] = float(_expansion_dict['lam_E'](_m, _e_i, _h_i, _gam_i))

        # Prepare guess for next iteration
        _h_guess = _h_i
        _gam_guess = _gam_i

    # Remove invalid values where energy exceeds altitude/Mach bounds
    _e_vals = _e_vals[idx0:idxf+1]
    _h_vals = _h_vals[idx0:idxf+1]
    _gam_vals = _gam_vals[idx0:idxf+1]
    _lam_e_vals = _lam_e_vals[idx0:idxf+1]

    # TODO -- remove
    from matplotlib import pyplot as plt
    plt.plot(_e_vals, _h_vals)

    # INNER SOLUTION ---------------------------------------------------------------------------------------------------
    _gam_inner_vals = np.empty((len(_e_vals), len(_h_vals)))
    for idx, (_e_i, _lam_e_i, _gam_glide_i) in enumerate(zip(_e_vals, _lam_e_vals, _gam_vals)):
        # Solve for increasing altitudes
        if idx < idxf:
            _gam_inner_guess = _gam_glide_i
            for jdx, _h_i in enumerate(_h_vals[idx:]):

                _g_i = mu / (Re + _h_i) ** 2
                _v_i2 = 2 * (_e_i - _g_i * _h_i)

                if _v_i2 > 0:
                    # Glide slope solved from f1
                    _fsolve_sol = sp.optimize.fsolve(
                        func=lambda _x_trial: np.asarray(_expansion_dict['f2'](_m, _e_i, _h_i, _x_trial[0], _lam_e_i)).flatten(),
                        x0=np.asarray((_gam_inner_guess,)),
                        fprime=lambda _x_trial: np.asarray(_expansion_dict['f2_jac'](_m, _e_i, _h_i, _x_trial[0], _lam_e_i))
                    )

                    _gam_inner_vals[idx, jdx] = _fsolve_sol[0]
                    _gam_inner_guess = _fsolve_sol[0]
                else:
                    # Altitude too high -> set remaining values to 0
                    _gam_inner_vals[idx, jdx:] = 0.
                    break

        # Solve for decreasing altitudes
        if idx > 0:
            _gam_inner_guess = _gam_glide_i
            for jdx, _h_i in enumerate(_h_vals[idx-1::-1]):
                # Glide slope solved from f1
                _fsolve_sol = sp.optimize.fsolve(
                    func=lambda _x_trial: np.asarray(_expansion_dict['f2'](_m, _e_i, _h_i, _x_trial[0], _lam_e_i)).flatten(),
                    x0=np.asarray((_gam_inner_guess,)),
                    fprime=lambda _x_trial: np.asarray(_expansion_dict['f2_jac'](_m, _e_i, _h_i, _x_trial[0], _lam_e_i))
                )

                _gam_inner_vals[idx, jdx] = _fsolve_sol[0]
                _gam_inner_guess = _fsolve_sol[0]

        # TODO -- remove
        plt.plot(_h_vals, _gam_inner_vals[idx, :])



def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


# def get_expansion_control_law(
#         _expansion_dict: Optional[dict] = None, _glide_slope_dict: Optional[dict] = None,
#         _use_qdyn_expansion: bool = False, _max_iter: int = 10, _fun_tol: float = 1e-6,
#         _mach_min: float = 0., _mach_max: float = np.inf
# ):
#     if _expansion_dict is None:
#         _expansion_dict = get_expansion_control_law(_use_qdyn_expansion=_use_qdyn_expansion)
#
#     if _glide_slope_dict is not None:
#         def form_initial_guess(_x):
#             _g = mu / (Re + _x[0]) ** 2
#             _e = _g * _x[0] + 0.5 * _x[3] ** 2
#             _h_glide_guess = _glide_slope_dict['h'](_e)
#             _gam_glide_guess = _glide_slope_dict['gam'](_e)
#             return _h_glide_guess, _gam_glide_guess
#     else:
#         def form_initial_guess(_x):
#
#
#
#     def control_law(_x):
#         # Conditions at current state
#         _mach = saturate(_x[3] / atm.speed_of_sound(_x[0]), _mach_min, _mach_max)
#         _qdyn = 0.5 * atm.density(_x[0]) * _x[3] ** 2
#         _g = mu / (_x[0] + Re) ** 2
#         _weight = _x[6] * _g
#         _ad0 = float(_qdyn * s_ref * cd0_fun(_mach)) / _weight
#         _adl = float(cdl_fun(_mach)) * _weight / (_qdyn * s_ref)
#
#         _cgam = np.cos(_x[4])
#
#         # Form Initial Guess
#
#
#     return control_law


if __name__ == '__main__':
    from x15_aero_model import weight_empty

    mass = weight_empty / g0
    use_qdyn = False

    glide_slope_dict = get_glide_slope(_m=mass)
    expansion_dict = derive_expansion_equations(_glide_slope_dict=glide_slope_dict, _use_qdyn_expansion=use_qdyn)

    get_solution_interpolant(_expansion_dict=expansion_dict, _m=mass)
