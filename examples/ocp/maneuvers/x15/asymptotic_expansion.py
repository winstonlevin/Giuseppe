from typing import Optional

import numpy as np
import scipy as sp
import casadi as ca

from x15_aero_model import cd0_fun, cdl_fun, cla_fun, s_ref
from x15_atmosphere import atm, dens_fun, sped_fun, mu, Re, g0


def derive_expansion_equations(_use_qdyn_expansion: bool = False):
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


if __name__ == '__main__':
    expansion_dict = derive_expansion_equations(_use_qdyn_expansion=True)
