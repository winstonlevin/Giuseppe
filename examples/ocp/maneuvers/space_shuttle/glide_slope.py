from typing import Optional

import casadi as ca
import numpy as np
from scipy import optimize

from space_shuttle_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, atm

_h_atm_max = atm.h_layers[-1]


def get_glide_slope(e_vals: Optional[np.array] = None,
                    h_min: float = 0., h_max: float = _h_atm_max,
                    mach_min: float = 0.3, mach_max: float = 30.,
                    manual_derivation=False, energy_state=False, correct_gam=True):

    # Initialize arrays for interpolant (h = f(E))
    if e_vals is None:
        e_min = mu/re - mu/(re+h_min) + 0.5 * (atm.speed_of_sound(h_min)*mach_min)**2
        e_max = mu/re - mu/(re+h_max) + 0.5 * (atm.speed_of_sound(h_max)*mach_max)**2
        n_vals = int(np.ceil((h_max - h_min)/100.))  # Space out every 100 ft on average
        e_vals = np.linspace(e_min, e_max, n_vals)
    else:
        if e_vals[-1] < e_vals[0]:
            e_vals = np.flip(e_vals)  # Put E in ascending order

    h_vals = np.empty(e_vals.shape)

    # Derive: d(hamiltonian)/dh = 0
    e_sym = ca.SX.sym('e')
    h_sym = ca.SX.sym('h')
    gam_sym = ca.SX.sym('gam')
    lift_sym = ca.SX.sym('lift')

    r_sym = re + h_sym
    g_sym = mu / r_sym ** 2
    v_sym = (2 * (e_sym + mu/r_sym - mu/re)) ** 0.5
    _, __, rho_sym = atm.get_ca_atm_expr(h_sym)
    qdyn_s_ref_sym = 0.5 * rho_sym * v_sym**2 * s_ref
    drag_sym = CD0 * qdyn_s_ref_sym + CD1 * lift_sym + CD2 / qdyn_s_ref_sym * lift_sym ** 2
    ddrag_dh_sym = ca.jacobian(drag_sym, h_sym)
    ddrag_dh_fun = ca.Function('Dh', (e_sym, h_sym, lift_sym), (ddrag_dh_sym,), ('E', 'h', 'L'), ('Dh',))


    def calc_zo_dict(_e, _h):
        _dict = {}
        _dict['r'] = re + _h
        _dict['g'] = mu / _dict['r'] ** 2
        _dict['v'] = (max(1., 2. * (_e + mu / _dict['r'] - mu / re))) ** 0.5
        _dict['rho'] = atm.density(_h)
        _dict['qdyn_s_ref'] = 0.5 * _dict['rho'] * _dict['v'] ** 2 * s_ref
        _dict['lift'] = mass * (_dict['g'] - _dict['v'] ** 2 / _dict['r'])
        _dict['drag'] = CD0 * _dict['qdyn_s_ref'] + CD1 * _dict['lift'] + CD2 / _dict['qdyn_s_ref'] * _dict['lift'] ** 2
        return _dict

    def energy_state_obj(_e, _h):
        _dict = calc_zo_dict(_e, _h)
        _f = _dict['drag'] * _dict['r']
        return _f

    def energy_state_grad(_e, _h):
        _dict = calc_zo_dict(_e, _h)
        _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['lift']))
        _df_dh = _ddrag_dh * _dict['r'] + _dict['drag']
        return _df_dh

    if manual_derivation:
        def dham_dh(_e, _h):
            _dict = calc_zo_dict(_e, _h)
            _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['lift']))
            _ddrag_dl = CD1 + 2 * CD2/_dict['qdyn_s_ref'] * _dict['lift']
            _dham_dh = _dict['v']/(_dict['drag']*_dict['r']**2) * (
                    _dict['drag'] + _dict['r'] * _ddrag_dh + mass * _dict['v']**2/_dict['r'] * _ddrag_dl)
            return _dham_dh

    else:
        lam_e = ca.SX.sym('lam_e')
        lam_h = ca.SX.sym('lam_h')
        lam_gam = ca.SX.sym('lam_gam')

        de_dt = - v_sym * drag_sym / mass
        dh_dt = v_sym * ca.sin(gam_sym)
        dgam_dt = lift_sym / (mass * v_sym) + (v_sym / r_sym - g_sym / v_sym) * ca.cos(gam_sym)
        dtha_dt = v_sym/r_sym * ca.cos(gam_sym)

        ddrag_dlift_sym = ca.jacobian(drag_sym, lift_sym)

        ham_sym = -dtha_dt + lam_e * de_dt + lam_h * dh_dt + lam_gam * dgam_dt
        ham_h_sym = ca.jacobian(ham_sym, h_sym)

        # Zeroth order values
        lam_e_zo_sym = -mass / (drag_sym * r_sym)
        lam_h_zo_sym = 0.
        lam_gam_zo_sym = lam_e_zo_sym * v_sym**2 * ddrag_dlift_sym
        gam_zo_sym = 0.
        lift_zo_sym = mass * (g_sym - v_sym**2/r_sym)

        ham_h_zo_sym = ham_h_sym
        zo_vars = (lam_e, lam_h, lam_gam, gam_sym, lift_sym)
        zo_vals = (lam_e_zo_sym, lam_h_zo_sym, lam_gam_zo_sym, gam_zo_sym, lift_zo_sym)

        for zo_var, zo_val in zip(zo_vars, zo_vals):
            ham_h_zo_sym = ca.substitute(ham_h_zo_sym, zo_var, zo_val)

        ham_h_zo_fun = ca.Function('dham_dh', (e_sym, h_sym), (ham_h_zo_sym,), ('E', 'h'), ('dham_dh',))

        def dham_dh(_e, _h):
            _dham_dh = float(ham_h_zo_fun(_e, _h))
            return _dham_dh

    if energy_state:
        # Find glide slope by minimizing DR at each energy state
        h_guess = h_min

        for idx, e_val in enumerate(e_vals):
            h_max_i = e_val / g0
            x_min = optimize.minimize(
                fun=lambda _x: energy_state_obj(e_val, _x[0]),
                x0=np.array((h_guess,)),
                jac=lambda _x: energy_state_grad(e_val, _x[0]),
                bounds=((h_min, h_max_i),)
            )
            h_guess = x_min[0]
            h_vals[idx] = x_min[0]
    else:
        # Find glide slope by solving d(hamiltonian)/dh = 0
        raise RuntimeError('Not implemented!')

    return e_vals, h_vals


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    e_vals, h_vals = get_glide_slope(energy_state=True)

    fig_ham_h = plt.figure()
    ax_ham_h = fig_ham_h.add_subplot(111)
    ax_ham_h.plot(e_vals, h_vals)

    plt.show()
