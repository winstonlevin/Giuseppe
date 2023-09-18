from typing import Optional

import casadi as ca
import numpy as np
from scipy import optimize

from space_shuttle_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, atm

_h_atm_max = atm.h_layers[-1]
_f_zero_converged_flag = 1


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
        _f = _dict['drag'] * _dict['r'] / re
        return _f

    def energy_state_grad(_e, _h):
        _dict = calc_zo_dict(_e, _h)
        _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['lift']))
        _df_dh = (_ddrag_dh * _dict['r'] + _dict['drag']) / re
        return _df_dh

    if manual_derivation:
        def dham_dh(_e, _h):
            _dict = calc_zo_dict(_e, _h)
            _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['lift']))
            _ddrag_dl = CD1 + 2 * CD2/_dict['qdyn_s_ref'] * _dict['lift']
            _dham_dh = _dict['v']/(_dict['drag']*_dict['r']**2) * (
                    _dict['drag'] + _dict['r'] * _ddrag_dh + mass * _dict['v']**2/_dict['r'] * _ddrag_dl)
            return _dham_dh

        d2ham_dh2 = None
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
        ham_hh_sym = ca.jacobian(ham_h_sym, h_sym)

        # Zeroth order values
        lam_e_zo_sym = -mass / (drag_sym * r_sym)
        lam_h_zo_sym = 0.
        lam_gam_zo_sym = lam_e_zo_sym * v_sym**2 * ddrag_dlift_sym
        gam_zo_sym = 0.
        lift_zo_sym = mass * (g_sym - v_sym**2/r_sym)

        ham_h_zo_sym = ham_h_sym
        ham_hh_zo_sym = ham_hh_sym
        zo_vars = (lam_e, lam_h, lam_gam, gam_sym, lift_sym)
        zo_vals = (lam_e_zo_sym, lam_h_zo_sym, lam_gam_zo_sym, gam_zo_sym, lift_zo_sym)

        for zo_var, zo_val in zip(zo_vars, zo_vals):
            ham_h_zo_sym = ca.substitute(ham_h_zo_sym, zo_var, zo_val)
            ham_hh_zo_sym = ca.substitute(ham_hh_zo_sym, zo_var, zo_val)

        ham_h_zo_fun = ca.Function('dham_dh', (e_sym, h_sym), (ham_h_zo_sym,), ('E', 'h'), ('dham_dh',))
        ham_hh_zo_fun = ca.Function('d2ham_dh2', (e_sym, h_sym), (ham_hh_zo_sym,), ('E', 'h'), ('d2ham_dh2',))

        def dham_dh(_e, _h):
            _dham_dh = float(ham_h_zo_fun(_e, _h))
            return _dham_dh

        d2ham_dh2 = None
        # def d2ham_dh2(_e, _h):
        #     _d2ham_dh2 = np.asarray(ham_hh_zo_fun(_e, _h)).flatten()
        #     return _d2ham_dh2

    h_guess = h_min

    if energy_state:
        # Find glide slope by minimizing DR at each energy state
        tol = 1e-3

        def solve_min(_e_val, _h_guess, _h_max_i):
            _sol_min = optimize.minimize(
                fun=lambda _x: energy_state_obj(_e_val, _x[0]),
                x0=np.array((_h_guess,)),
                jac=lambda _x: energy_state_grad(_e_val, _x[0]),
                bounds=((h_min, _h_max_i),),
                tol=tol
            )
            return _sol_min.x[0], _sol_min.fun, _sol_min.jac[0], _sol_min.success

        for idx, e_val in enumerate(e_vals):
            h_max_i = min(h_max, re**2*e_val/(mu - re*e_val))
            h_sol, obj_sol, grad_sol, success = solve_min(e_val, h_guess, h_max_i)

            if abs(grad_sol) > tol or not success:
                # Solution is not at unconstrained minimum. Try other guesses.
                h_guess0 = h_guess

                h_search_vals = h_min + np.linspace(0., 1., 100) * (h_max_i - h_min)
                grad_search_vals = np.empty(h_search_vals.shape)
                for jdx, h_search_val in enumerate(h_search_vals):
                    grad_search_vals[jdx] = energy_state_grad(e_val, h_search_val)
                sign_flips = np.not_equal(np.sign(grad_search_vals[1:]), np.sign(grad_search_vals[:-1]))

                if np.any(sign_flips):
                    idx_sign_flips = np.where(sign_flips)[0]
                    sol_list = []
                    for idx_sign_flip in idx_sign_flips:
                        h_guess = 0.5 * float(h_search_vals[idx_sign_flip] + h_search_vals[idx_sign_flip + 1])
                        h_new, obj_new, grad_new, success_new = solve_min(e_val, h_guess, h_max_i)
                        if success_new and obj_new < obj_sol:
                            h_sol = h_new
                            obj_sol = obj_new
                            success = success_new

            if success:
                h_guess = h_sol
                h_vals[idx] = h_sol
            else:
                h_vals[idx] = np.nan
    else:
        # Find glide slope by solving d(hamiltonian)/dh = 0

        if d2ham_dh2 is not None:
            def solve_dham_dh_zero(_e_val, _h_guess):
                _x_val, _, _flag, __ = optimize.fsolve(
                    func=lambda _x: dham_dh(e_val, _x[0]),
                    x0=np.array((h_guess,)),
                    fprime=lambda _x: d2ham_dh2(e_val, _x[0]),
                    full_output=True
                )
                return _x_val[0], _flag
        else:
            def solve_dham_dh_zero(_e_val, _h_guess):
                _x_val, _, _flag, __ = optimize.fsolve(
                    func=lambda _x: dham_dh(e_val, _x[0]),
                    x0=np.array((h_guess,)),
                    full_output=True
                )
                return _x_val[0], _flag

        for idx, e_val in enumerate(e_vals):
            h_max_i = min(h_max, re**2*e_val/(mu - re*e_val))

            dict_guess = calc_zo_dict(e_val, h_guess)

            h_sol, flag = solve_dham_dh_zero(e_val, h_guess)
            if flag != _f_zero_converged_flag or h_sol > h_max_i:
                # Solution did not converge, try other initial states.
                h_guess0 = h_guess

                h_search_vals = h_min + np.linspace(0., 1., 100) * (h_max_i - h_min)
                dham_dh_search_vals = np.empty(h_search_vals.shape)
                for jdx, h_search_val in enumerate(h_search_vals):
                    dham_dh_search_vals[jdx] = dham_dh(e_val, h_search_val)
                sign_flips = np.not_equal(np.sign(dham_dh_search_vals[1:]), np.sign(dham_dh_search_vals[:-1]))

                if np.any(sign_flips):
                    idx_sign_flips = np.where(sign_flips)[0]
                    sol_list = []
                    for idx_sign_flip in idx_sign_flips:
                        h_guess = h_search_vals[idx_sign_flip]
                        h_sol, flag = solve_dham_dh_zero(e_val, h_guess)
                        if flag == _f_zero_converged_flag:
                            sol_list.append(h_sol)

                    if len(sol_list) > 0:
                        sol_array = np.array(sol_list)
                        sol_distance = (sol_array - h_guess0)**2
                        h_sol = sol_array[np.where(sol_distance == np.min(sol_distance))]


            h_guess = h_sol
            if flag == _f_zero_converged_flag and h_sol < h_max:
                h_sol = min(max(h_sol, h_min), h_max_i)
                h_vals[idx] = h_sol
            else:
                h_vals[idx] = np.nan

    return e_vals, h_vals


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    h_max_plot = _h_atm_max
    mach_max_plot = 40.
    e_vals, h_vals = get_glide_slope(energy_state=False, h_max=h_max_plot, mach_max=mach_max_plot)
    _, h_vals_es = get_glide_slope(energy_state=True, h_max=h_max_plot, mach_max=mach_max_plot)

    v_vals = (2 * (e_vals + mu/re - mu/(re+h_vals)))**0.5
    v_vals_es = (2 * (e_vals + mu / re - mu / (re + h_vals_es))) ** 0.5

    fig_ham_h = plt.figure()
    ax_h = fig_ham_h.add_subplot(211)
    ax_h.plot(e_vals, h_vals, label='dH/dh = 0')
    ax_h.plot(e_vals, h_vals_es, '--', label='min(DR)')
    ax_h.plot((e_vals[0], e_vals[-1]), (0., 0.), 'k--')
    ax_h.plot((e_vals[0], e_vals[-1]), (h_max_plot, h_max_plot), 'k--')
    ax_h.legend()

    ax_v = fig_ham_h.add_subplot(212)
    ax_v.plot(e_vals, v_vals, label='Hh = 0')
    ax_v.plot(e_vals, v_vals_es, '--', label='Energy State')

    plt.show()
