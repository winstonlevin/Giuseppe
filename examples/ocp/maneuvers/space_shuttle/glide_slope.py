from typing import Optional, Callable

import casadi as ca
import numpy as np
import scipy as sp

from space_shuttle_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, atm

_h_atm_max = atm.h_layers[-1]
_f_zero_converged_flag = 1


def get_glide_slope(e_vals: Optional[np.array] = None,
                    h_min: float = 0., h_max: float = _h_atm_max,
                    mach_min: float = 0.3, mach_max: float = 30.,
                    manual_derivation=False, energy_state=False, correct_gam=True, flat_earth=False,
                    nondimensionalize_control=True) -> dict:
    """

    Parameters
    ----------
    e_vals : np.array, Specific energy values
    h_min : float, minimum altitude for glide slope
    h_max : float, maximum altitude for glide slope
    mach_min : float, approximate minimum Mach number for glide slope (used to calculate e_min)
    mach_max : float, approximate maximum Mach number for glide slope (used to calculate e_max)
    manual_derivation : bool, True -> use manual derivation for d(Hamiltonian)/dh = 0
    energy_state : bool, True -> use minimization from Energy State approximation instead of asymptotic expansion
    correct_gam : bool, True -> use dh/dE value to calculate corrected gamma
    flat_earth : bool, True -> use flat earth dynamics
    nondimensionalize_control : bool, True -> use u = L/W0. False -> use u = L

    Returns
    -------
    glide_dict : dict, dictionary of glide slope values

    """

    if flat_earth:
        def hv_to_e(_h, _v):
            # E = gh + 0.5 V**2
            return g0 * _h + 0.5 * _v**2

        def eh_to_v2(_e, _h):
            # E = gh + 0.5 V**2 -> V**2 = 2(E - gh)
            return 2 * (_e - g0 * _h)

        def ev_to_h(_e, _v):
            # E = gh + 0.5 V**2 -> h = (E - 0.5 V**2)/g
            return (_e - 0.5 * _v**2)/g0

        def h_to_g(_h):
            return g0

        def lift_gam0(_h, _v, _gam):
            return mass * g0 * np.cos(_gam)
    else:
        def hv_to_e(_h, _v):
            # E = mu/re - mu/r + 0.5 V**2
            return mu/re - mu/(re + _h) + 0.5 * _v**2

        def eh_to_v2(_e, _h):
            # E = mu/re - mu/r + 0.5 V**2 -> V**2 = 2(E + mu/r - mu/re)
            return 2 * (_e + mu/(_h + re) - mu/re)

        def ev_to_h(_e, _v):
            # E = mu/re - mu/(re + h) + 0.5 V**2 -> h = -re - mu/(E - mu/re - 0.5 V**2)
            return -re - mu/(_e - mu/re - 0.5 * _v**2)

        def h_to_g(_h):
            return mu / (re + _h)**2

        def lift_gam0(_h, _v, _gam):
            _g = h_to_g(_h)
            _r = re + _h
            return mass * (_g - _v ** 2 / _r) * np.cos(_gam)

    if nondimensionalize_control:
        def u_to_lift(_u):
            return _u * mass * g0

        def lift_to_u(_lift):
            return _lift / (mass * g0)
    else:
        def u_to_lift(_u):
            return _u

        def lift_to_u(_lift):
            return _lift

    # Initialize arrays for interpolant (h = f(E))
    if e_vals is None:
        e_min = hv_to_e(h_min, atm.speed_of_sound(h_min)*mach_min)
        e_max = hv_to_e(h_max, atm.speed_of_sound(h_max)*mach_max)
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
    u_sym = ca.SX.sym('u')
    lift_sym = u_to_lift(u_sym)

    r_sym = re + h_sym
    g_sym = h_to_g(h_sym)
    v_sym = eh_to_v2(e_sym, h_sym) ** 0.5
    _, __, rho_sym = atm.get_ca_atm_expr(h_sym)
    qdyn_s_ref_sym = 0.5 * rho_sym * v_sym**2 * s_ref
    drag_sym = CD0 * qdyn_s_ref_sym + CD1 * lift_sym + CD2 / qdyn_s_ref_sym * lift_sym ** 2
    ddrag_dh_sym = ca.jacobian(drag_sym, h_sym)
    ddrag_dh_fun = ca.Function('Dh', (e_sym, h_sym, u_sym), (ddrag_dh_sym,), ('E', 'h', 'u'), ('Dh',))
    dens_fun = ca.Function('rho', (h_sym,), (rho_sym,), ('h',), ('rho',))

    # Jacobians (for use with neighboring feedback gains)
    lam_e = ca.SX.sym('lam_e')
    lam_h = ca.SX.sym('lam_h')
    lam_gam = ca.SX.sym('lam_gam')

    if flat_earth:
        de_dt = -v_sym * drag_sym / mass
        dh_dt = v_sym * ca.sin(gam_sym)
        dgam_dt = lift_sym / (mass * v_sym) - g_sym / v_sym * ca.cos(gam_sym)
        dtha_dt = v_sym * ca.cos(gam_sym)
    else:
        de_dt = - v_sym * drag_sym / mass
        dh_dt = v_sym * ca.sin(gam_sym)
        dgam_dt = lift_sym / (mass * v_sym) + (v_sym / r_sym - g_sym / v_sym) * ca.cos(gam_sym)
        dtha_dt = v_sym / r_sym * ca.cos(gam_sym)

    ddrag_du_sym = ca.jacobian(drag_sym, u_sym)
    dlift_du_sym = ca.jacobian(lift_sym, u_sym)

    hamiltonian_sym = -dtha_dt + lam_e * de_dt + lam_h * dh_dt + lam_gam * dgam_dt

    # Derivation of values to calculate neighboring optimal control feedback gains from ARE
    states_sym = ca.vcat((h_sym, gam_sym))
    controls_sym = u_sym
    dynamics_sym = ca.vcat((dh_dt, dgam_dt))

    ham_x = ca.jacobian(hamiltonian_sym, states_sym)
    ham_u = ca.jacobian(hamiltonian_sym, controls_sym)

    a_noc_sym = ca.jacobian(dynamics_sym, states_sym)
    b_noc_sym = ca.jacobian(dynamics_sym, controls_sym)
    q_noc_sym = ca.jacobian(ham_x, states_sym)
    n_noc_sym = ca.jacobian(ham_x, controls_sym)
    r_noc_sym = ca.jacobian(ham_u, controls_sym)

    a_noc_fun = ca.Function('A', (e_sym, h_sym, gam_sym, lam_e, lam_h, lam_gam, u_sym),
                            (a_noc_sym,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'u'), ('A',))
    b_noc_fun = ca.Function('B', (e_sym, h_sym, gam_sym, lam_e, lam_h, lam_gam, u_sym),
                            (b_noc_sym,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'u'), ('B',))
    q_noc_fun = ca.Function('Q', (e_sym, h_sym, gam_sym, lam_e, lam_h, lam_gam, u_sym),
                            (q_noc_sym,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'u'), ('Q',))
    n_noc_fun = ca.Function('N', (e_sym, h_sym, gam_sym, lam_e, lam_h, lam_gam, u_sym),
                            (n_noc_sym,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'u'), ('N',))
    r_noc_fun = ca.Function('R', (e_sym, h_sym, gam_sym, lam_e, lam_h, lam_gam, u_sym),
                            (r_noc_sym,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'u'), ('R',))

    def calc_zo_dict(_e, _h, _gam=None):
        _dict = {}

        if _gam is None:  # Default to zero
            _gam = 0. * _e

        _dict['r'] = re + _h
        _dict['g'] = h_to_g(_h)
        _dict['v'] = np.maximum(1., 2. * (_e + mu / _dict['r'] - mu / re)) ** 0.5
        _dens_arr = np.asarray(dens_fun(_h)).flatten()
        if len(_dens_arr) > 1:
            _dict['rho'] = _dens_arr
        else:
            _dict['rho'] = _dens_arr[0]
        _dict['qdyn_s_ref'] = 0.5 * _dict['rho'] * _dict['v'] ** 2 * s_ref
        _dict['lift'] = lift_gam0(_h, _dict['v'], _gam)
        _dict['u'] = lift_to_u(_dict['lift'])
        _dict['drag'] = CD0 * _dict['qdyn_s_ref'] + CD1 * _dict['lift'] + CD2 / _dict['qdyn_s_ref'] * _dict['lift'] ** 2
        _dict['ddrag_dlift'] = CD1 + 2 * CD2 / _dict['qdyn_s_ref'] * _dict['lift']

        _dict['lam_e'] = -mass * np.cos(_gam) / (_dict['drag'] * _dict['r'])
        _dict['lam_gam'] = _dict['lam_e'] * _dict['v']**2 * _dict['ddrag_dlift']
        _dict['lam_h'] = ((_dict['lam_gam'] - 1) / _dict['r']
                          - _dict['lam_gam'] * _dict['g'] / _dict['v']**2) * np.tan(_gam)
        return _dict

    if flat_earth:
        def energy_state_zero(_e, _h):  # min(D) -> 0 = Dh
            _dict = calc_zo_dict(_e, _h)
            _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['u']))
            _df_dh = _ddrag_dh
            return _df_dh
    else:
        def energy_state_zero(_e, _h):  # min(DR) -> 0 = DhR + D
            _dict = calc_zo_dict(_e, _h)
            _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['u']))
            _df_dh = (_ddrag_dh * _dict['r'] + _dict['drag']) / re
            return _df_dh

    if manual_derivation:  # Valid for spherical earth
        if flat_earth:
            # d(Hamiltonian)/dh = F(x) * Dh = 0 -> ignore F(x), zero is at some location as energy state
            dham_dh = energy_state_zero
        else:
            def dham_dh(_e, _h):
                _dict = calc_zo_dict(_e, _h)
                _ddrag_dh = float(ddrag_dh_fun(_e, _h, _dict['u']))
                _ddrag_dl = CD1 + 2 * CD2/_dict['qdyn_s_ref'] * _dict['lift']
                _dham_dh = _dict['v']/(_dict['drag']*_dict['r']**2) * (
                        _dict['drag'] + _dict['r'] * _ddrag_dh + mass * _dict['v']**2/_dict['r'] * _ddrag_dl)
                return _dham_dh

        dham_dh_grad = None
    else:
        ham_h_sym = ca.jacobian(hamiltonian_sym, h_sym)

        # Zeroth order values
        if flat_earth:
            lam_e_zo_sym = -mass / drag_sym
        else:
            lam_e_zo_sym = -mass / (drag_sym * r_sym)
        lam_h_zo_sym = 0.
        lam_gam_zo_sym = lam_e_zo_sym * v_sym**2 * ddrag_du_sym / dlift_du_sym
        gam_zo_sym = 0.
        lift_zo_sym = lift_gam0(h_sym, v_sym, gam_zo_sym)
        u_zo_sym = lift_to_u(lift_zo_sym)

        ham_h_zo_sym = ham_h_sym
        zo_vars = (lam_e, lam_h, lam_gam, gam_sym, u_sym)
        zo_vals = (lam_e_zo_sym, lam_h_zo_sym, lam_gam_zo_sym, gam_zo_sym, u_zo_sym)

        for zo_var, zo_val in zip(zo_vars, zo_vals):
            ham_h_zo_sym = ca.substitute(ham_h_zo_sym, zo_var, zo_val)

        ham_h_zo_fun = ca.Function('dham_dh', (e_sym, h_sym), (ham_h_zo_sym,), ('E', 'h'), ('dham_dh',))
        ham_h_zo_grad_fun = ca.Function('dham_dh_grad', (e_sym, h_sym),
                                        (ca.jacobian(ham_h_zo_sym, h_sym),), ('E', 'h'), ('dham_dh_grad',))

        def dham_dh(_e, _h):
            _dham_dh = float(ham_h_zo_fun(_e, _h))
            return _dham_dh

        def dham_dh_grad(_e, _h):
            _dham_dh_grad = float(ham_h_zo_grad_fun(_e, _h))
            return _dham_dh_grad

    h_guess = h_min

    if energy_state:
        # Find glide slope by minimizing DR at each energy state
        fzero = energy_state_zero
        fgrad = None
    else:
        # Find glide slope by solving d(hamiltonian)/dh = 0
        fzero = dham_dh
        fgrad = dham_dh_grad

    # Find where the equation "fzero" is equal to zero
    if fgrad is Callable:
        def solve_zero(_e_val, _h_guess):
            _x_val, _, _flag, __ = sp.optimize.fsolve(
                func=lambda _x: fzero(e_val, _x[0]),
                x0=np.array((h_guess,)),
                fprime=lambda _x: fgrad(e_val, _x[0]),
                full_output=True
            )
            return _x_val[0], _flag
    else:
        def solve_zero(_e_val, _h_guess):
            _x_val, _, _flag, __ = sp.optimize.fsolve(
                func=lambda _x: fzero(e_val, _x[0]),
                x0=np.array((h_guess,)),
                full_output=True
            )
            return _x_val[0], _flag

    for idx, e_val in enumerate(e_vals):
        h_guess0 = h_guess
        h_max_i = min(h_max, ev_to_h(e_val, 1.))

        h_sol, flag = solve_zero(e_val, h_guess)
        if flag != _f_zero_converged_flag or h_sol > h_max_i:
            # Solution did not converge, try other initial states.
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
                    h_sol, flag = solve_zero(e_val, h_guess)
                    if flag == _f_zero_converged_flag:
                        sol_list.append(h_sol)

                if len(sol_list) > 0:
                    sol_array = np.array(sol_list)
                    sol_distance = (sol_array - h_guess0)**2
                    h_sol = sol_array[np.where(sol_distance == np.min(sol_distance))]

        if flag == _f_zero_converged_flag and h_sol < h_max:
            h_guess = h_sol
            h_sol = min(max(h_sol, h_min), h_max_i)
            h_vals[idx] = h_sol
        else:
            h_guess = h_guess0
            h_vals[idx] = np.nan

    # Remove invalid idces (indicated by nan)
    valid_idces = np.where(np.logical_not(np.isnan(h_vals)))
    e_vals = e_vals[valid_idces]
    h_vals = h_vals[valid_idces]

    # Calculate gliding flight-path angle
    if correct_gam:
        # Correct gamma vals based on interpolation dh/dE:
        # gam = - arcsin( [D/m] * [dh/dE] )
        h_interp = sp.interpolate.PchipInterpolator(e_vals, h_vals)
        _zo_dict = calc_zo_dict(e_vals, h_vals)
        dh_de_interp = h_interp.derivative(1)
        gam_vals = - np.arcsin(_zo_dict['drag'] / mass * dh_de_interp(e_vals))

    else:
        # Use the zeroth order outer value of gamma, i.e. gam = 0
        gam_vals = np.zeros(e_vals.shape)

    # Calculate neighboring feedback gains
    k_h_vals = np.empty(e_vals.shape)
    k_gam_vals = np.empty(e_vals.shape)
    for idx, (e_val, h_val, gam_val) in enumerate(zip(e_vals, h_vals, gam_vals)):
        zo_dict = calc_zo_dict(e_val, h_val, gam_val)
        a_noc = np.asarray(a_noc_fun(
            e_val, h_val, gam_val, zo_dict['lam_e'], zo_dict['lam_h'], zo_dict['lam_gam'], zo_dict['u']
        ))
        b_noc = np.asarray(b_noc_fun(
            e_val, h_val, gam_val, zo_dict['lam_e'], zo_dict['lam_h'], zo_dict['lam_gam'], zo_dict['u']
        ))
        q_noc = np.asarray(q_noc_fun(
            e_val, h_val, gam_val, zo_dict['lam_e'], zo_dict['lam_h'], zo_dict['lam_gam'], zo_dict['u']
        ))
        n_noc = np.asarray(n_noc_fun(
            e_val, h_val, gam_val, zo_dict['lam_e'], zo_dict['lam_h'], zo_dict['lam_gam'], zo_dict['u']
        ))
        r_noc = np.asarray(r_noc_fun(
            e_val, h_val, gam_val, zo_dict['lam_e'], zo_dict['lam_h'], zo_dict['lam_gam'], zo_dict['u']
        ))
        p_noc = sp.linalg.solve_continuous_are(a=a_noc, b=b_noc, q=q_noc, s=n_noc, r=r_noc)
        k_noc = np.linalg.solve(a=r_noc, b=(p_noc @ b_noc + n_noc).T)  # inv(R) * (PB + N)^T
        k_h_vals[idx] = k_noc[0, 0]
        k_gam_vals[idx] = k_noc[0, 1]

    glide_dict = {}
    glide_dict['E'] = e_vals
    glide_dict['h'] = h_vals
    glide_dict['gam'] = gam_vals
    glide_dict['k_h'] = k_h_vals
    glide_dict['k_gam'] = k_gam_vals

    return glide_dict


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    r2d = 180/np.pi

    h_max_plot = 275e3
    mach_max_plot = 40.
    glide_dict = get_glide_slope(energy_state=False, correct_gam=True, h_max=h_max_plot, mach_max=mach_max_plot)
    # glide_dict_es = get_glide_slope(energy_state=True, correct_gam=True, h_max=h_max_plot, mach_max=mach_max_plot)
    # glide_dict_gam0 = get_glide_slope(energy_state=False, correct_gam=False, h_max=h_max_plot, mach_max=mach_max_plot)
    glide_dict_flat = get_glide_slope(energy_state=False, correct_gam=True, flat_earth=True,
                                      h_max=h_max_plot, mach_max=mach_max_plot)
    glide_dict_flat_gam0 = get_glide_slope(energy_state=False, correct_gam=False, flat_earth=True,
                                      h_max=h_max_plot, mach_max=mach_max_plot)

    e_vals = glide_dict['E']
    h_vals = glide_dict['h']
    gam_vals = glide_dict['gam']
    v_vals = (2 * (e_vals + mu/re - mu/(re+h_vals)))**0.5
    k_h_vals = glide_dict['k_h']
    k_gam_vals = glide_dict['k_gam']

    e_vals_flat = glide_dict_flat['E']
    h_vals_flat = glide_dict_flat['h']
    gam_vals_flat = glide_dict_flat['gam']
    v_vals_flat = (2 * (e_vals_flat + mu / re - mu / (re + h_vals_flat))) ** 0.5
    k_h_vals_flat = glide_dict_flat['k_h']
    k_gam_vals_flat = glide_dict_flat['k_gam']

    e_vals_flat_gam0 = glide_dict_flat_gam0['E']
    k_h_vals_flat_gam0 = glide_dict_flat_gam0['k_h']
    k_gam_vals_flat_gam0 = glide_dict_flat_gam0['k_gam']

    # e_vals_es = glide_dict_es['E']
    # h_vals_es = glide_dict_es['h']
    # gam_vals_es = glide_dict_es['gam']
    # v_vals_es = (2 * (e_vals_es + mu / re - mu / (re + h_vals_es))) ** 0.5
    #
    # e_vals_gam0 = glide_dict_gam0['E']
    # k_h_vals_gam0 = glide_dict_gam0['k_h']
    # k_gam_vals_gam0 = glide_dict_gam0['k_gam']

    e_lab = r'$E$ [ft$^2$/s$^2$]'

    fig_glide_slope = plt.figure()
    ax_h = fig_glide_slope.add_subplot(221)
    ax_h.plot(e_vals, h_vals, label='Sphere')
    ax_h.plot(e_vals_flat, h_vals_flat, label='Flat')
    # ax_h.plot(e_vals_es, h_vals_es, '--', label='min(DR)')
    ax_h.plot((e_vals[0], e_vals[-1]), (0., 0.), 'k--')
    ax_h.plot((e_vals[0], e_vals[-1]), (h_max_plot, h_max_plot), 'k--')
    ax_h.legend()
    ax_h.set_xlabel(e_lab)
    ax_h.set_ylabel(r'$h$ [ft]')

    ax_v = fig_glide_slope.add_subplot(222)
    ax_v.plot(e_vals, v_vals, label='Sphere')
    ax_v.plot(e_vals_flat, v_vals_flat, label='Flat')
    # ax_v.plot(e_vals_es, v_vals_es, '--', label='min(DR)')
    ax_v.set_xlabel(e_lab)
    ax_v.set_ylabel(r'$V$ [ft/s]')

    ax_gam = fig_glide_slope.add_subplot(223)
    ax_gam.plot(e_vals, gam_vals * r2d, label='Sphere')
    ax_gam.plot(e_vals_flat, gam_vals_flat * r2d, label='Flat')
    # ax_gam.plot(e_vals_es, gam_vals_es * r2d, '--', label='min(DR)')
    ax_gam.set_xlabel(e_lab)
    ax_gam.set_ylabel(r'$\gamma$ [deg]')

    fig_glide_slope.tight_layout()

    fig_gains = plt.figure()

    ax_k_h = fig_gains.add_subplot(211)
    ax_k_h.plot(e_vals, k_h_vals, label='Spherical')
    ax_k_h.plot(e_vals_flat, k_h_vals_flat, '--', label='Flat')
    ax_k_h.plot(e_vals_flat_gam0, k_h_vals_flat_gam0, ':', label='Flat, gam = 0')
    ax_k_h.set_xlabel(e_lab)
    ax_k_h.set_ylabel(r'$K_h$')
    ax_k_h.legend()

    ax_k_gam = fig_gains.add_subplot(212)
    ax_k_gam.plot(e_vals, k_gam_vals, label='Spherical')
    ax_k_gam.plot(e_vals_flat, k_gam_vals_flat, '--', label='Flat')
    ax_k_gam.plot(e_vals_flat_gam0, k_gam_vals_flat_gam0, '--', label='Flat, gam = 0')
    ax_k_gam.set_xlabel(e_lab)
    ax_k_gam.set_ylabel(r'$K_\gamma$')

    fig_gains.tight_layout()

    plt.show()
