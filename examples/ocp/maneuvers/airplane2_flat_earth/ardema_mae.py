import numpy as np
import scipy as sp
import casadi as ca

from airplane2_aero_atm import g, weight0, dens_fun, sped_fun, s_ref, CD0_fun, CD2_fun, thrust_fun, Isp

_f_zero_converged_flag = 1
d2r = np.pi/180

# This example is Mark Ardema's solution to the Minimum Time to Climb problem by matched asymptotic expansions.
# https://doi.org/10.2514/3.7161

# State Variables
E = ca.MX.sym('E')
h = ca.MX.sym('h')
gam = ca.MX.sym('gam')
m = ca.MX.sym('m')
v = (2 * (E - g*h))**0.5

# Control Variables
lift = ca.MX.sym('lift')

# Expressions
rho = dens_fun(h)
mach = v / sped_fun(h)
qdyn = 0.5 * rho * v**2
CD0 = CD0_fun(mach)
CD2 = CD2_fun(mach)
D0 = qdyn * s_ref * CD0
D1 = CD2 / (qdyn * s_ref)
thrust = thrust_fun(mach, h)
drag = CD0 * qdyn * s_ref + CD2/(qdyn * s_ref) * lift**2
weight = m * g

# Dynamics
dhdt = v * ca.sin(gam)
dEdt = v * (thrust - drag) / m
dgamdt = (lift - weight * ca.cos(gam))/(m*v)

# Necessary Conditions (MAXIMUM)
lam_E = ca.MX.sym('lam_E')
lam_h = ca.MX.sym('lam_h')
lam_gam = ca.MX.sym('lam_gam')

hamiltonian = 1 + lam_h * dhdt + lam_E * dEdt + lam_gam * dgamdt

dlam_hdt = -ca.jacobian(hamiltonian, h)
dlam_gamdt = -ca.jacobian(hamiltonian, gam)

# Fast dynamics
z_vec = ca.vcat((h, gam, lam_h, lam_gam))
dzdt = ca.vcat((dhdt, dgamdt, dlam_hdt, dlam_gamdt))
L_cl = lam_gam/lam_E * rho * s_ref / (4 * CD2)
dzdt_cl = ca.substitute(dzdt, lift, L_cl)
G = ca.jacobian(dzdt_cl, z_vec)
G_fun = ca.Function(
    'G', (m, E, h, gam, lam_E, lam_h, lam_gam), (G,), ('m', 'E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam'), ('G',)
)
L_cl_fun = ca.Function(
    'L', (m, E, h, gam, lam_E, lam_h, lam_gam), (L_cl,), ('m', 'E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam'), ('G',)
)

# Refactor fast dynamics to use E instead of t
dzdE = dzdt / dEdt
dzdE_cl = ca.substitute(dzdE, lift, L_cl)
GE = ca.jacobian(dzdE_cl, z_vec)
GE_fun = ca.Function(
    'GE', (m, E, h, gam, lam_E, lam_h, lam_gam), (GE,), ('m', 'E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam'), ('GE',)
)

# Energy State Solution
obj_fun_full = ca.Function('F', (m, E, h, gam, lift), (ca.jacobian(dEdt, h),), ('m', 'E', 'h', 'gam', 'L'), ('F',))

L_es = weight * np.cos(gam)
drag_es = ca.substitute(drag, lift, L_es)
dEdt_es = v * (thrust - drag_es) / m
dmdt = - thrust / Isp

obj_es = ca.substitute(dEdt_es, gam, 0.)
zero_es = ca.jacobian(obj_es, h)
grad_es = ca.jacobian(zero_es, h)

obj_fun_es = ca.Function('F', (m, E, h), (obj_es,), ('m', 'E', 'h'), ('F',))
zero_fun_es = ca.Function('Fz', (m, E, h), (zero_es,), ('m', 'E', 'h'), ('Fz',))
grad_fun_es = ca.Function('DFz', (m, E, h), (grad_es,), ('m', 'E', 'h'), ('DFz',))

# Use higher derivatives to calculate h, gam, lam_h, lam_gam to ensure d(Eh)/dt = 0 for all time
# -> d2(Eh)/dt2 = d3(Eh)/dt3 = d4(Eh)/dt4 = 0
lam_E_hd = rho * s_ref * lam_gam / (4 * CD2 * lift)  # From Hu = 0

# H = 0 has two roots for L. The smaller root corresponds with L = W at energy climb and is therefore assumed.
half_b = (1 + lam_h * v * ca.sin(gam)) * m * v / lam_gam - m*g*ca.cos(gam)
c = (thrust - D0)/D1
lift_hd = -half_b - (half_b ** 2 - c) ** 0.5
lam_E_hd_cl = ca.substitute(lam_E_hd, lift, lift_hd)

lam_h_hd = 0.  # By assumption. Prevents having to take too many Jacobians

x_hd = ca.vcat((m, E, h, gam, lam_gam))  # Variables with Dynamics
z_hd = ca.vcat((h, gam, lam_gam))  # Design variables (m, E fixed)
dx_hd_dt = ca.vcat((dmdt, dEdt, dhdt, dgamdt, dlam_gamdt))
dx_hd_dt_cl = dx_hd_dt
dEdt_hd_cl = dEdt
lift_hd_cl = lift_hd
for var, val in zip((lam_E, lift, lam_h), (lam_E_hd_cl, lift_hd, 0.)):
    dx_hd_dt_cl = ca.substitute(dx_hd_dt_cl, var, val)
    dEdt_hd_cl = ca.substitute(dEdt_hd_cl, var, val)
    lift_hd_cl = ca.substitute(lift_hd_cl, var, val)

f1_hd = ca.jacobian(dEdt_hd_cl, h)
f2_hd = ca.jacobian(f1_hd, x_hd) @ dx_hd_dt_cl
f3_hd = ca.jacobian(f2_hd, x_hd) @ dx_hd_dt_cl

zero_hd = ca.vcat((f1_hd, f2_hd, f3_hd))
obj_hd = 0.5 * zero_hd.T @ zero_hd
hess_hd = ca.jacobian(zero_hd, z_hd)

obj_fun_hd = ca.Function(
    'R', (m, E, z_hd), (obj_hd,), ('m', 'E', 'x'), ('R',)
)
zero_fun_hd = ca.Function(
    'F', (m, E, z_hd), (zero_hd,), ('m', 'E', 'x'), ('F',)
)
hess_fun_hd = ca.Function(
    'H', (m, E, z_hd), (hess_hd,), ('m', 'E', 'x'), ('H',)
)
lift_fun_hd = ca.Function(
    'L', (m, E, z_hd), (lift_hd_cl,), ('m', 'E', 'x'), ('L',)
)

# Get G function in terms of states by pre-calculating u, lam ess
DL_es = 2 * CD2 / (qdyn * s_ref) * L_es
lam_E_es = -1 / dEdt_es
lam_gam_es = lam_E_es * v ** 2 * DL_es
lam_h_es = 0.
G_es_fun = ca.Function(
    'G', (m, E, h, gam), (G_fun(m, E, h, gam, lam_E_es, lam_h_es, lam_gam_es),), ('m', 'E', 'h', 'gam'), ('G',)
)
GE_es_fun = ca.Function(
    'GE', (m, E, h, gam), (GE_fun(m, E, h, gam, lam_E_es, lam_h_es, lam_gam_es),), ('m', 'E', 'h', 'gam'), ('GE',)
)
lam_gam_es_fun = ca.Function(
    'lam_gam', (m, E, h, gam), (lam_gam_es,), ('m', 'E', 'h', 'gam'), ('lam_gam',)
)


def solve_zero(mass, energy, h_guess):
    _x_val, _, _flag, __ = sp.optimize.fsolve(
        func=lambda _x: np.asarray(zero_fun_es(mass, energy, _x[0])).flatten(),
        x0=np.array((h_guess,)),
        fprime=lambda _x: np.asarray(grad_fun_es(mass, energy, _x[0])), full_output=True
    )

    return _x_val[0], _flag


def find_climb_path(mass, energy, h_guess):
    _h, _flag = solve_zero(mass, energy, h_guess)

    if _flag != _f_zero_converged_flag:
        h_search_vals = np.linspace(0., 2*h_guess, 1000)
        fzero_search_vals = np.empty(h_search_vals.shape)
        for jdx, h_search_val in enumerate(h_search_vals):
            fzero_search_vals[jdx] = zero_fun_es(mass, energy, h_search_val)
        sign_flips = np.not_equal(np.sign(fzero_search_vals[1:]), np.sign(fzero_search_vals[:-1]))

        if np.any(sign_flips):
            idx_sign_flips = np.where(sign_flips)[0]
            sol_list = []
            for idx_sign_flip in idx_sign_flips:
                _h_guess_new = 0.5 * (h_search_vals[idx_sign_flip] + h_search_vals[idx_sign_flip + 1])
                _h_new, _flag_new = solve_zero(mass, energy, _h_guess_new)
                if _flag_new == _f_zero_converged_flag:
                    sol_list.append(_h_new)

            if len(sol_list) > 0:
                sol_array = np.array(sol_list)
                sol_distance = (sol_array - h_guess) ** 2
                _h = sol_array[np.where(sol_distance == np.min(sol_distance))[0][0]]

    _v = (2 * (energy - g * _h))**0.5
    _mach = _v / float(sped_fun(_h))
    _qdyn_s_ref = 0.5 * float(dens_fun(_h)) * _v**2 * s_ref
    _thrust = float(thrust_fun(_mach, _h))
    _CD0 = float(CD0_fun(_mach))
    _CD2 = float(CD2_fun(_mach))
    _lift = g * mass
    _drag = _qdyn_s_ref * _CD0 + _CD2 / _qdyn_s_ref * _lift**2
    _DL = 2 * _CD2/_qdyn_s_ref * _lift
    _dEdt = _v * (_thrust - _drag) / mass
    _lam_E = -1 / _dEdt
    _lam_gam = _lam_E * _v**2 * _DL
    _lam_h = 0.
    _gam = 0.
    _G = np.asarray(G_fun(
        mass, energy, _h, _gam, _lam_E, _lam_h, _lam_gam
    ))
    _GE = np.asarray(GE_fun(
        mass, energy, _h, _gam, _lam_E, _lam_h, _lam_gam
    ))
    _climb_dict = {
        'm': mass, 'E': energy, 'h': _h, 'V': _v, 'gam': _gam,
        'lam_E': _lam_E, 'lam_h': _lam_h, 'lam_gam': _lam_gam,
        'L': _lift, 'D': _drag, 'T': _thrust, 'M': _mach, 'qdyn_s_ref': _qdyn_s_ref,
        'G': _G, 'GE': _GE
    }
    return _climb_dict


def newton_search_x_hd(
        _m, _E, _h, _gam=None, _lam_gam=None, max_iter=100, relax_fac_min=2**-5,
        _h_bounds=None, _gam_bounds=None, _lift_bounds=None
):
    _success = False

    # Default to energy state values
    if _gam is None:
        _gam = 0.
    if _lam_gam is None:
        _lam_gam = float(lam_gam_es_fun(_m, _E, _h, _gam))

    _x = np.vstack((_h, _gam, _lam_gam))
    _f = float(obj_fun_hd(_m, _E, _x))
    _lift = float(lift_fun_hd(_m, _E, _x))

    # Generous, but feasible, default bounds for the MTC problem
    if _h_bounds is None:
        _h_bounds = np.array((0., 0.95 * _E/g))
    else:
        _h_bounds = _h_bounds.reshape((-1,))
    if _gam_bounds is None:
        _gam_bounds = np.array((0., 45 * d2r))
    else:
        _gam_bounds = _gam_bounds.reshape((-1,))
    if _lift_bounds is None:
        _lift_bounds = np.array((0., 3. * _m * g))
    else:
        _lift_bounds = _lift_bounds.reshape((-1,))

    for _ in range(max_iter):
        _x_last = _x.copy()
        _f_last = _f
        _hess = np.asarray(hess_fun_hd(_m, _E, _x_last))
        _grad = np.asarray(zero_fun_hd(_m, _E, _x_last))

        # Full step
        _relax_fac = 1.
        _x = _x_last - _relax_fac * np.linalg.solve(_hess, _grad)
        _f = float(obj_fun_hd(_m, _E, _x))
        _lift = float(lift_fun_hd(_m, _E, _x))
        feasible = _h_bounds[0] < _x[0, 0] < _h_bounds[1] \
            and _gam_bounds[0] < _x[1, 0] < _gam_bounds[1] \
            and _lift_bounds[0] < _lift < _lift_bounds[1]

        while not feasible or _f > _f_last or np.any(np.isnan(_x)) or np.any(np.isinf(_x)):
            # Backtracking Line Search
            _relax_fac = 0.5 * _relax_fac

            if _relax_fac < relax_fac_min:
                break

            _x = _x_last - _relax_fac * np.linalg.solve(_hess, _grad)
            _f = float(obj_fun_hd(_m, _E, _x))
            _lift = float(lift_fun_hd(_m, _E, _x))
            feasible = _h_bounds[0] < _x[0, 0] < _h_bounds[1] \
                and _gam_bounds[0] < _x[1, 0] < _gam_bounds[1] \
                and _lift_bounds[0] < _lift < _lift_bounds[1]

        if _f < 1e-8:
            # Success, stop iterations
            _success = True
            break
        elif _relax_fac < relax_fac_min:
            # Failure, stop iteration
            break
        else:
            # Step ok, but not yet done.
            _relax_fac = 1.

    return _x.flatten(), _lift, _success


def newton_seach_multiple_start(_m, _E, _h=None, _gam=None, _lam_gam=None, max_iter=100, relax_fac_min=2**-5,
    _h_bounds=None, _gam_bounds=None, _lift_bounds=None):

    if _h_bounds is None:
        _h_bounds = np.array((0., 0.95 * _E / g))
    else:
        _h_bounds = _h_bounds.reshape((-1,))

    if _h is None:
        _h = _h_bounds[0]

    x, lift_val, success = newton_search_x_hd(
        _m, _E, _h, _gam, _lam_gam, max_iter, relax_fac_min, _h_bounds, _gam_bounds, _lift_bounds
    )

    if not success:
        # Try other initial conditions for h
        h_vals = np.linspace(_h_bounds[0], _h_bounds[1], 100)
        for _h in h_vals:
            x, lift_val, success = newton_search_x_hd(
                _m, _E, _h, _gam, _lam_gam, max_iter, relax_fac_min, _h_bounds, _gam_bounds, _lift_bounds
            )
            if success:
                break
    return x, lift_val, success


if __name__ == '__main__':
    # NUMERICAL SOLUTION [from ref] ------------------------------------------------------------------------------------
    from matplotlib import pyplot as plt

    r2d = 180 / np.pi

    mass0 = weight0 / g
    h0 = 40e3
    mach0 = 0.5
    v0 = mach0 * float(sped_fun(h0))
    E0 = g * h0 + 0.5 * v0**2

    # Outer Solution
    _outer_dict = find_climb_path(mass0, E0, 8.e3)

    # Other option: start with h = h_max, gam = 0. and work iteratively.
    x_es0, success = newton_search_x_hd(mass0, E0, 0.25*E0/g)

    # Linearization about E
    G00 = np.asarray(G_fun(
        mass0, E0, _outer_dict['h'], _outer_dict['gam'], _outer_dict['lam_E'], _outer_dict['lam_h'], _outer_dict['lam_gam']
    ))
    eig_G00, eig_vec_G00 = np.linalg.eig(G00)
    idx_stable = np.where(np.real(eig_G00) < 0)
    idx_unstable = np.where(np.real(eig_G00) > 0)
    eig_vec_stable = eig_vec_G00[:, idx_stable[0]]
    eig_vec_unstable = eig_vec_G00[:, idx_unstable[0]]
    Vs = np.hstack((np.real(eig_vec_stable[:, 0].reshape(-1, 1)), np.imag(eig_vec_stable[:, 0].reshape(-1, 1))))
    Vu = np.hstack((np.real(eig_vec_unstable[:, 0].reshape(-1, 1)), np.imag(eig_vec_unstable[:, 0].reshape(-1, 1))))
    Vs_known = Vs[(0, 3), :]  # h0 fixed, gam0 free -> lam_gam0 = 0
    Vs_unknown = Vs[(1, 2), :]

    # Solve for unknown initial conditions
    dh0 = h0 - _outer_dict['h']
    dlam_gam0 = 0.
    z0_known = np.vstack((dh0, dlam_gam0))
    z0_unknown = Vs_unknown @ np.linalg.solve(Vs_known, z0_known)
    dgam0 = z0_unknown[0, 0]
    dlam_h0 = z0_unknown[1, 0]
    gam0 = dgam0

    z0 = np.array((dh0, dgam0, dlam_h0, dgam0))
    sol_ivp = sp.integrate.solve_ivp(fun=lambda t, z: (G00 @ z).flatten(), t_span=np.array((0., 60.)), y0=z0)

    # Terminal Conditions
    hf = 80e3
    machf = 2.
    gamf = 0.
    vf = machf * float(sped_fun(hf))
    Ef = g * hf + 0.5 * vf**2
    outer_dictf = find_climb_path(mass0, Ef, 30e3)
    GEf = outer_dictf['GE']
    Ego = Ef - E0

    # Solve for NOC to achieve terminal conditions
    dhf = hf - outer_dictf['h']
    dgamf = gamf - outer_dictf['gam']
    zf_known = np.vstack((dhf, dgamf))
    eig_GEf, eig_vec_GEf = np.linalg.eig(GEf)
    idx_stable = np.where(np.real(eig_GEf) > 0)
    eig_stable = eig_GEf[idx_stable]
    eig_vec_stable = eig_vec_GEf[:, idx_stable[0]]
    Vsf = np.hstack((np.real(eig_vec_stable[:, 0].reshape(-1, 1)), np.imag(eig_vec_stable[:, 0].reshape(-1, 1))))
    _rotation_Vs = np.imag(eig_stable[0]) * Ego
    _c_rot = np.cos(_rotation_Vs)
    _s_rot = np.sin(_rotation_Vs)
    _DCM = np.vstack(((_c_rot, _s_rot), (-_s_rot, _c_rot)))
    Vs = Vsf @ _DCM
    Vsf_known = Vsf[(0, 1), :]  # hf, gamf fixed
    Vs_unknown = Vs[(2, 3), :]  # lam_h0, lam_gam0 free
    zf_Ego = np.exp(-np.real(eig_stable[0]) * Ego) * Vs @ np.linalg.solve(Vsf_known, zf_known)


    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(sol_ivp.t, sol_ivp.y[0, :] * g)
    ax1.grid()
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('dh [ft]')

    ax2 = fig.add_subplot(212)
    ax2.plot(sol_ivp.t, sol_ivp.y[1, :] * r2d)
    ax2.grid()
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('dgam [deg]')

    fig.tight_layout()

    plt.show()
