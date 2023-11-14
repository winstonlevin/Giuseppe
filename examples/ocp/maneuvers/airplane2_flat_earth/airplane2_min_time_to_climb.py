from copy import deepcopy
import casadi as ca
import numpy as np
import scipy as sp
import pickle

import giuseppe

from airplane2_aero_atm import g, mass as mass0, s_ref, CD0_fun, CD1, CD2_fun, atm, lut_data,\
    load_max, dens_fun, sped_fun, max_ld_fun_mach, gam_qdyn0, CL0, CLa_fun, alpha_max, qdyn_max, thrust_fun, Isp
from ardema_mae import find_climb_path

SWEEP_SOLUTION_SPACE = True


ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.MX)

# Independent Variables
t = ca.MX.sym('t', 1)
ocp.set_independent(t)

# State Variables
h_nd = ca.MX.sym('h_nd', 1)
h_scale = ca.MX.sym('h_scale', 1)
v_nd = ca.MX.sym('v_nd', 1)
v_scale = ca.MX.sym('v_scale', 1)
gam = ca.MX.sym('gam', 1)
m_nd = ca.MX.sym('m_nd', 1)
m_scale = ca.MX.sym('m_scale', 1)

h = h_nd * h_scale
v = v_nd * v_scale
mass = m_nd * m_scale

h_scale_val = 1e4
v_scale_val = 1e3
m_scale_val = mass0
ocp.add_constant(h_scale, h_scale_val)
ocp.add_constant(m_scale, m_scale_val)
ocp.add_constant(v_scale, v_scale_val)

# Atmosphere Func
_, __, rho = atm.get_ca_atm_expr(h)
sped = atm.get_ca_speed_of_sound_expr(h)
mach = v / sped

# Add Controls
load = ca.MX.sym('load', 1)
ocp.add_control(load)

# Expressions
dyn_pres = 0.5 * rho * v ** 2
CD0 = CD0_fun(mach)
CD2 = CD2_fun(mach)
weight = mass * g
lift = load * weight
drag = dyn_pres * s_ref * CD0 + CD1 * lift + CD2/(dyn_pres * s_ref) * lift**2
thrust = thrust_fun(mach, h)

# Add States & EOMs
ocp.add_state(h_nd, v * ca.sin(gam) / h_scale)
ocp.add_state(v_nd, ((thrust - drag) / mass - g * ca.sin(gam)) / v_scale)
ocp.add_state(gam, lift / (mass * v) - g/v * ca.cos(gam))
ocp.add_state(m_nd, -thrust / Isp / m_scale)

# Cost
ocp.set_cost(0., 1., 0.)

# Boundary Values
h_0 = ca.MX.sym('h_0', 1)
v_0 = ca.MX.sym('v_0', 1)
gam_0 = ca.MX.sym('gam_0', 1)
m_0 = ca.MX.sym('m_0', 1)

h_0_val = 260_000
v_0_val = 25_600
e_0_val = g * h_0_val + 0.5 * v_0_val**2
gam_0_val = -1 / 180 * np.pi

# ocp.add_constant(e_0, e_0_val)
ocp.add_constant(h_0, h_0_val)
ocp.add_constant(v_0, v_0_val)
ocp.add_constant(gam_0, gam_0_val)
ocp.add_constant(m_0, mass0)

h_f = ca.MX.sym('h_f', 1)
v_f = ca.MX.sym('v_f', 1)
gam_f = ca.MX.sym('gam_f', 1)

ocp.add_constant(h_f, 0.)
ocp.add_constant(v_f, 10.)
ocp.add_constant(gam_f, 0.)

# Initial state constrained
ocp.add_constraint('initial', t)
ocp.add_constraint('initial', h - h_0)
ocp.add_constraint('initial', v - v_0)
ocp.add_constraint('initial', gam - gam_0)
ocp.add_constraint('initial', mass - m_0)

# Terminal state
ocp.add_constraint('terminal', h - h_f)
ocp.add_constraint('terminal', v - v_f)
# ocp.add_constraint('terminal', gam - gam_f)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(
        adiff_dual, verbose=False, max_nodes=100, node_buffer=10
    )

if __name__ == '__main__':
    def binary_search(_x_min, _x_max, _f, _f_val_target, max_iter: int = 1000, tol: float = 1e-3):
        increasing = _f(_x_max) > _f(_x_min)
        if increasing:
            def _f_wrapped(_x):
                return _f(_x)
        else:
            _f_val_target = -_f_val_target

            def _f_wrapped(_x):
                return -_f(_x)

        for _ in range(max_iter):
            # Binary search
            _x_guess = 0.5 * (_x_min + _x_max)
            _f_val = _f_wrapped(_x_guess)
            if _f_val < _f_val_target:
                # x too low, try higher
                _x_min = _x_guess
            else:
                # x too high, try lower
                _x_max = _x_guess
            if _x_max - _x_min < tol:
                break

        _x_guess = 0.5 * (_x_min + _x_max)
        return _x_guess

    # Initial Conditions
    mach0 = 0.5
    h0 = 40e3
    v0 = mach0 * atm.speed_of_sound(h0)
    gam0 = 0.
    energy0 = g * h0 + 0.5 * v0**2
    _climb_dict0 = find_climb_path(mass0, energy0, h_guess=8e3)

    # Terminal Conditions
    machf = 2.0
    hf = 80e3
    vf = machf * atm.speed_of_sound(hf)
    energyf = g * hf + 0.5 * vf**2
    _climb_dictf = find_climb_path(mass0, energyf, h_guess=40e3)

    def ctrl_law(_t, _x, _p, _k):
        _h = _x[0] * h_scale_val
        _v = _x[1] * v_scale_val
        _gam = _x[2]
        _m = _x[3]

        _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
        _CL_gam0 = g*_m / _qdyn_s_ref
        return np.array((_CL_gam0,))


    # Flat earth costates
    h_guess = _climb_dict0['h']
    v_guess = _climb_dict0['V']
    gam_guess = _climb_dict0['gam']
    lam_e_ec = _climb_dict0['lam_E']
    lam_h_ec = _climb_dict0['lam_h']
    lam_gam_ec = _climb_dict0['lam_gam']
    lam_m_ec = Isp / _climb_dict0['T']

    # Convert E,h,gam costates to h,V,gam costates
    # d(E)/dt = g d(h)/dt + V d(V)/dt
    # -> lam_v = V lam_E
    # -> lam_h = lam_h + g lam_E
    lam_v_guess = lam_e_ec * v_guess
    lam_h_guess = lam_h_ec + lam_e_ec / g
    lam_gam_guess = lam_gam_ec
    lam_m_guess = lam_m_ec

    costate_scales = np.array((h_scale_val, v_scale_val, 1., m_scale_val))
    lam_guess = np.array((lam_h_guess, lam_v_guess, lam_gam_guess, lam_m_guess)) * costate_scales

    guess = giuseppe.guess_generation.auto_propagate_guess(
        adiff_dual, control=ctrl_law, t_span=np.linspace(0., 5., 5),
        initial_states=np.array((h_guess / h_scale_val, v_guess / v_scale_val, gam_guess, mass0/m_scale_val)),
        fit_states=False, immutable_constants=('h_scale', 'v_scale', 'm_scale'),
        initial_costates=lam_guess, fit_adjoints=False
    )

    with open('guess_mtc.data', 'wb') as file:
        pickle.dump(guess, file)

    seed_sol = num_solver.solve(guess)

    with open('seed_sol_mtc.data', 'wb') as file:
        pickle.dump(seed_sol, file)

    idx_hf = adiff_dual.annotations.constants.index('h_f')
    idx_vf = adiff_dual.annotations.constants.index('v_f')
    idx_gamf = adiff_dual.annotations.constants.index('gam_f')
    idx_h0 = adiff_dual.annotations.constants.index('h_0')
    idx_v0 = adiff_dual.annotations.constants.index('v_0')
    idx_gam0 = adiff_dual.annotations.constants.index('gam_0')

    # Create altitude/velocity interpolant
    hf_seed = seed_sol.x[0, -1] * h_scale_val
    vf_seed = seed_sol.x[1, -1] * v_scale_val
    ef_seed = g * hf_seed + 0.5 * vf_seed**2
    n_coninuations = 25
    e_vals = np.linspace(ef_seed, energyf, n_coninuations)
    h_vals = np.empty(e_vals.shape)
    v_vals = np.empty(h_vals.shape)
    h_previous = hf_seed
    for idx, e_val in enumerate(e_vals):
        _out_dict = find_climb_path(mass0, e_val, h_guess=h_previous)
        h_previous = _out_dict['h']
        h_vals[idx] = _out_dict['h']
        v_vals[idx] = _out_dict['V']
    h_interp = sp.interpolate.PchipInterpolator(v_vals, h_vals)


    def energy_state_continuation(previous_sol, frac_complete):
        _constants = previous_sol.k.copy()
        _vf = vf_seed + frac_complete * (v_vals[-1] - vf_seed)
        _hf = h_interp(_vf)
        _constants[idx_hf] = _hf
        _constants[idx_vf] = _vf
        return _constants

    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_linear_series(3, {'h_f': h_interp(vf_seed), 'gam_f': 0.})
    cont.add_custom_series(n_coninuations, energy_state_continuation, 'Energy State (hf, vf)')
    cont.add_linear_series(15, {'h_0': h0, 'v_0': v0})
    sol_set = cont.run_continuation()
    sol_set.save('sol_set_mtc.data')

    cont = giuseppe.continuation.ContinuationHandler(num_solver, sol_set.solutions[-1])
    cont.add_linear_series(25, {'h_f': hf, 'v_f': vf})
    sol_set_xf = cont.run_continuation()
    sol_set_xf.save('sol_set_mtc_xf.data')

    hf = sol_set.solutions[-1].x[0, -1] * h_scale_val
    mf = sol_set.solutions[-1].x[3, -1] * m_scale_val
    vf = sol_set.solutions[-1].x[1, -1] * v_scale_val
    gamf = sol_set.solutions[-1].x[2, -1]
    ef = g * hf + 0.5 * vf ** 2
    h0 = sol_set.solutions[-1].x[0, 0] * h_scale_val
    v0 = sol_set.solutions[-1].x[1, 0] * v_scale_val
    gam0 = sol_set.solutions[-1].x[2, 0]
    e0 = g * h0 + 0.5 * v0 ** 2
