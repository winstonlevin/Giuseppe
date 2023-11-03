from copy import deepcopy
import casadi as ca
import numpy as np
import scipy as sp
import pickle

import giuseppe

from airplane2_aero_atm import g, mass as mass0, s_ref, CD0_fun, CD1, CD2_fun, atm, lut_data,\
    load_max, dens_fun, sped_fun, max_ld_fun_mach, gam_qdyn0, CL0, CLa_fun, alpha_max, qdyn_max, thrust_fun, Isp

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
v_scale_val = 1e4
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
ocp.add_constraint('terminal', gam - gam_f)

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

    # Find velocity where h = 0
    h_f = 0.
    rho_f = atm.density(h_f)
    a_f = atm.speed_of_sound(h_f)
    v_f = binary_search(
        10., 1e3,
        lambda _v: rho_f * _v**2 * s_ref * max_ld_fun_mach(_v/a_f)['CL'] / (2*weight0) - 1, 0.
    )
    mach_f = v_f / atm.speed_of_sound(h_f)

    # Find altitude where Mach = 2.5
    mach_0 = 2.5
    CL_0 = float(max_ld_fun_mach(mach_0)['CL'])
    h_0 = binary_search(
        1e3, 90e3,
        lambda _h: atm.density(_h) * (mach_0 * atm.speed_of_sound(_h))**2 * s_ref * CL_0 / (2*weight0) - 1, 0.
    )

    def ctrl_law(_t, _x, _p, _k):
        _h = _x[0] * h_scale_val
        _xd = _x[1] * xd_scale_val
        _v = _x[2] * v_scale_val
        _gam = _x[3]

        _qdyn_s_ref = 0.5 * atm.density(_h) * _v**2 * s_ref
        _CL_gam0 = weight0 / _qdyn_s_ref
        return np.array((_CL_gam0,))


    # Flat earth costates
    h_guess = h_0
    v_guess = mach_0 * atm.speed_of_sound(h_guess)
    gam_guess = gam_qdyn0(h_guess, v_guess)
    CD_guess = float(CD0_fun(mach_0)) + CD1 * CL_0 + float(CD2_fun(mach_0)) * CL_0**2
    CDu_guess = CD1 + 2 * float(CD2_fun(mach_0)) * CL_0
    qdyn_s_ref_guess = 0.5 * atm.density(h_guess) * v_guess**2 * s_ref
    drag_guess = qdyn_s_ref_guess * CD_guess
    lam_e_glide = -mass / drag_guess
    lam_h_glide = 0.
    lam_gam_glide = lam_e_glide * v_guess**2 * CDu_guess

    # Convert E,h,gam costates to h,V,gam costates
    # d(E)/dt = g d(h)/dt + V d(V)/dt
    # -> lam_v = V lam_E
    # -> lam_h = lam_h + g lam_E
    lam_v_guess = lam_e_glide * v_guess
    lam_h_guess = lam_h_glide + lam_e_glide / g
    lam_gam_guess = lam_gam_glide
    lam_xd_guess = 0.

    costate_scales = np.array((h_scale_val/xd_scale_val, 1., v_scale_val/xd_scale_val, 1./xd_scale_val))
    lam_guess = np.array((lam_h_guess, lam_xd_guess, lam_v_guess, lam_gam_guess)) * costate_scales

    guess = giuseppe.guess_generation.auto_propagate_guess(
        adiff_dual, control=ctrl_law, t_span=np.linspace(0., 25., 5),
        initial_states=np.array((h_guess / h_scale_val, 0., v_guess / v_scale_val, gam_guess)), fit_states=False,
        immutable_constants=('h_scale', 'v_scale'), initial_costates=lam_guess, fit_adjoints=False
    )

    with open('guess_range.data', 'wb') as file:
        pickle.dump(guess, file)

    seed_sol = num_solver.solve(guess)

    with open('seed_sol_range.data', 'wb') as file:
        pickle.dump(seed_sol, file)

    idx_hf = adiff_dual.annotations.constants.index('h_f')
    idx_vf = adiff_dual.annotations.constants.index('v_f')
    idx_gamf = adiff_dual.annotations.constants.index('gam_f')
    idx_h0 = adiff_dual.annotations.constants.index('h_0')
    idx_v0 = adiff_dual.annotations.constants.index('v_0')
    idx_gam0 = adiff_dual.annotations.constants.index('gam_0')

    # Create altitude/Mach interpolant
    hf_seed = seed_sol.x[0, -1] * h_scale_val
    vf_seed = seed_sol.x[2, -1] * v_scale_val
    machf_seed = vf_seed / atm.speed_of_sound(hf_seed)
    n_coninuations = 100
    mach_vals = np.linspace(mach_f, machf_seed, n_coninuations)
    h_vals = np.empty(mach_vals.shape)
    for idx, mach_val in enumerate(mach_vals):
        h_vals[idx] = binary_search(
            h_f, h_0,
            lambda _h: atm.density(_h) * (mach_val * atm.speed_of_sound(_h)) ** 2 * s_ref * CL_0 / (2 * weight0) - 1, 0.
        )
    h_interp = sp.interpolate.PchipInterpolator(mach_vals, h_vals)

    def glide_slope_continuation_xf(previous_sol, frac_complete):
        _mach_f = machf_seed + frac_complete * (mach_f - machf_seed)
        _hf = h_interp(_mach_f)
        _vf = _mach_f * atm.speed_of_sound(_hf)
        _gamf = gam_qdyn0(_hf, _vf)

        previous_sol.k[idx_hf] = _hf
        previous_sol.k[idx_vf] = _vf
        previous_sol.k[idx_gamf] = _gamf
        return previous_sol.k

    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_custom_series(n_coninuations, glide_slope_continuation_xf, 'Glide Slope (tf)')
    sol_set = cont.run_continuation()
    sol_set.save('sol_set_range.data')

    hf = sol_set.solutions[-1].x[0, -1] * h_scale_val
    xdf = sol_set.solutions[-1].x[1, -1] * xd_scale_val
    vf = sol_set.solutions[-1].x[2, -1] * v_scale_val
    gamf = sol_set.solutions[-1].x[3, -1]
    ef = g * hf + 0.5 * vf ** 2
    h0 = sol_set.solutions[-1].x[0, 0] * h_scale_val
    v0 = sol_set.solutions[-1].x[2, 0] * v_scale_val
    gam0 = sol_set.solutions[-1].x[3, 0]
    e0 = g * h0 + 0.5 * v0 ** 2

    # Sweep Solution Space (perturb initial h/v with gam0 = 0, e0 const.)
    if SWEEP_SOLUTION_SPACE:
        def generate_energy_sweep_continuation(_h0_0, _h0_f):
            def energy_sweep_continuation(previous_sol, frac_complete):
                _constants = previous_sol.k.copy()

                _h0 = previous_sol.k[idx_h0]
                _v0 = previous_sol.k[idx_v0]
                _e0 = g * _h0 + 0.5 * _v0**2

                _h0_next = _h0_0 + frac_complete * (_h0_f - _h0_0)
                _v0_next = (2 * (_e0 - g * _h0_next)) ** 0.5
                _gam0_next = gam_qdyn0(_h0_next, _v0_next)
                _constants[idx_h0] = _h0_next
                _constants[idx_v0] = _v0_next
                _constants[idx_gam0] = _gam0_next
                return _constants
            return energy_sweep_continuation

        # Maximum altitude where stall occurs
        h0_max = binary_search(
            145e3, h0,
            lambda _h:
            atm.density(_h) * (e0 - g*_h) * s_ref
            * float(CL0 + CLa_fun((2*(e0 - g*_h)**0.5/atm.speed_of_sound(_h))) * alpha_max)
            / weight0 - 1, 0.
        )

        # Minimum altitude where initial dynamic pressure is maximized
        h0_min = binary_search(
            100., h0,
            lambda _h: atm.density(_h) * (e0 - g*_h) / qdyn_max - 1, 0.
        )

        # Get Low altitude solution
        cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
        cont.add_custom_series(
            100, generate_energy_sweep_continuation(h0, h0_min), 'Const. energy', keep_bisections=False
        )
        sol_set_sweep1 = cont.run_continuation()

        cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
        cont.add_custom_series(
            100, generate_energy_sweep_continuation(h0, h0_max), 'Const. energy', keep_bisections=False
        )
        sol_set_sweep2 = cont.run_continuation()

        sol_set_sweep = deepcopy(sol_set_sweep2)
        sol_set_sweep.solutions = deepcopy([
            sol_set_sweep1.solutions[-1], sol_set_sweep1.solutions[0], sol_set_sweep2.solutions[-1]
        ])

        sol_set_sweep.save('sol_set_range_sweep.data')

        h0_case1 = sol_set_sweep.solutions[0].x[0, 0] * h_scale_val
        xdf_case1 = sol_set_sweep.solutions[0].x[1, -1] * xd_scale_val
        v0_case1 = sol_set_sweep.solutions[0].x[2, 0] * v_scale_val
        gam0_case1 = sol_set_sweep.solutions[0].x[3, 0]
        mach0_case1 = v0_case1 / atm.speed_of_sound(h0_case1)

        h0_case3 = sol_set_sweep.solutions[-1].x[0, 0] * h_scale_val
        xdf_case3 = sol_set_sweep.solutions[-1].x[1, -1] * xd_scale_val
        v0_case3 = sol_set_sweep.solutions[-1].x[2, 0] * v_scale_val
        gam0_case3 = sol_set_sweep.solutions[-1].x[3, 0]
