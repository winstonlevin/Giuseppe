from copy import deepcopy
import casadi as ca
import numpy as np
import scipy as sp
import pickle

import giuseppe

from airplane2_aero_atm import mu, re, g0, mass, s_ref, CD0_fun, CD1, CD2_fun, atm, lut_data, mach_boundary_thickness,\
    load_max, alpha_max, CL0, CLa_fun, dens_fun, sped_fun
from glide_slope import get_glide_slope

SWEEP_SOLUTION_SPACE = True


ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.MX)

# Independent Variables
t = ca.MX.sym('t', 1)
ocp.set_independent(t)

# Constants
weight0 = g0 * mass

# State Variables
h_nd = ca.MX.sym('h_nd', 1)
h_scale = ca.MX.sym('h_scale', 1)
tha = ca.MX.sym('tha', 1)
v_nd = ca.MX.sym('v_nd', 1)
v_scale = ca.MX.sym('v_scale', 1)
gam = ca.MX.sym('gam', 1)

h = h_nd * h_scale
v = v_nd * v_scale

h_scale_val = 1e4
v_scale_val = 1e4
ocp.add_constant(h_scale, h_scale_val)
ocp.add_constant(v_scale, v_scale_val)

# Atmosphere Func
_, __, rho = atm.get_ca_atm_expr(h)
sped = atm.get_ca_speed_of_sound_expr(h)
mach = v / sped

# Add Controls
lift_nd = ca.MX.sym('lift_nd', 1)
lift = lift_nd * weight0

ocp.add_control(lift_nd)

# Expressions
r = re + h
g = mu / r**2
dyn_pres = 0.5 * rho * v ** 2
CD0 = CD0_fun(mach)
CD2 = CD2_fun(mach)
drag = CD0 * s_ref * dyn_pres + CD1 * lift + CD2 / (s_ref * dyn_pres) * lift**2

# Energy
pe = mu/re - mu/r
ke = 0.5 * v**2
e = pe + ke

# Add States & EOMs
ocp.add_state(h_nd, v * ca.sin(gam) / h_scale)
ocp.add_state(tha, v * ca.cos(gam) / r)
ocp.add_state(v_nd, (-drag / mass - g * ca.sin(gam)) / v_scale)
ocp.add_state(gam, lift / (mass * v) + ca.cos(gam) * (v / r - g / v))

# Cost
ocp.set_cost(0, 0, -tha)

# Boundary Values
# e_0 = ca.MX.sym('e_0', 1)
h_0 = ca.MX.sym('h_0', 1)
tha_0 = ca.MX.sym('tha_0', 1)
v_0 = ca.MX.sym('v_0', 1)
gam_0 = ca.MX.sym('gam_0', 1)

h_0_val = 260_000
v_0_val = 25_600
e_0_val = mu/re - mu/(re + h_0_val) + 0.5 * v_0_val**2
gam_0_val = -1 / 180 * np.pi

# ocp.add_constant(e_0, e_0_val)
ocp.add_constant(h_0, h_0_val)
ocp.add_constant(tha_0, 0.)
ocp.add_constant(v_0, v_0_val)
ocp.add_constant(gam_0, gam_0_val)

e_f = ca.MX.sym('e_f', 1)
e_f_val = mu/re - mu/(re + 80e3) + 0.5 * 2_500.**2
ocp.add_constant(e_f, e_f_val)
h_f = ca.MX.sym('h_f', 1)
v_f = ca.MX.sym('v_f', 1)
gam_f = ca.MX.sym('gam_f', 1)

ocp.add_constant(h_f, 0.)
ocp.add_constant(v_f, 10.)
ocp.add_constant(gam_f, 0.)

# Initial state constrained
ocp.add_constraint('initial', t)
ocp.add_constraint('initial', h - h_0)
ocp.add_constraint('initial', tha - tha_0)
ocp.add_constraint('initial', v - v_0)
ocp.add_constraint('initial', gam - gam_0)

# Terminal state free (except for energy)
# ocp.add_constraint('terminal', e - e_f)
ocp.add_constraint('terminal', h - h_f)
ocp.add_constraint('terminal', v - v_f)
ocp.add_constraint('terminal', gam - gam_f)

# # Add altitude constraint
# h_max_val = atm.h_layers[-1]
# h_min_val = 0.
#
# h_max = ca.MX.sym('h_max')
# h_min = ca.MX.sym('h_min')
# eps_h = ca.MX.sym('eps_h')
#
# ocp.add_constant(h_max, h_max_val)
# ocp.add_constant(h_min, h_min_val)
# ocp.add_constant(eps_h, 1e-4)
# ocp.add_inequality_constraint(
#     'path', h_nd, h_min/h_scale, h_max/h_scale,
#     regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
#         eps_h, method='utm'
#     )
# )

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

    # Find energy where h = 0
    h_f = 0.
    e_h_f = binary_search(
        _x_min=1., _x_max=6e6,
        _f=lambda _e: get_glide_slope(e_vals=np.array((_e,)), h_min=-2e3)['h'],
        _f_val_target=h_f
    )

    # Maximum energy is at maximum Mach (for minimum altitude)
    mach_max = 3.0
    e_max = 0.5 * (3.0 * atm.speed_of_sound(0.))**2

    e_vals = np.logspace(np.log10(e_h_f), np.log10(e_max), 100)
    glide_dict = get_glide_slope(e_vals=e_vals)
    h_guess = glide_dict['h'][-1]
    v_guess = glide_dict['v'][-1]
    gam_guess = glide_dict['gam'][-1]
    u_interp = sp.interpolate.PchipInterpolator(glide_dict['v'], glide_dict['u'])

    def ctrl_law(_t, _x, _p, _k):
        return np.array((u_interp(_x[2] * v_scale_val),))

    # Convert E, h, gam costates to V, h, gam costates
    # d(E)/dt = g d(h)/dt + V d(V)/dt
    # -> lam_v = V lam_E
    # -> lam_h = lam_h + g lam_E
    lam_v_glide = glide_dict['lam_E'][-1] * glide_dict['v'][-1]
    lam_h_glide = glide_dict['lam_h'][-1] + glide_dict['g'][-1] * glide_dict['lam_E'][-1]
    lam_gam_glide = glide_dict['lam_gam'][-1]
    lam_tha_guess = -1.  # With terminal cost formulation, H = - d(tha)/dt + ... -> lam_tha = -1

    guess = giuseppe.guess_generation.auto_propagate_guess(
        adiff_dual, control=ctrl_law, t_span=np.linspace(0., 25., 5),
        initial_states=np.array((h_guess / h_scale_val, 0., v_guess / v_scale_val, gam_guess)), fit_states=False,
        immutable_constants=('h_scale', 'v_scale'),
        initial_costates=np.array((lam_h_glide * h_scale_val, lam_tha_guess, lam_v_glide * v_scale_val, lam_gam_glide)),
        fit_adjoints=False
    )

    with open('guess_range.data', 'wb') as file:
        pickle.dump(guess, file)

    seed_sol = num_solver.solve(guess)

    with open('seed_sol_range.data', 'wb') as file:
        pickle.dump(seed_sol, file)

    idx_ef = adiff_dual.annotations.constants.index('e_f')
    idx_hf = adiff_dual.annotations.constants.index('h_f')
    idx_vf = adiff_dual.annotations.constants.index('v_f')
    idx_gamf = adiff_dual.annotations.constants.index('gam_f')
    idx_h0 = adiff_dual.annotations.constants.index('h_0')
    idx_v0 = adiff_dual.annotations.constants.index('v_0')
    idx_gam0 = adiff_dual.annotations.constants.index('gam_0')

    e0_0 = e_vals[-1]
    ef_0 = mu/re - mu/(re + seed_sol.x[0, -1]*h_scale_val) + 0.5 * (seed_sol.x[2, -1] * v_scale_val)**2
    ef_f = e_vals[0]
    h_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['h'])
    v_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['v'])
    gam_interp = sp.interpolate.PchipInterpolator(glide_dict['E'], glide_dict['gam'])


    def glide_slope_continuation_xf(previous_sol, frac_complete):
        _ef = (ef_f/ef_0)**frac_complete * ef_0  # Log step
        _gamf = gam_interp(_ef)
        _hf = h_interp(_ef)
        _vf = v_interp(_ef)
        previous_sol.k[idx_ef] = _ef
        previous_sol.k[idx_hf] = _hf
        previous_sol.k[idx_vf] = _vf
        previous_sol.k[idx_gamf] = _gamf
        return previous_sol.k

    # def glide_slope_continuation(previous_sol, frac_complete):
    #     e_val = e0_0 + frac_complete * (e0_f - e0_0)
    #     _v0 = v0_0 + frac_complete * (v0_1 - v0_0)
    #     _h0 = h_v_interp(_v0)
    #     _gam0 = gam_v_interp(_v0)
    #     previous_sol.k[idx_h0] = _h0
    #     previous_sol.k[idx_v0] = _v0
    #     previous_sol.k[idx_gam0] = _gam0
    #     return previous_sol.k

    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    # cont.add_logarithmic_series(25, {'eps_h': 1e-8})
    # cont.add_logarithmic_series(100, {'e_f': e_vals[0]})
    # cont.add_logarithmic_series(25, {'eps_h': 1e-6})
    cont.add_custom_series(100, glide_slope_continuation_xf, 'Glide Slope (tf)')
    sol_set = cont.run_continuation()
    sol_set.save('sol_set_range.data')

    hf = sol_set.solutions[-1].x[0, -1] * h_scale_val
    vf = sol_set.solutions[-1].x[2, -1] * v_scale_val
    ef = mu / re - mu / (re + hf) + 0.5 * vf ** 2
    h0 = sol_set.solutions[-1].x[0, 0] * h_scale_val
    v0 = sol_set.solutions[-1].x[2, 0] * v_scale_val
    e0 = mu / re - mu / (re + h0) + 0.5 * v0 ** 2

    # Sweep Solution Space (perturb initial h/v with gam0 = 0, e0 const.)
    if SWEEP_SOLUTION_SPACE:
        # qdyn_max = 500.


        def generate_energy_sweep_continuation(_h0_0, _h0_f):
            def energy_sweep_continuation(previous_sol, frac_complete):
                _constants = previous_sol.k.copy()

                _h0 = previous_sol.k[idx_h0]
                _v0 = previous_sol.k[idx_v0]
                _e0 = mu/re - mu/(re + _h0) + 0.5 * _v0**2

                _h0_next = _h0_0 + frac_complete * (_h0_f - _h0_0)
                _v0_next = (2 * (_e0 - mu/re + mu/(re + _h0_next))) ** 0.5
                _constants[idx_h0] = _h0_next
                _constants[idx_v0] = _v0_next

                return _constants
            return energy_sweep_continuation

        # def find_h_qdyn(_e, _qdyn, _h_max=atm.h_layers[-1]):
        #     _h_min = 0.
        #     _h = 0.5 * (_h_min + _h_max)
        #
        #     for idx in range(1000):
        #         _v = (2 * (_e - mu/re + mu/(re + _h))) ** 0.5
        #         _qdyn_trial = 0.5 * atm.density(_h) * _v ** 2
        #         if _qdyn_trial < _qdyn:
        #             # Altitude too high
        #             _h_max = _h
        #         else:
        #             _h_min = _h
        #         _h = 0.5 * (_h_min + _h_max)
        #         if _h_max - _h < 1e-3:
        #             break
        #
        #     return _h

        # v0_min = binary_search(
        #     _x_min=10., _x_max=v0,
        #     _f=lambda _v: _v / atm.speed_of_sound(-re - mu/(e0 - mu/re - 0.5 * _v**2)),
        #     _f_val_target=0.5
        # )
        # h0_max = min(atm.h_layers[-1], -re - mu/(e0 - mu/re - 0.5 * v0_min**2))
        h0_max = 145e3
        h0_min = 100.  # Compatible with e0 chosen from Mach max.

        # Get Low altitude solution
        cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
        cont.add_custom_series(
            100, generate_energy_sweep_continuation(h0, h0_min), 'Const. energy', keep_bisections=False
        )
        sol_set_sweep1 = cont.run_continuation()

        # Choose solution that does not violate g-load bound.
        idx_sweep1 = 0
        for idx, sol in enumerate(sol_set_sweep1.solutions):
            _load = sol.u[0, :]
            _load_max = np.max(abs(_load))
            if _load_max > load_max:
                idx_sweep1 = idx - 1
                break

        cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
        cont.add_custom_series(
            100, generate_energy_sweep_continuation(h0, h0_max), 'Const. energy', keep_bisections=False
        )
        sol_set_sweep2 = cont.run_continuation()

        sol_set_sweep = deepcopy(sol_set_sweep2)
        sol_set_sweep.solutions = deepcopy([
            sol_set_sweep1.solutions[idx_sweep1], sol_set_sweep1.solutions[0], sol_set_sweep2.solutions[-1]
        ])

        sol_set_sweep.save('sol_set_range_sweep.data')
