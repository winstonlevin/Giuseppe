from copy import deepcopy
import casadi as ca
import numpy as np
import scipy as sp
import pickle

import giuseppe

from space_shuttle_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, atm, sped_fun
from glide_slope import get_glide_slope

SWEEP_SOLUTION_SPACE = False

ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb()

# Independent Variables
t = ca.SX.sym('t', 1)
ocp.set_independent(t)

# Constants
weight0 = g0 * mass

# State Variables
h_nd = ca.SX.sym('h_nd', 1)
h_scale = ca.SX.sym('h_scale', 1)
tha = ca.SX.sym('tha', 1)
v_nd = ca.SX.sym('v_nd', 1)
v_scale = ca.SX.sym('v_scale', 1)
gam = ca.SX.sym('gam', 1)

h = h_nd * h_scale
v = v_nd * v_scale

h_scale_val = 1e4
v_scale_val = 1e4
ocp.add_constant(h_scale, h_scale_val)
ocp.add_constant(v_scale, v_scale_val)

# Atmosphere Func
_, __, rho = atm.get_ca_atm_expr(h)

# Add Controls
lift_nd = ca.SX.sym('lift_nd', 1)
lift = lift_nd * weight0

ocp.add_control(lift_nd)

# Expressions
r = re + h
g = mu / r**2
dyn_pres = 0.5 * rho * v ** 2
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
e_0 = ca.SX.sym('e_0', 1)
h_0 = ca.SX.sym('h_0', 1)
tha_0 = ca.SX.sym('tha_0', 1)
v_0 = ca.SX.sym('v_0', 1)
gam_0 = ca.SX.sym('gam_0', 1)

h_0_val = 260_000
v_0_val = 25_600
e_0_val = mu/re - mu/(re + h_0_val) + 0.5 * v_0_val**2
gam_0_val = -1 / 180 * np.pi

ocp.add_constant(e_0, e_0_val)
ocp.add_constant(h_0, h_0_val)
ocp.add_constant(tha_0, 0.)
ocp.add_constant(v_0, v_0_val)
ocp.add_constant(gam_0, gam_0_val)

# e_f = ca.SX.sym('e_f', 1)
# e_f_val = mu/re - mu/(re + 80e3) + 0.5 * 2_500.**2
# ocp.add_constant(e_f, e_f_val)
h_f = ca.SX.sym('h_f', 1)
v_f = ca.SX.sym('v_f', 1)
gam_f = ca.SX.sym('gam_f', 1)

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
# ocp.add_constraint('terminal', gam - gam_f)

# # Add altitude constraint
# h_max_val = atm.h_layers[-1]
# h_min_val = 0.
#
# h_max = ca.SX.sym('h_max')
# h_min = ca.SX.sym('h_min')
# eps_h = ca.SX.sym('eps_h')
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
        adiff_dual, verbose=2, max_nodes=100, node_buffer=10
    )

if __name__ == '__main__':
    e_guess = np.linspace(1e7, 9e6, 10)
    de = e_guess[1] - e_guess[0]
    glide_dict = get_glide_slope(e_vals=e_guess)
    h_guess = glide_dict['h']
    v_guess = glide_dict['v']
    drag_guess = glide_dict['drag']
    gam_guess = glide_dict['gam']
    u_guess = glide_dict['u']
    r_guess = glide_dict['r']

    # Determine costates from energy state (convert E, h, gam to V, h, gam)
    # d(E)/dt = g d(h)/dt + V d(V)/dt
    # -> lam_v = V lam_E
    # -> lam_h = lam_h + g lam_E

    lam_h_guess = glide_dict['lam_h'] + glide_dict['g'] * glide_dict['lam_E']
    lam_tha_guess = -1. * np.ones(e_guess.shape)  # With terminal cost formulation, H = - d(tha)/dt + ... -> lam_tha = -1
    lam_v_guess = glide_dict['lam_E'] * glide_dict['v']
    lam_gam_guess = glide_dict['lam_gam']

    # Determine downrange and time from dynamics
    dt_de_guess = -mass / (v_guess * drag_guess)
    dt_guess = dt_de_guess * de
    dtha_de_guess = mass * np.cos(gam_guess) / (-r_guess * drag_guess)
    dtha_guess = dtha_de_guess * de
    t_guess = np.empty(e_guess.shape)
    tha_guess = np.empty(e_guess.shape)
    t_guess[0] = 0.
    tha_guess[0] = 0.
    for idx in range(len(e_guess[:-1])):
        t_guess[idx+1] = t_guess[idx] + dt_guess[idx]
        tha_guess[idx+1] = tha_guess[idx] + dtha_guess[idx]

    # Build guess
    x_guess = np.vstack((h_guess/h_scale_val, tha_guess, v_guess/v_scale_val, gam_guess))
    lam_guess = np.vstack((lam_h_guess*h_scale_val, lam_tha_guess, lam_v_guess*v_scale_val, lam_gam_guess))
    u_guess = u_guess.reshape((1, -1))
    guess = giuseppe.guess_generation.initialize_guess(
        prob=adiff_dual, t_span=t_guess, x=x_guess, u=u_guess, lam=lam_guess
    )

    e_0_guess = 1.e7
    glide_dict = get_glide_slope(e_vals=np.array((0.9*e_0_guess, e_0_guess, 1.1*e_0_guess)))
    h_guess = glide_dict['h'][1]
    v_guess = glide_dict['v'][1]
    gam_guess = glide_dict['gam'][1]
    u_guess = glide_dict['u'][1]

    # Convert E, h, gam costates to V, h, gam costates
    # d(E)/dt = g d(h)/dt + V d(V)/dt
    # -> lam_v = V lam_E
    # -> lam_h = lam_h + g lam_E
    lam_v_glide = glide_dict['lam_E'][1] * glide_dict['v'][1]
    lam_h_glide = glide_dict['lam_h'][1] + glide_dict['g'][1] * glide_dict['lam_E'][1]
    lam_gam_glide = glide_dict['lam_gam'][1]
    lam_tha_guess = -1.  # With terminal cost formulation, H = - d(tha)/dt + ... -> lam_tha = -1

    guess = giuseppe.guess_generation.auto_propagate_guess(
        adiff_dual, control=u_guess, t_span=20.,
        initial_states=np.array((h_guess / h_scale_val, 0., v_guess / v_scale_val, gam_guess)), fit_states=False,
        immutable_constants=('h_scale', 'v_scale')
    )

    with open('guess_range.data', 'wb') as file:
        pickle.dump(guess, file)

    seed_sol = num_solver.solve(guess)

    with open('seed_sol_range.data', 'wb') as file:
        pickle.dump(seed_sol, file)

    # Continue until the glide-slope Mach number is 1.5
    # (Flat earth is used since Mach monotonically increases)
    # glide_dict_flat_full = get_glide_slope(flat_earth=True)
    # mach_interp = sp.interpolate.PchipInterpolator(
    #     glide_dict_flat_full['v'] / np.asarray(sped_fun(glide_dict_flat_full['h'])).flatten(), glide_dict_flat_full['E']
    # )
    # glide_dict_full = get_glide_slope()

    # idx_ef = adiff_dual.annotations.constants.index('e_f')
    # idx_h_scale = adiff_dual.annotations.constants.index('h_scale')
    #
    #
    # def find_hf_series(previous_sol, frac_complete):
    #     _constants = previous_sol.k.copy()
    #
    #     _hf_last = previous_sol.x[0, -1] * previous_sol.k[idx_h_scale]
    #     _ef_last = previous_sol.k[idx_ef]
    #     # hf < 0 -> increase ef. hf > 0 -> decrease ef.
    #     _constants[idx_ef] = _ef_last - _hf_last * g0
    #     return _constants

    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    # cont.add_logarithmic_series(25, {'eps_h': 1e-6})
    # cont.add_logarithmic_series(100, {'e_f': 100e3})
    # cont.add_custom_series(100, find_hf_series, 'Find hf')
    # cont.add_linear_series(100, {'h_0': 250e3, 'v_0': 1e3, 'gam_0': 0.})
    sol_set = cont.run_continuation()
    sol_set.save('sol_set_range.data')

    # Sweep Solution Space
    if SWEEP_SOLUTION_SPACE:
        # Cover:
        # h in {25, 50, ..., 250} k ft
        # V in {1, 2, ..., 20} k ft/s
        h_vals = np.arange(25., 250. + 12.5, 12.5) * 1e3
        v_vals = np.arange(1., 20. + 1., 1.) * 1e3
        n_h_vals = len(h_vals)

        def trim_mesh(_sol0, _n_nodes_max):
            _sol0 = deepcopy(_sol0)
            _dt_min = np.sort(np.diff(_sol0.t))[len(_sol0.t) - _n_nodes_max]

            # Cycle through indices, removing nodes when the step size is lower than _dt_min
            _idx = 1
            while _idx < len(_sol0.t) - 2:
                if _sol0.t[_idx] - _sol0.t[_idx - 1] < _dt_min:
                    # Step is too small: remove
                    _sol0.t = np.delete(_sol0.t, _idx, 0)
                    _sol0.h_u = np.delete(_sol0.h_u, _idx, 1)
                    _sol0.eig_h_uu = np.delete(_sol0.eig_h_uu, _idx, 1)
                    _sol0.lam = np.delete(_sol0.lam, _idx, 1)
                    _sol0.u = np.delete(_sol0.u, _idx, 1)
                    _sol0.x = np.delete(_sol0.x, _idx, 1)
                else:
                    # Step is sufficiently large: keep, move to next idx.
                    _idx += 1

            return _sol0

        def sweep_altitudes(_sol0):
            # Decrease altitude to h_min
            _cont = giuseppe.continuation.ContinuationHandler(num_solver, _sol0)
            _cont.add_linear_series(n_h_vals - 1, {'h_0': h_vals[0]}, keep_bisections=False)
            _sol_set_sweep_h = _cont.run_continuation()
            return _sol_set_sweep_h

        # Get solution set for first v_val
        sol0 = sol_set.solutions[-1]
        sol_set_sweep = sweep_altitudes(sol0)

        # Generate solution set for subsequent v_vals
        for v_val in v_vals[1::]:
            cont = giuseppe.continuation.ContinuationHandler(num_solver, sol0)
            cont.add_linear_series(25, {'v_0': v_val})
            sol0 = cont.run_continuation().solutions[-1]
            _sol_set_v_val = sweep_altitudes(sol0)
            sol_set_sweep.solutions.extend(_sol_set_v_val.solutions)

        sol_set_sweep.save('sol_set_range_sweep_envelope.data')

        # glide_dict_full = get_glide_slope()
        # h_v_interp = sp.interpolate.PchipInterpolator(glide_dict_full['v'], glide_dict_full['h'])
        # gam_v_interp = sp.interpolate.PchipInterpolator(glide_dict_full['v'], glide_dict_full['gam'])
        #
        # sol0 = sol_set.solutions[-1]
        # v0_0 = v_scale_val * sol0.x[2, 0]
        # v0_1 = v_vals[-1]
        #
        # idx_h0 = adiff_dual.annotations.constants.index('h_0')
        # idx_v0 = adiff_dual.annotations.constants.index('v_0')
        # idx_gam0 = adiff_dual.annotations.constants.index('gam_0')
        #
        # def glide_slope_continuation(previous_sol, frac_complete):
        #     _v0 = v0_0 + frac_complete * (v0_1 - v0_0)
        #     _h0 = h_v_interp(_v0)
        #     _gam0 = gam_v_interp(_v0)
        #     previous_sol.k[idx_h0] = _h0
        #     previous_sol.k[idx_v0] = _v0
        #     previous_sol.k[idx_gam0] = _gam0
        #     return previous_sol.k

        # cont = giuseppe.continuation.ContinuationHandler(num_solver, sol_set.solutions[-1])
        # cont.add_custom_series(200, get_next_constants=glide_slope_continuation, series_name='Glide Slope')
        # cont.add_linear_series(25, {'h_0': h_vals[-1], 'gam_0': 0.})
        # sol_set_glide_slope = cont.run_continuation()

        sol_set_sweep = giuseppe.SolutionSet([sol_set.solutions[-1]], annotations=adiff_dual.annotations)

