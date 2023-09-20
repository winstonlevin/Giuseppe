import casadi as ca
import numpy as np
import scipy as sp
import pickle

import giuseppe

from space_shuttle_aero_atm import mu, re, g0, mass, s_ref, CD0, CD1, CD2, atm, sped_fun
from glide_slope import get_glide_slope

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

e_f = ca.SX.sym('e_f', 1)
e_f_val = mu/re - mu/(re + 80e3) + 0.5 * 2_500.**2
ocp.add_constant(e_f, e_f_val)

# Initial state constrained
ocp.add_constraint('initial', t)
ocp.add_constraint('initial', h - h_0)
ocp.add_constraint('initial', tha - tha_0)
ocp.add_constraint('initial', v - v_0)
ocp.add_constraint('initial', gam - gam_0)

# Terminal state free (except for energy)
ocp.add_constraint('terminal', e - e_f)
# ocp.add_constraint('terminal', gam - gam_f)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(
        adiff_dual, verbose=2, max_nodes=100, node_buffer=10
    )

if __name__ == '__main__':
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
        adiff_dual, control=u_guess, t_span=30.,
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
    glide_dict_flat_full = get_glide_slope(flat_earth=True)
    mach_interp = sp.interpolate.PchipInterpolator(
        glide_dict_flat_full['v'] / np.asarray(sped_fun(glide_dict_flat_full['h'])).flatten(), glide_dict_flat_full['E']
    )

    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_logarithmic_series(100, {'e_f': mach_interp(1.5)})
    sol_set = cont.run_continuation()
    sol_set.save('sol_set_range.data')

    # Sweep Solution Space

