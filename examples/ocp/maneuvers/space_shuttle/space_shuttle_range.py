import casadi as ca
import numpy as np

import giuseppe

from space_shuttle_aero_atm import mu as mu_val, re as re_val, g0, mass as mass_val, s_ref as s_ref_val,\
    CD0, CD1, CD2, atm
from glide_slope import get_glide_slope

FIND_GLIDE_SLOPE = True

ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb()

# Independent Variables
t = ca.SX.sym('t', 1)
ocp.set_independent(t)

# Constants
re = ca.SX.sym('re')
mu = ca.SX.sym('mu')
ocp.add_constant(re, re_val)
ocp.add_constant(mu, mu_val)

m = ca.SX.sym('m', 1)
weight0 = ca.SX.sym('weight0')
s_ref = ca.SX.sym('s_ref', 1)
xi = ca.SX.sym('xi', 1)

ocp.add_constant(m, mass_val)
ocp.add_constant(weight0, g0 * mass_val)
ocp.add_constant(s_ref, s_ref_val)
ocp.add_constant(xi, 0)

# State Variables
h = ca.SX.sym('h', 1)
phi = ca.SX.sym('phi', 1)
tha = ca.SX.sym('tha', 1)
v = ca.SX.sym('v', 1)
gam = ca.SX.sym('gam', 1)
psi = ca.SX.sym('psi', 1)

# Atmosphere Func
_, __, rho = atm.get_ca_atm_expr(h)

# Add Controls
lift_nd = ca.SX.sym('lift_nd', 1)
bank = ca.SX.sym('bank', 1)
lift = lift_nd * weight0

ocp.add_control(lift_nd)
ocp.add_control(bank)

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
ocp.add_state(h, v * ca.sin(gam))
ocp.add_state(phi, v * ca.cos(gam) * ca.sin(psi) / (r * ca.cos(tha)))
ocp.add_state(tha, v * ca.cos(gam) * ca.cos(psi) / r)
ocp.add_state(v, -drag / m - g * ca.sin(gam))
ocp.add_state(gam, lift * ca.cos(bank) / (m * v) + ca.cos(gam) * (v / r - g / v))
ocp.add_state(psi,
              lift * ca.sin(bank)/(m * v * ca.cos(gam)) + v * ca.cos(gam) * ca.sin(psi) * ca.sin(tha)/(r * ca.cos(tha)))

# Cost
ocp.set_cost(0, 0, -phi * ca.cos(xi) - tha * ca.sin(xi))

# Boundary Values
e_0 = ca.SX.sym('e_0', 1)
h_0 = ca.SX.sym('h_0', 1)
phi_0 = ca.SX.sym('phi_0', 1)
tha_0 = ca.SX.sym('tha_0', 1)
v_0 = ca.SX.sym('v_0', 1)
gam_0 = ca.SX.sym('gam_0', 1)
psi_0 = ca.SX.sym('psi_0', 1)

h_0_val = 260_000
v_0_val = 25_600
e_0_val = mu_val/re_val - mu_val/(re_val + h_0_val) + 0.5 * v_0_val**2
gam_0_val = -1 / 180 * np.pi
psi_0_val = np.pi / 2.

ocp.add_constant(e_0, e_0_val)
ocp.add_constant(h_0, h_0_val)
ocp.add_constant(phi_0, 0.)
ocp.add_constant(tha_0, 0.)
ocp.add_constant(v_0, v_0_val)
ocp.add_constant(gam_0, gam_0_val)
ocp.add_constant(psi_0, psi_0_val)

e_f = ca.SX.sym('e_f', 1)
e_f_val = mu_val/re_val - mu_val/(re_val + 80e3) + 0.5 * 2_500.**2
ocp.add_constant(e_f, e_f_val)

if FIND_GLIDE_SLOPE:  # Only initial energy is constrained -> initial h, V, gam free
    ocp.add_constraint('initial', t)
    ocp.add_constraint('initial', e - e_0)
    ocp.add_constraint('initial', phi - phi_0)
    ocp.add_constraint('initial', tha - tha_0)
    ocp.add_constraint('initial', psi - psi_0)
else:  # Constrain initial h, V, gam
    ocp.add_constraint('initial', t)
    ocp.add_constraint('initial', h - h_0)
    ocp.add_constraint('initial', phi - phi_0)
    ocp.add_constraint('initial', tha - tha_0)
    ocp.add_constraint('initial', v - v_0)
    ocp.add_constraint('initial', gam - gam_0)
    ocp.add_constraint('initial', psi - psi_0)

ocp.add_constraint('terminal', e - e_f)

# ocp.add_constraint('terminal', h - h_f)
# ocp.add_constraint('terminal', v - v_f)
# ocp.add_constraint('terminal', gam - gam_f)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(
        adiff_dual, verbose=False, max_nodes=100, node_buffer=10
    )

if __name__ == '__main__':
    e_0_guess = e_0_val
    glide_dict = get_glide_slope(e_vals=np.array((e_0_guess,)), correct_gam=False)  # TODO - fix, returns empty dict
    h_guess = glide_dict['h'][0]
    v_guess = glide_dict['v'][0]
    gam_guess = glide_dict['gam'][0]
    u_guess = glide_dict['u'][0]
    guess = giuseppe.guess_generation.auto_propagate_guess(
        adiff_dual, control=u_guess, t_span=10.,
        initial_states=np.array((h_guess, 0., 0., v_guess, gam_guess, psi_0_val)))
    seed_sol = num_solver.solve(guess)

    cont1 = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)  # TODO - fix series.
    cont1.add_linear_series(100, {'h_f': 200_000, 'v_f': 20_000})
    cont1.add_linear_series(50, {'h_f': 80_000, 'v_f': 2_500, 'gam_f': -5 * np.pi / 180})
    cont1.add_linear_series(90, {'xi': 0.5 * np.pi}, bisection=True)
    sol_set1 = cont1.run_continuation()

    sol_set1.save('sol_set_range.data')
