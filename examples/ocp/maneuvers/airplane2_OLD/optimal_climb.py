import numpy as np
import casadi as ca
import pickle

import giuseppe

from lookup_tables import thrust_table, cl_alpha_table, cd0_table, temp_table, dens_table, atm

ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.MX)

# Independent Variable
t = ca.MX.sym('t', 1)
ocp.set_independent(t)

# Controls
alpha = ca.MX.sym('alpha', 1)
# sigma = ca.MX.sym('sigma', 1)
ocp.add_control(alpha)
# ocp.add_control(sigma)

# States
h = ca.MX.sym('h', 1)
v = ca.MX.sym('v', 1)
gam = ca.MX.sym('gam', 1)
# psi = ca.MX.sym('psi', 1)
m = ca.MX.sym('m', 1)

# Immutable Constant Parameters
Isp = 2800.0
s_ref = 500.
eta = 1.0

mu = 1.4076539e16
Re = 20902900
g0 = mu / Re ** 2
g = mu / (Re + h) ** 2
# g = g0

# Look-Up Tables & Atmospheric Expressions
T = temp_table(h)
rho = dens_table(h)

a = ca.sqrt(atm.specific_heat_ratio * atm.gas_constant * T)
M = v / a

thrust = thrust_table(ca.vertcat(M, h))
CLalpha = cl_alpha_table(M)
cd0 = cd0_table(M)

# Aerodynamics
qdyn = 0.5 * rho * v ** 2
cl = CLalpha * alpha
cd = cd0 + eta * CLalpha * alpha ** 2
lift = qdyn * s_ref * cl
drag = qdyn * s_ref * cd

# Dynamics
ocp.add_state(h, v * ca.sin(gam))
ocp.add_state(v, (thrust * ca.cos(alpha) - drag) / m - g * ca.sin(gam))
ocp.add_state(gam, (thrust * ca.sin(alpha) + lift) / (m * v) - g / v * ca.cos(gam))
ocp.add_state(m, -thrust / (Isp * g0))

# Reference Scales
t_ref = ca.MX.sym('t_ref')
v_ref = ca.MX.sym('v_ref')
gam_ref = ca.MX.sym('gam_ref')
h_ref = ca.MX.sym('h_ref')
m_ref = ca.MX.sym('m_ref')

t_ref_val = 100.
v_ref_val = (3 * atm.speed_of_sound(65_600.) - 0.38 * atm.speed_of_sound(0.)) / 2
gam_ref_val = 30 * np.pi / 180
h_ref_val = 65_600. / 2
m_ref_val = (34_200. / g0) / 2

ocp.add_constant(t_ref, t_ref_val)
ocp.add_constant(v_ref, v_ref_val)
ocp.add_constant(gam_ref, gam_ref_val)
ocp.add_constant(h_ref, h_ref_val)
ocp.add_constant(m_ref, m_ref_val)

# Boundary Conditions
h0 = ca.MX.sym('h0')
v0 = ca.MX.sym('v0')
gam0 = ca.MX.sym('gam0')
m0 = ca.MX.sym('m0')

ocp.add_constant(h0, 0.)
ocp.add_constant(v0, 0.38 * atm.speed_of_sound(0.))
ocp.add_constant(gam0, 0.)
# ocp.add_constant(m0, 34_200. / g0)
ocp.add_constant(m0, 20_000. / g0)

ocp.add_constraint(location='initial', expr=t / t_ref)
ocp.add_constraint(location='initial', expr=(h - h0) / h_ref)
ocp.add_constraint(location='initial', expr=(v - v0) / v_ref)
ocp.add_constraint(location='initial', expr=(gam - gam0) / gam_ref)
ocp.add_constraint(location='initial', expr=(m - m0) / m_ref)

hf = ca.MX.sym('hf')
vf = ca.MX.sym('vf')
gamf = ca.MX.sym('gamf')

ocp.add_constant(hf, 65_600.)
ocp.add_constant(vf, 3 * atm.speed_of_sound(65_600.))
ocp.add_constant(gamf, 0.)

ocp.add_constraint(location='terminal', expr=(h - hf) / h_ref)
ocp.add_constraint(location='terminal', expr=(v - vf) / v_ref)
ocp.add_constraint(location='terminal', expr=(gam - gamf) / gam_ref)

# Cost
weight_time = ca.MX.sym('Wt')  # 1.0 -> min time, 0.0 -> min fuel
ocp.add_constant(weight_time, 1.0)
ocp.set_cost(0, 0, weight_time * t / t_ref + (1 - weight_time) * m / m_ref)

# # Path Constraint (h > 0)
# h_min = ca.MX.sym('h_min')
# h_max = ca.MX.sym('h_max')
# eps_h = ca.MX.sym('eps_h')
#
# ocp.add_constant(h_min, -2.e3)
# ocp.add_constant(h_max, 105e3)
# ocp.add_constant(eps_h, 1e-3)
#
# ocp.add_inequality_constraint(
#         'path', h,
#         lower_limit=h_min, upper_limit=h_max,
#         regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
#                 regulator=eps_h / h_max))
#
# mach_min = ca.MX.sym('mach_min')
# mach_max = ca.MX.sym('mach_max')
# eps_mach = ca.MX.sym('eps_mach')
#
# ocp.add_constant(mach_min, 0.)
# ocp.add_constant(mach_max, 3.2)
# ocp.add_constant(eps_mach, 1e-3)
#
# ocp.add_inequality_constraint(
#     'path', M,
#     lower_limit=mach_min, upper_limit=mach_max,
#     regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
#         regulator=eps_mach / mach_max
#     )
# )

# Compilation
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(adiff_dual, verbose=2, max_nodes=100, node_buffer=10,
                                                      bc_tol=1e-8)

guess = giuseppe.guess_generation.auto_propagate_guess(adiff_dual, control=9 * np.pi/180, t_span=0.1)

with open('../guess_climb.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('../seed_sol_climb.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Continuations (from guess BCs to desired BCs)
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(100, {'vf': 0.5 * atm.speed_of_sound(0.), 'hf': 0., 'gamf': 0.})
cont.add_linear_series(100, {'m0': 30_000. / g0, 'hf': 65_600., 'vf': 3. * atm.speed_of_sound(65_600.)})
cont.add_linear_series(25, {'m0': 34_200. / g0})
sol_set = cont.run_continuation()

# Save Solution
sol_set.save('sol_set_climb.data')
