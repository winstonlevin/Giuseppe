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

t_ref = 100.
v_ref = (3 * atm.speed_of_sound(65_600.) - 0.38 * atm.speed_of_sound(0.)) / 2
gam_ref = 30 * np.pi / 180
h_ref = 65_600. / 2
m_ref = 34_200. / g0 / 2

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

# Compilation
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(adiff_dual, verbose=False, max_nodes=100, node_buffer=10,
                                                      bc_tol=1e-8)

guess = giuseppe.guess_generation.auto_propagate_guess(adiff_dual, control=9 * np.pi/180, t_span=2.5)

with open('guess_climb.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('seed_sol_climb.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Continuations (from guess BCs to desired BCs)
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(100, {'vf': 0.5 * atm.speed_of_sound(0.), 'hf': 0., 'gamf': 0.})
cont.add_linear_series(100, {'m0': 30_000. / g0, 'hf': 65_600., 'vf': 3. * atm.speed_of_sound(65_600.)})
cont.add_linear_series(25, {'m0': 34_200. / g0})
sol_set = cont.run_continuation()

# Save Solution
sol_set.save('sol_set_climb.data')
