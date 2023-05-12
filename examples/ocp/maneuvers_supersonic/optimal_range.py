from copy import deepcopy

import numpy as np
import casadi as ca
import pickle

import giuseppe

from lookup_tables import thrust_table, cl_alpha_table, cd0_table, temp_table, dens_table, atm

d2r = np.pi / 180

ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.MX)

# Independent Variable
t = ca.MX.sym('t', 1)
ocp.set_independent(t)

# Controls
alpha = ca.MX.sym('alpha', 1)
phi = ca.MX.sym('phi', 1)
ocp.add_control(alpha)
ocp.add_control(phi)

# States
h = ca.MX.sym('h', 1)
xn = ca.MX.sym('xn', 1)
xd = ca.MX.sym('xe', 1)
v = ca.MX.sym('v', 1)
gam = ca.MX.sym('gam', 1)
psi = ca.MX.sym('psi', 1)
m = ca.MX.sym('m', 1)

# Immutable Constant Parameters
Isp = ca.MX.sym('Isp')
s_ref = ca.MX.sym('s_ref')
eta = ca.MX.sym('eta')
mu = ca.MX.sym('mu')
Re = ca.MX.sym('Re')

ocp.add_constant(Isp, 2800.0)
ocp.add_constant(s_ref, 500.)
ocp.add_constant(eta, 1.0)
ocp.add_constant(mu, 1.4076539e16)
ocp.add_constant(Re, 20902900)

g0 = 1.4076539e16 / 20902900 ** 2
g = mu / (Re + h) ** 2
# g = g0

# Look-Up Tables & Atmospheric Expressions
T = temp_table(h)
rho = dens_table(h)

a = ca.sqrt(atm.specific_heat_ratio * atm.gas_constant * T)
M = v / a

# thrust = thrust_table(ca.vertcat(M, h))
thrust = 0.
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
ocp.add_state(xn, v * ca.cos(gam) * ca.cos(psi))
ocp.add_state(xd, v * ca.cos(gam) * ca.sin(psi))
ocp.add_state(v, (thrust * ca.cos(alpha) - drag) / m - g * ca.sin(gam))
ocp.add_state(gam, (thrust * ca.sin(alpha) + lift) * ca.cos(phi) / (m * v) - g / v * ca.cos(gam))
ocp.add_state(psi, (thrust * ca.sin(alpha) + lift) * ca.sin(phi) / (m * v * ca.cos(gam)))
ocp.add_state(m, -thrust / (Isp * g0))

# Boundary Conditions
h0 = ca.MX.sym('h0')
xn0 = ca.MX.sym('xn0')
xd0 = ca.MX.sym('xd0')
v0 = ca.MX.sym('v0')
gam0 = ca.MX.sym('gam0')
psi0 = ca.MX.sym('psi0')
m0 = ca.MX.sym('m0')

ocp.add_constant(h0, 65_600.)
ocp.add_constant(xn0, 0.)
ocp.add_constant(xd0, 0.)
ocp.add_constant(v0, 3 * atm.speed_of_sound(65_600.))
ocp.add_constant(gam0, 0.)
ocp.add_constant(psi0, 0.)
# ocp.add_constant(m0, 34_200. / g0)
ocp.add_constant(m0, 32138.594625382884 / g0)

t_ref = ca.MX.sym('t_ref')
v_ref = ca.MX.sym('v_ref')
gam_ref = ca.MX.sym('gam_ref')
psi_ref = ca.MX.sym('psi_ref')
h_ref = ca.MX.sym('h_ref')
m_ref = ca.MX.sym('m_ref')
x_ref = ca.MX.sym('x_ref')

v_ref_val = (3 * atm.speed_of_sound(65_600.) - 0.38 * atm.speed_of_sound(0.)) / 2
t_ref_val = 100.

ocp.add_constant(t_ref, t_ref_val)
ocp.add_constant(v_ref, v_ref_val)
ocp.add_constant(gam_ref, 30 * d2r)
ocp.add_constant(psi_ref, 90 * d2r)
ocp.add_constant(h_ref, 65_600. / 2)
ocp.add_constant(m_ref, 34_200. / g0 / 2)
ocp.add_constant(x_ref, v_ref_val * t_ref_val)

ocp.add_constraint(location='initial', expr=t / t_ref)
ocp.add_constraint(location='initial', expr=(h - h0) / h_ref)
ocp.add_constraint(location='initial', expr=(xn - xn0) / x_ref)
ocp.add_constraint(location='initial', expr=(xd - xd0) / x_ref)
ocp.add_constraint(location='initial', expr=(v - v0) / v_ref)
ocp.add_constraint(location='initial', expr=(gam - gam0) / gam_ref)
ocp.add_constraint(location='initial', expr=(psi - psi0) / psi_ref)
ocp.add_constraint(location='initial', expr=(m - m0) / m_ref)

hf = ca.MX.sym('hf')
vf = ca.MX.sym('vf')
gamf = ca.MX.sym('gamf')

ocp.add_constant(hf, 0.)
ocp.add_constant(vf, 0.38 * atm.speed_of_sound(0.))
ocp.add_constant(gamf, 0.)

ocp.add_constraint(location='terminal', expr=(h - hf) / h_ref)
ocp.add_constraint(location='terminal', expr=(v - vf) / v_ref)
ocp.add_constraint(location='terminal', expr=(gam - gamf) / gam_ref)

# Altitude Constraint
eps_h = ca.MX.sym('eps_h')
h_min = ca.MX.sym('h_min')
h_max = ca.MX.sym('h_max')
ocp.add_constant(eps_h, 1e-1)
ocp.add_constant(h_min, 0.)
ocp.add_constant(h_max, 100e3)
ocp.add_inequality_constraint(
    'path', h, h_min, h_max,
    regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
        eps_h/h_ref, 'utm'
    )
)

# Cost
terminal_angle = ca.MX.sym('terminal_angle')  # 0 deg -> max downrange, 90 deg -> max crossrange
ocp.add_constant(terminal_angle, 0.0)
ocp.set_cost(0, 0, -(xn * ca.cos(terminal_angle) - xd * ca.sin(terminal_angle)) / x_ref)

# Compilation
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(adiff_dual, verbose=False, max_nodes=100, node_buffer=10)


guess = giuseppe.guess_generation.auto_propagate_guess(
    adiff_dual,
    control=np.asarray(adiff_dual.ca_control2pseudo(np.array((3., 0.)) * d2r, adiff_dual.default_values)).flatten(),
    t_span=30)

with open('guess_range.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('seed_sol_range.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Continuations (from guess BCs to desired BCs)
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(100, {'vf': 0.38 * atm.speed_of_sound(0.), 'hf': 60., 'gamf': 0. * d2r})
cont.add_logarithmic_series(100, {'eps_h': 1e-6})
sol_set = cont.run_continuation()

# Save Solution
sol_set.save('sol_set_range.data')

# Sweep Altitudes
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
cont.add_linear_series(100, {'h0': 40_000., 'v0': 3 * atm.speed_of_sound(40_000.)})
sol_set_altitude = cont.run_continuation()
sol_set_altitude.save('sol_set_range_altitude.data')

# Sweep Velocities
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set_altitude.solutions[-1]))
cont.add_linear_series(100, {'v0': 0.5 * atm.speed_of_sound(40_000.)})
sol_set_altitude = cont.run_continuation()
sol_set_altitude.save('sol_set_range_velocity.data')

# Sweep Cross-Range
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
cont.add_linear_series(179, {'terminal_angle': 180. * d2r})
sol_set_crossrange = cont.run_continuation()
sol_set_crossrange.save('sol_set_range_crossrange.data')
