from copy import deepcopy

import numpy as np
import casadi as ca
import pickle

import giuseppe

from lookup_tables import cl_alpha_fun, cd0_fun, atm, mu as mu_val, Re as Re_val, lut_data
from glide_slope import get_glide_slope

d2r = np.pi / 180


ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.SX)

# Independent Variable
t = ca.SX.sym('t', 1)
ocp.set_independent(t)

# Controls
alpha = ca.SX.sym('alpha', 1)
phi = ca.SX.sym('phi', 1)
ocp.add_control(alpha)
ocp.add_control(phi)

# States
h = ca.SX.sym('h', 1)
xn = ca.SX.sym('xn', 1)
xe = ca.SX.sym('xe', 1)
v = ca.SX.sym('v', 1)
gam = ca.SX.sym('gam', 1)
psi = ca.SX.sym('psi', 1)
m = ca.SX.sym('m', 1)

# Immutable Constant Parameters
Isp = ca.SX.sym('Isp')
s_ref = ca.SX.sym('s_ref')
eta = ca.SX.sym('eta')
mu = ca.SX.sym('mu')
Re = ca.SX.sym('Re')

eta_val = 1.0
s_ref_val = 500.

ocp.add_constant(Isp, 2800.0)
ocp.add_constant(s_ref, s_ref_val)
ocp.add_constant(eta, eta_val)
ocp.add_constant(mu, mu_val)
ocp.add_constant(Re, Re_val)

g0 = mu_val / Re_val ** 2
g = mu / (Re + h) ** 2


def eh2v(_e, _h):
    _g = mu_val / (Re_val + _h) ** 2
    _v = (2 * (_e - _h * _g)) ** 0.5
    return _v


def vh2e(_v, _h):
    _g = mu_val / (Re_val + _h) ** 2
    _e = _g * _h + 0.5 * _v ** 2
    return _e


# Look-Up Tables & Atmospheric Expressions
a = atm.get_ca_speed_of_sound_expr(h)
_, __, rho = atm.get_ca_atm_expr(h)

M = v / a

# thrust = thrust_table(ca.vertcat(M, h))
thrust = 0.
CLalpha = cl_alpha_fun(M)
cd0 = cd0_fun(M)

# Aerodynamics
qdyn = 0.5 * rho * v ** 2
cl = CLalpha * alpha
cd = cd0 + eta * CLalpha * alpha ** 2
lift = qdyn * s_ref * cl
drag = qdyn * s_ref * cd

# Dynamics
ocp.add_state(h, v * ca.sin(gam))
ocp.add_state(xn, v * ca.cos(gam) * ca.cos(psi))
ocp.add_state(xe, v * ca.cos(gam) * ca.sin(psi))
ocp.add_state(v, (thrust * ca.cos(alpha) - drag) / m - g * ca.sin(gam))
ocp.add_state(gam, (thrust * ca.sin(alpha) + lift) * ca.cos(phi) / (m * v) - g / v * ca.cos(gam))
ocp.add_state(psi, (thrust * ca.sin(alpha) + lift) * ca.sin(phi) / (m * v * ca.cos(gam)))
ocp.add_state(m, -thrust / (Isp * g0))

# Reference Values
t_ref = ca.SX.sym('t_ref')
v_ref = ca.SX.sym('v_ref')
gam_ref = ca.SX.sym('gam_ref')
psi_ref = ca.SX.sym('psi_ref')
h_ref = ca.SX.sym('h_ref')
m_ref = ca.SX.sym('m_ref')
x_ref = ca.SX.sym('x_ref')

t_ref_val = 100.
v_ref_val = (3 * atm.speed_of_sound(65_600.) - 0.38 * atm.speed_of_sound(0.)) / 2
gam_ref_val = 30 * d2r
psi_ref_val = 90 * d2r
h_ref_val = 65_600. / 2.
x_ref_val = v_ref_val * t_ref_val
m_ref_val = (34_200. / g0) / 2.

ocp.add_constant(t_ref, t_ref_val)
ocp.add_constant(h_ref, h_ref_val)
ocp.add_constant(x_ref, x_ref_val)
ocp.add_constant(v_ref, v_ref_val)
ocp.add_constant(gam_ref, gam_ref_val)
ocp.add_constant(psi_ref, psi_ref_val)
ocp.add_constant(m_ref, m_ref_val)

# Cost
terminal_angle = ca.SX.sym('terminal_angle')  # 0 deg -> max downrange, 90 deg -> max crossrange
ocp.add_constant(terminal_angle, 0.0)
ocp.set_cost(0, 0, -(xn * ca.cos(terminal_angle) - xe * ca.sin(terminal_angle)) / x_ref)

# Boundary Conditions
h0 = ca.SX.sym('h0')
xn0 = ca.SX.sym('xn0')
xe0 = ca.SX.sym('xd0')
v0 = ca.SX.sym('v0')
gam0 = ca.SX.sym('gam0')
psi0 = ca.SX.sym('psi0')
m0 = ca.SX.sym('m0')

# h0_val = 65_600.
# m0_val = 34_200. / g0
m0_val = 32138.594625382884 / g0

# Base initial conditions off glide slope
h_interp, v_interp, gam_interp, _ = get_glide_slope(mu_val, Re_val, m0_val, s_ref_val, eta_val,
                                                    _h_min=0., _h_max=np.max(lut_data['h']) - 1e3,
                                                    _mach_min=np.min(lut_data['M']) + 0.25,
                                                    _mach_max=np.max(lut_data['M']) - 0.25)

e0_val = vh2e(2.5 * atm.speed_of_sound(65_600.), 65_600.)
v0_val = v_interp(e0_val)
h0_val = h_interp(e0_val)
gam0_val = gam_interp(e0_val)


def ctrl_law(_t, _x, _p, _k):
    # Unpack state
    _h = _x[0]
    _v = _x[3]
    _gam = _x[4]
    _m = _x[6]

    _g = mu_val / (Re_val + _h) ** 2
    _e = _g * _h + 0.5 * _v ** 2

    _qdyn = 0.5 * atm.density(_h) * _v ** 2
    _mach = _v / atm.speed_of_sound(_h)
    _cla = float(cl_alpha_fun(_mach))
    _gam_glide = float(gam_interp(_e))
    _tau = 0.1

    # Control Law: dgam = (gam_glide - gam) / tau
    _alp = (_g / _v * np.cos(_gam) + (_gam_glide - _gam) / _tau) * _m * _v / (_qdyn * s_ref_val * _cla)
    _phi = 0.0
    return np.array((_alp, _phi))


ef_val = vh2e(0.6 * atm.speed_of_sound(0.), 0.)
hf_val = h_interp(ef_val)
vf_val = v_interp(ef_val)
gamf_val = gam_interp(ef_val)

ocp.add_constant(h0, h0_val)
ocp.add_constant(xn0, 0.)
ocp.add_constant(xe0, 0.)
ocp.add_constant(v0, v0_val)
ocp.add_constant(gam0, gam0_val)
ocp.add_constant(psi0, 0.)
ocp.add_constant(m0, m0_val)

ocp.add_constraint(location='initial', expr=t / t_ref)
ocp.add_constraint(location='initial', expr=(h - h0) / h_ref)
ocp.add_constraint(location='initial', expr=(xn - xn0) / x_ref)
ocp.add_constraint(location='initial', expr=(xe - xe0) / x_ref)
ocp.add_constraint(location='initial', expr=(v - v0) / v_ref)
ocp.add_constraint(location='initial', expr=(gam - gam0) / gam_ref)
ocp.add_constraint(location='initial', expr=(psi - psi0) / psi_ref)
ocp.add_constraint(location='initial', expr=(m - m0) / m_ref)

hf = ca.SX.sym('hf')
vf = ca.SX.sym('vf')
gamf = ca.SX.sym('gamf')

ocp.add_constant(hf, 0.)
ocp.add_constant(vf, vf_val)
ocp.add_constant(gamf, 0.)

ocp.add_constraint(location='terminal', expr=(h - hf) / h_ref)
ocp.add_constraint(location='terminal', expr=(v - vf) / v_ref)
ocp.add_constraint(location='terminal', expr=(gam - gamf) / gam_ref)
# ocp.add_constraint(location='terminal', expr=(ca.atan2(xe, xn)) / psi_ref)

# Altitude Constraint
eps_h = ca.SX.sym('eps_h')
h_min = ca.SX.sym('h_min')
h_max = ca.SX.sym('h_max')
# ocp.add_constant(eps_h, 1e-1)
# ocp.add_constant(h_min, 0.)
ocp.add_constant(eps_h, 1e-7)
ocp.add_constant(h_min, -1.5e3)
ocp.add_constant(h_max, 100e3)
ocp.add_inequality_constraint(
    'path', h, h_min, h_max,
    regularizer=giuseppe.problems.automatic_differentiation.regularization.ADiffPenaltyConstraintHandler(
        eps_h/(h_max - h_min), 'utm'
    )
)

# Compilation
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    adiff_dual = giuseppe.problems.automatic_differentiation.ADiffDual(ocp)
    num_solver = giuseppe.numeric_solvers.SciPySolver(adiff_dual, verbose=False, max_nodes=100, node_buffer=10)


guess = giuseppe.guess_generation.auto_propagate_guess(
    adiff_dual,
    control=ctrl_law(_t=0, _x=(h0_val, 0., 0., v0_val, gam0_val, 0., m0_val), _p=None, _k=adiff_dual.default_values),
    t_span=30)

with open('../guess_range.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('../seed_sol_range.data', 'wb') as file:
    pickle.dump(seed_sol, file)


# Continuations (from guess BCs to desired BCs)
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(15, {'hf': hf_val, 'vf': vf_val, 'gamf': gamf_val})
sol_set = cont.run_continuation()


sol_set.solutions[-1] = num_solver.solve(sol_set.solutions[-1], bc_tol=1e-15)

# Save Solution
sol_set.save('sol_set_range.data')

# Sweep Altitudes
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
cont.add_linear_series(25, {'h0': 40_000., 'v0': 3 * atm.speed_of_sound(40_000.)})
sol_set_altitude = cont.run_continuation()
sol_set_altitude.save('sol_set_range_altitude.data')

# Sweep Velocities
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
cont.add_linear_series(15, {'v0': 0.5 * atm.speed_of_sound(40_000.)})
sol_set_altitude = cont.run_continuation()
sol_set_altitude.save('sol_set_range_velocity.data')

# Sweep Cross-Range
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
cont.add_linear_series(179, {'terminal_angle': 180. * d2r})
sol_set_crossrange = cont.run_continuation()
sol_set_crossrange.save('sol_set_range_crossrange.data')

# Sweep Flight Envelope
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(sol_set.solutions[-1]))
cont.add_linear_series(5, {'h0': 75e3, 'v0': 2.5e3, 'gam0': 0.})
sol_set_gam0 = cont.run_continuation()

# Cover a grid spaced h0 in {5, 10, ..., 75} kft, V0 in {1.0, ..., 2.5} kft/s
last_sol = deepcopy(sol_set_gam0.solutions[-1])
cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(last_sol))
cont.add_linear_series(14, {'h0': 5e3})
sol_set_sweep_envelope = cont.run_continuation()

for velocity in (2.0e3, 1.5e3, 1.0e3):
    cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(last_sol))
    cont.add_linear_series(5, {'v0': velocity})
    last_sol = cont.run_continuation().solutions[-1]

    cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(last_sol))
    cont.add_linear_series(14, {'h0': 5e3})
    new_sol_set = cont.run_continuation()
    sol_set_sweep_envelope.solutions.extend(new_sol_set.solutions)

sol_set_sweep_envelope.save('sol_set_range_sweep_envelope.data')