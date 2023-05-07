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
sigma = ca.MX.sym('sigma', 1)
ocp.add_control(alpha)
ocp.add_control(sigma)

# States
h = ca.MX.sym('h', 1)
v = ca.MX.sym('v', 1)
gam = ca.MX.sym('gam', 1)
psi = ca.MX.sym('psi', 1)
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
qdyn = 0.5 * rho * v**2
cl = CLalpha * alpha
cd = cd0 + eta * CLalpha * alpha ** 2
lift = qdyn * s_ref * cl
drag = qdyn * s_ref * cd

# Dynamics
ocp.add_state(h, v * ca.sin(gam))
ocp.add_state(v, (thrust * ca.cos(alpha) - drag) / m - g * ca.cos(gam))
ocp.add_state(gam, (thrust * ca.sin(alpha) + lift) / (m * v) - g/v * ca.cos(gam))
ocp.add_state(m, -thrust / (Isp * g0))

# Cost
weight_time = ca.MX.sym('Wt')  # 1.0 -> min time, 0.0 -> min fuel
ocp.add_constant(weight_time, 1.0)
ocp.set_cost(0, 0, weight_time * t + (1 - weight_time) * m)

# Boundary Conditions
h0 = ca.MX.sym('h0')
v0 = ca.MX.sym('v0')
gam0 = ca.MX.sym('gam0')
m0 = ca.MX.sym('m0')

ocp.add_constant(h0, 0.)
ocp.add_constant(v0, 0.38 * atm.speed_of_sound(0.))
ocp.add_constant(gam0, 0.)
ocp.add_constant(m0, 34_200.)

hf = ca.MX.sym('hf')
vf = ca.MX.sym('vf')
gamf = ca.MX.sym('gamf')

ocp.add_constant(hf, 65_600.)
ocp.add_constant(vf, 3 * atm.speed_of_sound(0.))
ocp.add_constant(gamf, 0.)

# TODO -- guess generation, continuations
