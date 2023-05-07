import numpy as np
import casadi as ca
import pickle

import giuseppe

from lookup_tables import thrust_table, cl_alpha_table, cd0_table

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
S = 500.
eta = 1.0

mu = 1.4076539e16
Re = 20902900
g0 = mu / Re ** 2

atm = giuseppe.utils.examples.Atmosphere1976(use_metric=False)
T = temp_table_bspline(h)
rho = dens_table_bspline(h)
a = ca.sqrt(atm.specific_heat_ratio * atm.gas_constant * T)
a_func = ca.Function('a', (h,), (a,), ('h',), ('a',))
M = v / a

# Look-Up Tables
thrust = thrust_table(ca.vertcat(M, h))
CLalpha = cl_alpha_table(M)
CD0 = cd0_table(M)
