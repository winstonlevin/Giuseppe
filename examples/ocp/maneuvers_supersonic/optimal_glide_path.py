import numpy as np
import casadi as ca
import pickle

import giuseppe

from lookup_tables import cl_alpha_table, cd0_table, sped_table, dens_table

d2r = np.pi / 180

# Load Constrained Optimal Glide Path ----------------------------------------------------------------------------------
with open('sol_set_range.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]

# Generate dictionaries
k_dict = {}
x_dict = {}
lam_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val

for key, x_val, lam_val in zip(sol.annotations.states, list(sol.x), list(sol.lam)):
    x_dict[key] = x_val
    lam_dict[key] = lam_val

for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val

# Set Up OCP -----------------------------------------------------------------------------------------------------------
ocp = giuseppe.problems.automatic_differentiation.ADiffInputProb(dtype=ca.MX)

# Independent Variable
t = ca.MX.sym('t', 1)
ocp.set_independent(t)

# Controls
alpha = ca.MX.sym('alpha', 1)
ocp.add_control(alpha)

# States
h = ca.MX.sym('h', 1)
xn = ca.MX.sym('xn', 1)
v = ca.MX.sym('v', 1)
gam = ca.MX.sym('gam', 1)

# Constant Parameters
Isp = ca.MX.sym('Isp')
s_ref = ca.MX.sym('s_ref')
eta = ca.MX.sym('eta')
mu = ca.MX.sym('mu')
Re = ca.MX.sym('Re')
m = ca.MX.sym('m')

ocp.add_constant(Isp, k_dict['Isp'])
ocp.add_constant(s_ref, k_dict['s_ref'])
ocp.add_constant(eta, k_dict['eta'])
ocp.add_constant(mu, k_dict['mu'])
ocp.add_constant(Re, k_dict['Re'])
ocp.add_constant(m, k_dict['m0'])

g0 = k_dict['mu'] / k_dict['Re'] ** 2
g = mu / (Re + h) ** 2
e = g * h + 0.5 * v ** 2

# Look-Up Tables & Atmospheric Expressions
a = sped_table(h)
rho = dens_table(h)
M = v / a

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
ocp.add_state(xn, v * ca.cos(gam))
ocp.add_state(v, (thrust * ca.cos(alpha) - drag) / m - g * ca.sin(gam))
ocp.add_state(gam, (thrust * ca.sin(alpha) + lift) / (m * v) - g / v * ca.cos(gam))

# Reference Values
t_ref = ca.MX.sym('t_ref')
h_ref = ca.MX.sym('h_ref')
x_ref = ca.MX.sym('x_ref')

t_ref_val = k_dict['t_ref']
h_ref_val = k_dict['h_ref']
x_ref_val = k_dict['x_ref']

ocp.add_constant(t_ref, t_ref_val)
ocp.add_constant(h_ref, h_ref_val)
ocp.add_constant(x_ref, x_ref_val)

# Cost
ocp.set_cost(0, 0, -xn / x_ref)

# Boundary Conditions
h0 = ca.MX.sym('h0')
hf = ca.MX.sym('hf')

h0_val = x_dict['h'][0]
hf_val = x_dict['h'][-1]

ocp.add_constant(h0, h0_val)
ocp.add_constant(hf, hf_val)

ocp.add_constraint(location='initial', expr=t / t_ref)
ocp.add_constraint(location='initial', expr=(h - h0) / h_ref)
ocp.add_constraint(location='initial', expr=xn / x_ref)

ocp.add_constraint(location='terminal', expr=(h - hf) / h_ref)

# Altitude Constraint
eps_h = ca.MX.sym('eps_h')
h_min = ca.MX.sym('h_min')
h_max = ca.MX.sym('h_max')
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
    num_solver = giuseppe.numeric_solvers.SciPySolver(adiff_dual, verbose=2, max_nodes=100, node_buffer=10)


# Form guess
guess = giuseppe.guess_generation.initialize_guess(adiff_dual)

t_start = 550
t_end = 650
t_idces = np.where(np.logical_and(sol.t > t_start, sol.t < t_end))

guess.t = sol.t[t_idces]
guess.x = np.vstack((x_dict['h'][t_idces], x_dict['xn'][t_idces], x_dict['v'][t_idces], x_dict['gam'][t_idces]))
guess.p = sol.p
guess.k = adiff_dual.default_values
guess.u = np.array(((u_dict['alpha'][t_idces]),))
guess.lam = np.vstack((lam_dict['h'][t_idces], lam_dict['xn'][t_idces], lam_dict['v'][t_idces], lam_dict['gam'][t_idces]))
guess.nu0 = sol.nu0[((0, 1, 2),)]
guess.nuf = sol.nuf[((0,),)]

# Continuations (from guess BCs to desired BCs)
glide_path = num_solver.solve(guess)

# Save Solution
with open('glide_path.data', 'wb') as file:
    pickle.dump(glide_path, file)
