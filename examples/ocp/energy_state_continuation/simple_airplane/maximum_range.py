import pickle
from copy import deepcopy

import math
import numpy as np
from scipy import optimize, interpolate
import matplotlib
import casadi as ca

import giuseppe

matplotlib.use('TkAgg', force=True)

ocp = giuseppe.problems.symbolic.StrInputProb()

# -------------------------------------------------------------------------------------------------------------------- #
# PROBLEM SETUP                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
# Independent Variables
ocp.set_independent('t')

# Constants
# (from Airplane 1 at subsonic speeds:
# https://doi/org/10.2514/6.1968-877)
weight = 42_000  # lb
g = 32.2  # gravitational acceleration [ft/s2]
mass = weight / g
ocp.add_constant('W', weight)
ocp.add_constant('g', g)
ocp.add_expression('m', 'W/g')  # Mass [slug]

# Aerodynamic constants
Sref = 530.
CLa = 3.44
CD0 = 0.013
eta = 0.54
CL_max_ld = (CLa*CD0/eta)**0.5
ocp.add_constant('Sref', Sref)  # ft2
ocp.add_constant('CLa', CLa)  # 1/rad
ocp.add_constant('CD0', CD0)  # [-]
ocp.add_constant('eta', eta)  # [-]

# State and control variables
ocp.add_expression('V2', '2*g*(hE-h)')
ocp.add_expression('V', 'V2**0.5')
ocp.add_state('hE', '-V*D/W')
ocp.add_state('h', 'V*sin(gam)')
ocp.add_state('gam', 'L/(m*V) - g/V*cos(gam)')

ocp.add_control('CL')
ocp.add_expression('Qdyn', '0.5*rho*V2')
ocp.add_expression('L', 'Qdyn*Sref*CL')
ocp.add_expression('CD', 'CD0 + eta/CLa * CL*CL')
ocp.add_expression('D', 'Qdyn*Sref*CD')

# Atmosphere function
rho0 = 0.002378
h_ref = 23_800
ocp.add_expression('rho', 'rho0 * exp(-h/h_ref)')
ocp.add_constant('rho0', rho0)  # slug/ft3
ocp.add_constant('h_ref', h_ref)  # ft

# Cost
ocp.set_cost('0', '-V * cos(gam)', '0')

# Boundary conditions (initial)
ocp.add_constraint('initial', 't')
hE0 = 80_000.
ocp.add_constant('hE0', hE0)
ocp.add_constraint('initial', 'hE - hE0')
h0 = 70_000.
ocp.add_constant('h0', h0)
ocp.add_constraint('initial', 'h - h0')
gam0 = 0.
ocp.add_constant('gam0', gam0)
ocp.add_constraint('initial', 'gam - gam0')

# (terminal)
hEf = 20_000.
ocp.add_constant('hEf', hEf)
ocp.add_constraint('terminal', 'hE - hEf')

# -------------------------------------------------------------------------------------------------------------------- #
# NUMERICAL SOLUTION TO FULL FIDELITY OCP                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
with giuseppe.utils.Timer('Setup Time: '):
    comp_dual = giuseppe.problems.SymDual(ocp, control_method='differential').compile(use_jit_compile=True)
    solver = giuseppe.numeric_solvers.SciPySolver(comp_dual, verbose=2)
    guess = giuseppe.guess_generation.auto_propagate_guess(comp_dual, control=(CL_max_ld,), t_span=10)

with giuseppe.utils.Timer('Solve Time: '):
    cont = giuseppe.continuation.ContinuationHandler(solver, guess)
    cont.add_linear_series(100, {'hEf': hEf})
    sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
sol = sol_set[-1]


# -------------------------------------------------------------------------------------------------------------------- #
# NUMERICAL SOLUTION TO OUTER SOLUTION                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
def outer_indirect_control(_v, _hE):
    # Control input
    _h = _hE - _v*_v/(2*g)  # Algebraic conversion from (E, V) -> h
    _gam = 0.  # Enforces d(h)/dt = 0
    _x = np.array((_hE, _h, _gam))

    _Qdyn = 0.5 * rho0 * math.exp(-_h/h_ref) * _v*_v
    _u = weight / (_Qdyn * Sref)  # enforces d(gam)/dt = 0

    # Outer costates
    _path_cost = -_v
    _cd = CD0 + eta/CLa * _u*_u
    _cdu = 2 * eta/CLa * _u
    _drag = _Qdyn * Sref * _cd
    _dhE_dt = -_v * _drag / weight
    _lam_hE = -_path_cost / _dhE_dt
    _lam_V = 0.  # enforces H_gam = 0.
    _lam_gam = _v * _cdu * (_v/g * _lam_hE + _lam_V)  # enforces Hu = 0
    _lam = np.array((_lam_hE, _lam_V, _lam_gam))

    return _x, _u, _lam


def outer_residual(_v, _hE):
    _x, _u, _lam = outer_indirect_control(_v, _hE)
    _h_x = comp_dual.compute_costate_dynamics(0., _x, _lam, np.array((_u,)), sol.p, sol.k)
    return _h_x[1]  # enforces H_h = 0


sol_outer = deepcopy(sol)
v_values = np.empty_like(sol_outer.x[1, :])

# At first index, use "true" solution as initial guess
idx = 0
v0 = 990.
v1 = 1000.
hE = sol_outer.x[0, idx]
sol_root = optimize.root_scalar(lambda _v: outer_residual(_v, hE), x0=v0, x1=v1)
v_values[idx] = sol_root.root
sol_outer.x[:, idx], sol_outer.u[0, idx], sol_outer.lam[:, idx] = \
    outer_indirect_control(sol_root.root, sol_outer.x[0, idx])
v0 = sol_root.root - 10.
v1 = sol_root.root + 10.

for idx, hE in enumerate(sol_outer.x[0, 1:], start=1):
    # Cycle through energy height values, using full fidelity values as initial guess
    sol_root = optimize.root_scalar(lambda _v: outer_residual(_v, hE), x0=v0, x1=v1)
    v_values[idx] = sol_root.root
    sol_outer.x[:, idx], sol_outer.u[0, idx], sol_outer.lam[:, idx] = \
        outer_indirect_control(sol_root.root, sol_outer.x[0, idx])
    v0 = sol_root.root - 10.
    v1 = sol_root.root + 10.


# Adjust time so that E dissipates optimally
v_interp = interpolate.PchipInterpolator(sol_outer.x[0, ::-1], v_values[::-1])


def outer_energy_dynamics(_hE):
    _v = v_interp(_hE)
    _h = _hE - _v * _v / (2 * g)  # Algebraic conversion from (E, V) -> h
    _Qdyn = 0.5 * rho0 * math.exp(-_h/h_ref) * _v*_v
    _u = weight / (_Qdyn * Sref)  # enforces d(gam)/dt = 0
    _cd = CD0 + eta/CLa * _u*_u
    _drag = _Qdyn * Sref * _cd
    return -_v*_drag / weight


def outer_time_residual(_dt, _hE0, _hE1):
    _f0 = outer_energy_dynamics(_hE0)
    _f1 = outer_energy_dynamics(_hE1)
    _hE_middle = 0.5 * (_hE0 + _hE1) - _dt/8 * (_f1 - _f0)
    _f_middle = outer_energy_dynamics(_hE_middle)
    return _hE1 - _hE0 - _dt/6 * (_f0 + 4*_f_middle + _f1)


dt_vector = np.diff(sol_outer.t)
dt0 = 0.
dt1 = np.diff(sol_outer.t[:2])
for idx, (hEi0, hEi1) in enumerate(zip(sol_outer.x[0, :-1], sol_outer.x[0, 1:])):
    sol_root = optimize.root_scalar(lambda _dt: outer_time_residual(_dt, hEi0, hEi1), x0=dt0, x1=dt1)
    dt_vector[idx] = sol_root.root

sol_outer.t = np.concatenate(((0.,), np.cumsum(dt_vector)))

with open('sol_outer.data', 'wb') as f:
    pickle.dump(sol_outer, f)

# -------------------------------------------------------------------------------------------------------------------- #
# CONTINUATION SOLUTION FROM OUTER TO FULL FIDELITY                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
hE_sym = ca.SX.sym('hE')
h_sym = ca.SX.sym('h')
gam_sym = ca.SX.sym('gam')
CL_sym = ca.SX.sym('CL')
CD_sym = CD0 + eta/CLa * CL_sym**2

v2_sym = 2*g*(hE_sym - h_sym)
v_sym = ca.sqrt(v2_sym)

rho_sym = rho0 * ca.exp(-h_sym/h_ref)
qdyn_sym = 0.5 * rho_sym * v2_sym
wing_load_sym = qdyn_sym * Sref
lift_sym = wing_load_sym * CL_sym
drag_sym = wing_load_sym * CD_sym

x_sym = ca.vcat((hE_sym, h_sym, gam_sym))
tf_sym = ca.SX.sym('tf')
f_sym = ca.vcat((
    -v_sym * drag_sym / weight,
    v_sym * ca.sin(gam_sym),
    lift_sym / (mass * v_sym) - g/v_sym * ca.cos(gam_sym)
))
f_fun_ca = ca.Function('f', (x_sym, CL_sym), (f_sym,))
path_cost_sym = v_sym * ca.cos(gam_sym)

lam_sym = ca.vcat([ca.SX.sym('lam_' + _x_sym.name()) for _x_sym in ca.vertsplit(x_sym)])
ham_sym = path_cost_sym + ca.dot(f_sym, lam_sym)
f_lam_sym = -ca.jacobian(ham_sym, x_sym).T
f_lam_fun_ca = ca.Function('flam', (x_sym, lam_sym, CL_sym), (f_lam_sym,))

# Discretized states / costates
tau_mesh = sol_outer.t / sol_outer.t[-1]
dtau_mesh = np.diff(tau_mesh)
x_mesh = ca.SX.sym('X', x_sym.shape[0], sol.t.shape[0])
u_mesh = ca.SX.sym('U', (1, sol.t.shape[0]))
f_mesh = f_fun_ca(x_mesh, u_mesh)
lam_mesh = ca.SX.sym('Lam', x_mesh.shape)
f_lam_mesh = f_lam_fun_ca(x_mesh, lam_mesh, u_mesh)
f_u_mesh = ca.SX.zeros(u_mesh.shape)  # TODO
y_mesh = ca.vcat((x_mesh, lam_mesh, u_mesh))
fp_mesh = ca.vcat((f_mesh, f_lam_mesh, f_u_mesh)) * tf_sym

# Continuation parameter
s_sym = ca.SX.sym('s')  # 0 -> outer solution, 1 -> full dynamics

# Initial boundary conditions
bc0 = ca.vcat((
    x_mesh[0, 0] - hE0,
    s_sym * (x_mesh[1, 0] - h0) + (1 - s_sym) * f_sym[1],
    s_sym * (x_mesh[2, 0] - gam0) + (1 - s_sym) * f_sym[2]
))

# Terminal boundary conditions
bcf = ca.vcat((
    x_mesh[0, -1] - hEf,
    s_sym * lam_sym[1] + (1 - s_sym) * f_lam_sym[1],
    s_sym * lam_sym[2] + (1 - s_sym) * f_lam_sym[2]
))

# continuation condition
bcs = s_sym - 1

# Dynamic residual
tau_middle = tau_mesh[:-1] + 0.5*dtau_mesh
y_middle = 0.5 * (y_mesh[:, :-1] + y_mesh[:, 1:]) - dtau_mesh/8 * (fp_mesh[:, 1:] - fp_mesh[:, :-1])  # TODO - fix broadcast of dtau_mesh
# col_res = TODO
