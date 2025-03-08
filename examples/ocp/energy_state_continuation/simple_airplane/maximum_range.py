import pickle
from copy import deepcopy

import math
import numpy as np
from scipy import optimize, interpolate
from scipy.sparse.linalg import splu
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
path_cost_sym = -v_sym * ca.cos(gam_sym)  # -dx/dt -> maximum range

lam_sym = ca.vcat([ca.SX.sym('lam_' + _x_sym.name()) for _x_sym in ca.vertsplit(x_sym)])
ham_sym = path_cost_sym + ca.dot(f_sym, lam_sym)
hu_sym = ca.jacobian(ham_sym, CL_sym)
f_lam_sym = -ca.jacobian(ham_sym, x_sym).T
f_lam_fun_ca = ca.Function('flam', (x_sym, lam_sym, CL_sym), (f_lam_sym,))

# Index reduction to obtain control dynamics
huu_sym = ca.jacobian(hu_sym, CL_sym)
hux_sym = ca.jacobian(hu_sym, x_sym)
fu_sym = ca.jacobian(f_sym, CL_sym)
f_u_sym = -(hux_sym@f_sym + ca.dot(fu_sym, f_lam_sym))/huu_sym

y_sym = ca.vcat((x_sym, lam_sym, CL_sym))
fy_sym = ca.vcat((f_sym, f_lam_sym, f_u_sym))
fy_fun_ca = ca.Function('fy', (y_sym,), (fy_sym,))
hu_fun_ca = ca.Function('Hu', (y_sym,), (hu_sym,))
h_fun_ca = ca.Function('H', (y_sym,), (ham_sym,))

# Discretized states / costates
tau_mesh = sol_outer.t / sol_outer.t[-1]
dtau_mesh = np.diff(tau_mesh)
x_mesh = ca.SX.sym('X', x_sym.shape[0], sol.t.shape[0])
lam_mesh = ca.SX.sym('Lam', x_mesh.shape)
u_mesh = ca.SX.sym('U', (1, sol.t.shape[0]))
y_mesh = ca.vcat((x_mesh, lam_mesh, u_mesh))
fy_mesh = fy_fun_ca(y_mesh)
f_x_mesh = fy_mesh[:x_sym.shape[0], :]
f_lam_mesh = fy_mesh[x_sym.shape[0]:2*x_sym.shape[0], :]
fyp_mesh = fy_mesh * tf_sym
hu_mesh = hu_fun_ca(y_mesh)
h_mesh = h_fun_ca(y_mesh)

# Continuation parameter
s_sym = ca.SX.sym('s')  # 0 -> outer solution, 1 -> full dynamics

# Initial boundary conditions
bc0 = ca.vcat((
    x_mesh[0, 0] - hE0,
    s_sym * (x_mesh[1, 0] - h0) + (1 - s_sym) * f_x_mesh[1, 0]*tf_sym,
    s_sym * (x_mesh[2, 0] - gam0) + (1 - s_sym) * f_x_mesh[2, 0]*tf_sym,
    hu_mesh[:, 0]  # Control initial BC
))

# Terminal boundary conditions
bcf = ca.vcat((
    x_mesh[0, -1] - hEf,
    s_sym * lam_mesh[1, -1] + (1 - s_sym) * f_lam_mesh[1, -1]*tf_sym,
    s_sym * lam_mesh[2, -1] + (1 - s_sym) * f_lam_mesh[2, -1]*tf_sym,
    h_mesh[:, -1],  # Time terminal BC
))

# continuation condition
bcs = s_sym - 1

# Dynamic residual
dtau_tile = np.tile(dtau_mesh[None, :], y_sym.shape)
# tau_middle = tau_mesh[:-1] + 0.5*dtau_mesh
y_middle = \
    0.5 * (y_mesh[:, :-1] + y_mesh[:, 1:])\
    - dtau_tile/8 * (fyp_mesh[:, 1:] - fyp_mesh[:, :-1])
fyp_middle = tf_sym * fy_fun_ca(y_middle)
col_res = y_mesh[:, 1:] - y_mesh[:, :-1] - dtau_tile/6 * (fyp_mesh[:, :-1] + 4*fyp_middle + fyp_mesh[:, 1:])

# Replace U residual with Hu=0
col_res[-1, :] = hu_mesh[:, 1:]

# Continue collocation for fast states from dx/dt=0 to continuity
# (NOTE: for x,   INITIAL BC is enforced,  so collocation continued with f(1), ..., f(N)
#        for lam, TERMINAL BC is enforced, so collocation continued with f(0), ..., f(N-1)
idces_x_fast = (1, 2)
idces_lam_fast = (4, 5)

col_res[idces_x_fast, :] *= s_sym
col_res[idces_x_fast, :] += (1 - s_sym) * fyp_mesh[idces_x_fast, 1:]

col_res[idces_lam_fast, :] *= s_sym
col_res[idces_lam_fast, :] += (1 - s_sym) * fyp_mesh[idces_lam_fast, :-1]

# Global system
z_sym = ca.vcat((
    ca.vec(y_mesh),
    tf_sym,
    s_sym
))
res_sym = ca.vcat((
    bc0,
    ca.vec(col_res),
    bcf,
    bcs
))
jac_sym = ca.jacobian(res_sym, z_sym)

res_fun = ca.Function('r', (z_sym,), (res_sym,))
jac_fun = ca.Function('J', (z_sym,), (jac_sym,))

y_mesh0 = np.vstack((sol_outer.x, sol_outer.lam, sol_outer.u))
fy_mesh0 = fy_fun_ca(y_mesh0)
fyp_mesh0 = fy_mesh0 * sol_outer.t[-1]
z0 = np.concatenate((
    np.ravel(y_mesh0, order='F'),
    sol_outer.t[-1:],
    (0.,)
))[:, None]

# Run simple damped Newton search
acc_min = 0.95
max_iter = 1_000
alpha_min = 1E-5
alpha_max = 0.1

z = z0.copy()
res = res_fun(z).full()
alpha = alpha_max
recompute_jac = True
singular = False
for iteration in range(max_iter):
    if recompute_jac:
        print('Computing Jac...')
        J = jac_fun(z).sparse()

        try:
            LU = splu(J)
        except RuntimeError:
            print('Jac is singular! Stopping search.')
            singular = True
            break

        step = LU.solve(res)
        cost = np.dot(step.T, step)[0, 0]
        # cost = np.abs(res).max(initial=0.)

    accept = False
    print('Running backtracking line search...')
    while alpha > alpha_min:
        z_new = z - alpha * step
        res = res_fun(z_new).full()
        step_new = LU.solve(res)
        cost_new = np.dot(step_new.T, step_new)[0, 0]
        # cost_new = np.abs(res).max(initial=0.)
        if cost_new < (1 + acc_min*alpha*(alpha-2)) * cost:
        # if cost_new < (1 - acc_min * alpha) * cost:
            # Accept step
            print(f'Accept alpha={alpha}')
            accept = True
            alpha *= 8
            if alpha > alpha_max:
                alpha = alpha_max
            break
        else:
            print(f'Reject alpha={alpha}')
            alpha /= 2

    if accept:
        z = z_new
    else:
        print(f'Backtrack failed! Stopping search.')
        break

    if np.all(np.abs(res) < 1E-3):
        print(f'Success! Huzzah!')
        break

    # If the full step was taken, then we are going to continue with
    # the same Jacobian. This is the approach of BVP_SOLVER.
    if alpha == 1:
        step = step_new
        # step = LU.solve(res)
        cost = cost_new
        recompute_jac = False
    else:
        recompute_jac = True
