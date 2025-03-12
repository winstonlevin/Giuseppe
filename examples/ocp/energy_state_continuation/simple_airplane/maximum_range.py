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
# TODO - use sol_outer and implement continuation
sol0 = sol

# Interpolations of (x, lam, u) in order to
f0 = np.vstack([comp_dual.compute_dynamics(
    sol0.t[idx], sol0.x[:, idx], sol0.u[:, idx], sol0.p, sol0.k
) for idx in range(len(sol0.t))]).T
flam0 = np.vstack([comp_dual.compute_costate_dynamics(
    sol0.t[idx], sol0.x[:, idx], sol0.lam[:, idx], sol0.u[:, idx], sol0.p, sol0.k
) for idx in range(len(sol0.t))]).T
fu0 = np.vstack([comp_dual.control_handler.compute_control_dynamics(
    sol0.t[idx], sol0.x[:, idx], sol0.lam[:, idx], sol0.u[:, idx], sol0.p, sol0.k
) for idx in range(len(sol0.t))]).T
x_interp = interpolate.BPoly.from_derivatives(
    sol0.t, np.concatenate((sol0.x[:, None, :], f0[:, None, :]), axis=1).transpose((2, 1, 0))
)
lam_interp = interpolate.BPoly.from_derivatives(
    sol0.t, np.concatenate((sol0.lam[:, None, :], flam0[:, None, :]), axis=1).transpose((2, 1, 0))
)
u_interp = interpolate.BPoly.from_derivatives(
    sol0.t, np.concatenate((sol0.u[:, None, :], fu0[:, None, :]), axis=1).transpose((2, 1, 0))
)

# Hard-coded mesh [for phases]
p_order = 4  # Number of collocation points
t_phase_vals = np.array((0., 50., 100., 150., 200., 600., 650., sol0.t[-1]))
taug_vals = 2*t_phase_vals / t_phase_vals[-1] - 1

# Local mesh info for each polynomial of order ``p_order''
tau_sym = ca.SX.sym('tau')
taul_vals, w_vals = giuseppe.utils.pseudospectral.lg(p_order)
x_sym = ca.SX.sym('x', p_order)  # Symbolic interpolation points
lagrange_polynomials = ca.SX.ones(p_order)
l_idces = np.arange(0, p_order, 1)
for num_idx, tau_num in enumerate(taul_vals):
    idces_poly_i = np.delete(l_idces, num_idx)
    lagrange_polynomials[idces_poly_i] *= (tau_sym - tau_num) / (taul_vals[idces_poly_i] - tau_num)

dl_dtau = ca.jacobian(lagrange_polynomials, tau_sym).T
diff_mat = ca.DM(ca.vcat([
    ca.substitute(dl_dtau, tau_sym, _xi) for _xi in taul_vals[1:]
])).full()
diff_mat_T = diff_mat.T
int_vec = (w_vals[None, :] @ diff_mat)[0, 1:]  # xf = x0 + dot([f1, f2, ..., fN], I)

# Global tau values
phases = []
for idx in range(taug_vals.shape[0] - 1):
    phase_dict = {}
    phase_dict['taug_vals'] = \
        0.5*(taug_vals[idx] + taug_vals[idx+1]) \
        + 0.5*(taug_vals[idx+1] - taug_vals[idx]) * taul_vals
    phase_dict['t_vals'] = t_phase_vals[-1]/2 * (phase_dict['taug_vals'] + 1.)
    phases.append(phase_dict)

# Build Necessary Conditions for Optimality -------------------------------------------------------------------------- #
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
u_sym = CL_sym
tf_sym = ca.SX.sym('tf')
f_sym = ca.vcat((
    -v_sym * drag_sym / weight,
    v_sym * ca.sin(gam_sym),
    lift_sym / (mass * v_sym) - g/v_sym * ca.cos(gam_sym)
))
f_fun_ca = ca.Function('f', (x_sym, u_sym), (f_sym,), ('x', 'u'), ('f',))
path_cost_sym = -v_sym * ca.cos(gam_sym)  # -dx/dt -> maximum range

lam_sym = ca.vcat([ca.SX.sym('lam_' + _x_sym.name()) for _x_sym in ca.vertsplit(x_sym)])
ham_sym = path_cost_sym + ca.dot(f_sym, lam_sym)
h_fun_ca = ca.Function('H', (x_sym, lam_sym, u_sym), (ham_sym,), ('x', 'lam', 'u'), ('H',))
hu_sym = ca.jacobian(ham_sym, u_sym)
hu_fun_ca = ca.Function('Hu', (x_sym, lam_sym, u_sym), (hu_sym,), ('x', 'lam', 'u'), ('Hu',))
f_lam_sym = -ca.jacobian(ham_sym, x_sym).T
f_lam_fun_ca = ca.Function('flam', (x_sym, lam_sym, u_sym), (f_lam_sym,), ('x', 'lam', 'u'), ('flam',))

# --- Problem definition / derivation -------------------------------------------------------------------------------- #
# Find:
#
# (1) X   (N+2, n)
# (2) Lam (N+2, n)
# (3) U   (N+2, m)
# (4) tf  (1)
#
# Such that:
#
# (1) BC0(X0, Lam0, u0)      = 0 [n+m   Conditions for X0, Lam0    ]
# (2) DX - F(X, U)           = 0 [Nxn   Conditions for Lam         ]
# (3) DLam - Flam(X, Lam, U) = 0 [Nxn   Conditions for X           ]
# (4) Hu                     = 0 [Nxm   Conditions for U           ]
# (5) BCF(Xf, Lamf, tf, uf)  = 0 [n+m+1 Conditions for Xf, Lamf, tf]
nx = x_sym.shape[0]
nlam = nx
nu = u_sym.shape[0]

# Unknown values
X_mesh_sym = [ca.SX.sym('X' + str(idx), nx, p_order) for idx in range(taug_vals.shape[0]-1)]
Lam_mesh_sym = [ca.SX.sym('Lam' + str(idx), nlam, p_order) for idx in range(taug_vals.shape[0]-1)]
U_mesh_sym = [ca.SX.sym('U' + str(idx), nu, p_order) for idx in range(taug_vals.shape[0]-1)]
tf_sym = ca.SX.sym('tf')

# Initial guesses
X_mesh0 = [x_interp(_phase_dict['t_vals']).T for _phase_dict in phases]
Lam_mesh0 = [lam_interp(_phase_dict['t_vals']).T for _phase_dict in phases]
U_mesh0 = [u_interp(_phase_dict['t_vals']).T for _phase_dict in phases]
tf0 = t_phase_vals[-1]

DX_mesh_sym = [_X @ diff_mat_T for _X in X_mesh_sym]
Xf_mesh_sym = [_X[:, 0] + _X[:, 1:] @ int_vec[:, None] for _X in X_mesh_sym]
DLam_mesh_sym = [_L @ diff_mat_T for _L in Lam_mesh_sym]
Lamf_mesh_sym = [_L[:, 0] + _L[:, 1:] @ int_vec[:, None] for _L in Lam_mesh_sym]
Uf_mesh_sym = [_U[:, 0] + _U[:, 1:] @ int_vec[:, None] for _U in U_mesh_sym]

# Gains
dtaug_vals = np.diff(taug_vals)
dt_dtaul = tf_sym * dtaug_vals/4

# Initial boundary condition
hE0_sym = ca.SX.sym('hE0')
h0_sym = ca.SX.sym('h0')
gam0_sym = ca.SX.sym('gam0')
x0_sym = ca.vcat((hE0_sym, h0_sym, gam0_sym))
x0 = ca.vcat((hE0, h0, gam0))

# Initial mesh
bc00 = ca.vcat((
    X_mesh_sym[0][:, 0] - x0,
    hu_fun_ca(X_mesh_sym[0][:, 0], Lam_mesh_sym[0][:, 0], U_mesh_sym[0][:, 0]),
))
col_res0 = ca.vcat((
    DX_mesh_sym[0] - dt_dtaul[0] * f_fun_ca(X_mesh_sym[0][:, 1:], U_mesh_sym[0][:, 1:]),
    DLam_mesh_sym[0] - dt_dtaul[0] * f_lam_fun_ca(X_mesh_sym[0][:, 1:], Lam_mesh_sym[0][:, 1:], U_mesh_sym[0][:, 1:]),
    hu_fun_ca(X_mesh_sym[0][:, 1:], Lam_mesh_sym[0][:, 1:], U_mesh_sym[0][:, 1:]),
))
bcf0 = ca.vcat((
    Xf_mesh_sym[0] - X_mesh_sym[1][:, 0],
    Lamf_mesh_sym[0] - Lamf_mesh_sym[1][:, 0],
))

z0 = ca.vec(ca.vcat((X_mesh_sym[0], Lam_mesh_sym[0], U_mesh_sym[0])))
res0 = ca.vcat((
    bc00,
    ca.vec(col_res0),
    bcf0
))

# Intermediate meshes
residuals = [res0]
unknowns = [z0]
guesses = [np.vstack((X_mesh0[0], Lam_mesh0[0], U_mesh0[0])).ravel(order='F')]
for mesh_idx in range(1, len(X_mesh_sym)-1):
    X_i = X_mesh_sym[mesh_idx]
    Lam_i = Lam_mesh_sym[mesh_idx]
    U_i = U_mesh_sym[mesh_idx]

    bc0i = ca.vcat((
        hu_fun_ca(X_i[:, 0], Lam_i[:, 0], U_i[:, 0]),
    ))
    col_resi = ca.vcat((
        DX_mesh_sym[mesh_idx] - dt_dtaul[mesh_idx] * f_fun_ca(X_i[:, 1:], U_i[:, 1:]),
        DLam_mesh_sym[mesh_idx] - dt_dtaul[mesh_idx] * f_lam_fun_ca(X_i[:, 1:], Lam_i[:, 1:], U_i[:, 1:]),
        hu_fun_ca(X_i[:, 1:], Lam_i[:, 1:], U_i[:, 1:]),
    ))
    bcfi = ca.vcat((
        Xf_mesh_sym[mesh_idx] - X_mesh_sym[mesh_idx + 1][:, 0],
        Lamf_mesh_sym[mesh_idx] - Lam_mesh_sym[mesh_idx + 1][:, 0],
    ))

    residuals.append(ca.vcat((
        bc0i,
        ca.vec(col_resi),
        bcfi
    )))
    unknowns.append(ca.vec(ca.vcat((X_i, Lam_i, U_i))))
    guesses.append(np.vstack((X_mesh0[mesh_idx], Lam_mesh0[mesh_idx], U_mesh0[mesh_idx])).ravel(order='F'))

# Terminal mesh residual
X_i = X_mesh_sym[-1]
Lam_i = Lam_mesh_sym[-1]
U_i = U_mesh_sym[-1]

bc0f = ca.vcat((
    hu_fun_ca(X_i[:, 0], Lam_i[:, 0], U_i[:, 0]),
))
col_resf = ca.vcat((
    DX_mesh_sym[-1] - dt_dtaul[-1] * f_fun_ca(X_i[:, 1:], U_i[:, 1:]),
    DLam_mesh_sym[-1] - dt_dtaul[-1] * f_lam_fun_ca(X_i[:, 1:], Lam_i[:, 1:], U_i[:, 1:]),
    hu_fun_ca(X_i[:, 1:], Lam_i[:, 1:], U_i[:, 1:]),
))

bcff = ca.vcat((
    Xf_mesh_sym[-1][0] - hEf,  # Terminal energy fixed
    Lamf_mesh_sym[-1][1:],  # Terminal fast costates = 0 b/c terminal fast states are free
    h_fun_ca(Xf_mesh_sym[-1], 0., Uf_mesh_sym[-1]),
))

residuals.append(ca.vcat((
    bc0f,
    ca.vec(col_resf),
    bcff
)))
unknowns.append(ca.vcat((
    ca.vec(ca.vcat((X_i, Lam_i, U_i))),
    tf_sym,
)))
guesses.append(np.append(
    np.vstack((X_mesh0[-1], Lam_mesh0[-1], U_mesh0[-1])).ravel(order='F'),
    tf0
))

res_sym = ca.vcat(residuals)
z_sym = ca.vcat(unknowns)
jac_sym = ca.jacobian(res_sym, z_sym)

res_fun = ca.Function('r', (z_sym,), (res_sym,), ('z',), ('r',))
jac_fun = ca.Function('J', (z_sym,), (jac_sym,), ('z',), ('J',))

z0 = np.concatenate(guesses)[:, None]

sol_root = optimize.root(fun=lambda _z: res_fun(_z).full()[:, 0], x0=z0[:, 0], jac=jac_fun)

# Run simple damped Newton search ------------------------------------------------------------------------------------ #
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
        print(f'Cost: {cost_new}')
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
