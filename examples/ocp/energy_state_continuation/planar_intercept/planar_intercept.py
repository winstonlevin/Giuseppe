import os
from copy import copy
import pickle

import numpy as np
import casadi as ca

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

# -------------------------------------------------------------------------------------------------------------------- #
# ANALYTICAL VALUES                                                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
# Boundary conditions
x0 = np.zeros(shape=(3,), dtype=float)
x0[2] = np.pi/2
xf = np.zeros(shape=(3,), dtype=float)
xf[0] = 10.
xf[2] = -np.pi/2

# Outer solution
dp = xf[:2] - x0[:2]
p_hat = dp / np.linalg.norm(dp)

psi_outer = np.arctan2(dp[1], dp[0])
u_outer = np.array((0.,))
lam_outer = np.array((-p_hat[0], -p_hat[1], 0.))
tf_outer = np.dot(dp, dp)**0.5

# -------------------------------------------------------------------------------------------------------------------- #
# GIUSEPPE SOLUTION                                                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
intercept = giuseppe.problems.input.StrInputProb()

intercept.set_independent('t')

intercept.add_state('x', 'cos(psi)')
intercept.add_state('y', 'sin(psi)')
intercept.add_state('psi', 'u')

intercept.add_control('u')


intercept.add_constant('x_0', x0[0])
intercept.add_constant('y_0', x0[1])
intercept.add_constant('psi_0', psi_outer)


intercept.add_constant('x_f', xf[0])
intercept.add_constant('y_f', xf[1])
intercept.add_constant('psi_f', psi_outer)

k = 1.
intercept.add_constant('k', k)
intercept.set_cost('0', '1 + k/2*(u*u)', '0')

intercept.add_constraint('initial', 't')
intercept.add_constraint('initial', 'x - x_0')
intercept.add_constraint('initial', 'y - y_0')
intercept.add_constraint('initial', 'psi - psi_0')

intercept.add_constraint('terminal', 'x - x_f')
intercept.add_constraint('terminal', 'y - y_f')
intercept.add_constraint('terminal', 'psi - psi_f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_dual = giuseppe.problems.symbolic.SymDual(intercept, control_method='algebraic').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_dual, verbose=2)

guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_dual, control=0., t_span=1.0, immutable_constants=('k', 'x_0', 'y_0')
)
seed_sol = num_solver.solve(guess)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'x_0': x0[0], 'y_0': x0[1], 'x_f': xf[0], 'y_f': xf[1]})
cont.add_linear_series(1, {'psi_0': x0[2], 'psi_f': xf[2]})
sol_set = cont.run_continuation()
sol_set.save('sol_set.data')

# -------------------------------------------------------------------------------------------------------------------- #
# CUSTOM SOLUTION                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
# Dynamic model
x_sym = ca.SX.sym('x', 3)
u_sym = ca.SX.sym('u', 1)
nx = x_sym.shape[0]
nu = u_sym.shape[0]
eom_sym = ca.vcat((ca.cos(x_sym[2]), ca.sin(x_sym[2]), u_sym))
eom_fun = ca.Function('f', (x_sym, u_sym,), (eom_sym,), ('x', 'u',), ('f',))

# Path cost model
path_cost_sym = 1. + k/2 * (u_sym*u_sym)
path_cost_fun = ca.Function('L', (x_sym, u_sym), (path_cost_sym,), ('x', 'u'), ('L',))

# Pseudospectral optimal control problem statement
n_phase = 2
n_col = 10
col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lg(n_col+1)
_, diff_mat_local = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local, col_points_local[1:], compute_diff_matrix=True, compute_interp_matrix=False
)

# Expand values for multi-phase
diff_mat = np.zeros(shape=(n_col*n_phase, (n_col+1)*n_phase))
col_points = np.tile(col_points_local, n_phase)
col_weights = np.tile(col_weights_local, n_phase)
intf_matrix = np.zeros(shape=((n_col+1)*n_phase, n_phase), dtype=col_weights.dtype)
col_points_global = np.empty_like(col_points)
col_points_global_linkeage = np.linspace(-1, 1, n_phase+1)
for phase in range(n_phase):
    diff_mat[phase*n_col:(phase+1)*n_col, phase*(n_col+1):(phase+1)*(n_col+1)] = diff_mat_local
    intf_matrix[phase*(n_col+1):(phase+1)*(n_col+1), phase] = col_weights_local @ diff_mat_local

    _middle = 0.5*(col_points_global_linkeage[phase] + col_points_global_linkeage[phase+1])
    _range = 0.5*(col_points_global_linkeage[phase+1] - col_points_global_linkeage[phase])
    col_points_global[phase*(n_col+1):(phase+1)*(n_col+1)] = _middle + _range*col_points_local

X_sym = ca.SX.sym('X', nx, (n_col+1)*n_phase)  # Include initial state
U_sym = ca.SX.sym('U', nu, n_col*n_phase)
idces_initial = np.arange(0, X_sym.shape[1], n_col+1)
idces_interior = np.delete(np.arange(0, X_sym.shape[1], 1), idces_initial)
# X0i_sym = X_sym[:, ::n_col+1]
tf_sym = ca.SX.sym('tf')
dt_phase = tf_sym / n_phase
z_sym = ca.vcat((
    ca.vec(X_sym),
    ca.vec(U_sym),
    tf_sym,
))

L_col = path_cost_fun(X_sym[:, idces_interior], U_sym)
f_col = eom_fun(X_sym[:, idces_interior], U_sym)
X0i_sym = X_sym[:, idces_initial]
Xfi_sym = X0i_sym + X_sym @ intf_matrix

integrated_cost = dt_phase/2 * (L_col @ col_weights)
dynamic_constraint = (dt_phase/2*f_col - X_sym @ diff_mat.T) * np.tile(col_weights[None, :], (3, 1))
initial_state_constraint = X0i_sym[:, 0] - x0
phase_linkage_constraint = X0i_sym[:, 1:] - Xfi_sym[:, :-1]
terminal_state_constraint = Xfi_sym[:, -1] - xf
boundary_constraints = ca.vec(ca.hcat((initial_state_constraint, phase_linkage_constraint, terminal_state_constraint)))

nlp = {
    'x': z_sym,  # Unknown variables
    'f': integrated_cost,  # Objective function
    'g': ca.vcat((
        boundary_constraints, ca.vec(dynamic_constraint)
    )),  # (In)equality constraints
}
nlp_solver = ca.nlpsol('NLP', 'ipopt', nlp)

# Initial guess (from outer solution)
X_outer = np.empty(shape=X_sym.shape, dtype=float)
X_outer[:2, :] = 0.5*(x0[:2, None] + xf[:2, None]) + 0.5*(xf[:2, None] - x0[:2, None]) * col_points_global[None, :]
X_outer[2, :] = psi_outer
U_outer = np.zeros(shape=U_sym.shape, dtype=float)

z_outer = np.concatenate((
    X_outer.ravel(order='F'),
    U_outer.ravel(order='F'),
    (tf_outer,),
))

# Bounds (stolen from known optimal solution)
ubx = sol_set[-1].x.max(axis=1, initial=-np.inf)
ubx[:2] += 10.
ubx[2] += 30 * np.pi/180
lbx = sol_set[-1].x.min(axis=1, initial=np.inf) - 1.
lbx[:2] -= 10.
lbx[2] -= 30 * np.pi/180
ubu = sol_set[-1].u.max(initial=-np.inf) + 10.
lbu = sol_set[-1].u.min(initial=np.inf) - 10.
ubtf = sol_set[-1].t[-1] + 10.
lbtf = 0.

ubz = np.empty_like(z_outer)
ubz[:X_sym.numel()] = np.tile(ubx, (n_col+1)*n_phase)
ubz[X_sym.numel():-1] = np.tile(ubu, n_col*n_phase)
ubz[-1] = ubtf
lbz = np.empty_like(z_outer)
lbz[:X_sym.numel()] = np.tile(lbx, (n_col+1)*n_phase)
lbz[X_sym.numel():-1] = np.tile(lbu, n_col*n_phase)
lbz[-1] = lbtf

nlp_sol = nlp_solver(x0=z_outer, lbg=0, ubg=0, lbx=lbz, ubx=ubz)

# Unpack solution
z_nlp = nlp_sol['x'].full().ravel()
X_nlp = z_nlp[:X_sym.numel()].reshape((nx, -1), order='F')
U_nlp = z_nlp[X_sym.numel():-1].reshape((nu, -1), order='F')
tf_nlp = z_nlp[-1]
t_nlp = tf_nlp*(1+col_points_global)/2

adjoints_nlp = nlp_sol['lam_g'].full().ravel()
nu0_nlp = adjoints_nlp[:nx]
nu_linkage_nlp = adjoints_nlp[nx:(n_phase-1)*n_phase*nx]
nuf_nlp = adjoints_nlp[n_phase*nx:(n_phase+1)*nx]
Lam_nlp = adjoints_nlp[(n_phase+1)*nx:].reshape((nx, -1), order='F')

# Save solution
sol_nlp = copy(sol_set[-1])
sol_nlp.t = t_nlp
sol_nlp.x = X_nlp
sol_nlp.lam = np.empty_like(X_nlp)
sol_nlp.lam[:, idces_initial] = np.nan
sol_nlp.lam[:, idces_interior] = Lam_nlp
sol_nlp.u = np.empty(shape=(nu, t_nlp.shape[0]), dtype=U_nlp.dtype)
sol_nlp.u[:, idces_initial] = np.nan
sol_nlp.u[:, idces_interior] = U_nlp
sol_nlp.nu0 = nu0_nlp
sol_nlp.nuf = nuf_nlp

with open('sol_nlp.data', 'wb') as f:
    pickle.dump(sol_nlp, f)
