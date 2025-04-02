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
xf[1] = -2.
xf[2] = np.pi/3

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

k0 = 1.
kf = 1E-2
intercept.add_constant('k', k0)
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
cont.add_logarithmic_series(5, {'k': kf})
sol_set = cont.run_continuation()
sol_set.save('sol_set.data')

# -------------------------------------------------------------------------------------------------------------------- #
# NLP SOLUTION                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# CONTINUATION SETTINGS:                                                                              #
# IF use_continuation -> s=0 corresponds with "energy state" (psi is a control variable, not a state) #
#                        s=1 corresponds with true state (psi is time-varying)                        #
# --------------------------------------------------------------------------------------------------- #
use_continuation = False
idces_slow = np.array((0, 1))
idces_fast = np.array((2,))  # The heading is the "fast" state

# --------------------------------------------------------------------------------- #
# STATE REDUCTION SETTINGS:                                                         #
# IF reduce_state -> Eliminate x(t), y(t), lamX(t), lamY(t),                        #
#                    replace with integral constraint xf-x0-int(dx/dt dt) = 0       #
# --------------------------------------------------------------------------------- #
reduce_state = True
if reduce_state:
    idces_int_output = idces_slow  # Dynamics do not depend on (x,y)
    idces_state = idces_fast
else:
    idces_state = np.arange(0, 3, 1)
    idces_int_output = np.empty(shape=(0,), dtype=idces_state.dtype)

n_int_output = idces_int_output.size

# Dynamic model
x_sym = ca.SX.sym('x', 3)
u_sym = ca.SX.sym('u', 1)
eom_sym = ca.vcat((ca.cos(x_sym[2]), ca.sin(x_sym[2]), u_sym))

# Path cost model
end_cost_sym = ca.SX.zeros(1)
path_cost_sym = 1. + kf / 2 * (u_sym * u_sym)

end_cost_fun = ca.Function('Phi', (x_sym,), (end_cost_sym,), ('x',), ('Phi',))
path_cost_fun = ca.Function('L', (x_sym, u_sym), (path_cost_sym,), ('x', 'u'), ('L',))
eom_fun = ca.Function('f', (x_sym, u_sym,), (eom_sym,), ('x', 'u',), ('f',))

# Pseudospectral optimal control problem statement
nx = x_sym.shape[0]
nu = u_sym.shape[0]

if reduce_state:
    nxr = idces_state.size
else:
    nxr = nx

n_phase = 1
n_col = 5

collocation_method = 'lg'

if collocation_method == 'lg':
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lg(n_col + 1)
    # col_weights_local = np.insert(col_weights_local, 0, 0)
    idces_anchor_local = np.arange(0, 1, 1)
    idces_collocation_local = np.arange(1, n_col + 1, 1)
elif collocation_method == 'lgr':
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points_local = np.append(col_points_local, 1.)
    idces_anchor_local = np.array((n_col,))
    idces_collocation_local = np.arange(0, n_col, 1)
elif collocation_method == 'lgl':
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgl(n_col)
    idces_anchor_local = np.empty(shape=(0,), dtype=int)
    idces_collocation_local = np.arange(0, n_col, 1)
elif collocation_method == 'zlg':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lg(n_col+1)
    col_points_local = np.sort(np.append(col_points_local[1:], 0))
    idces_anchor_local = np.where(col_points_local == 0)[0]
    idces_collocation_local = np.delete(np.arange(0, n_col+1, 1), idces_anchor_local)
elif collocation_method == 'zlgr':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points_local = np.sort(np.append(col_points_local, 0))
    idces_anchor_local = np.where(col_points_local == 0)[0]
    idces_collocation_local = np.delete(np.arange(0, n_col+1, 1), idces_anchor_local)
elif collocation_method == 'zlgl':
    assert n_col % 2 == 0, f"ZLGL requires an even number of collocation points, but n_col={n_col}!"
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgl(n_col)
    col_points_local = np.sort(np.append(col_points_local, 0))
    idces_anchor_local = np.where(col_points_local == 0)[0]
    idces_collocation_local = np.delete(np.arange(0, n_col+1, 1), idces_anchor_local)
else:
    raise ValueError(f'collocation_method=={collocation_method} is not implemented!')

n_mesh = len(col_points_local)
_, diff_mat_local = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local, col_points_local[idces_collocation_local], compute_diff_matrix=True, compute_interp_matrix=False
)
interp0f_col_local, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local[idces_collocation_local], np.array((-1, +1)), compute_interp_matrix=True, compute_diff_matrix=False
)  # Interpolate Lam/U to get 0/f values
interp0f_mesh_local, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local, np.array((-1, +1)), compute_interp_matrix=True, compute_diff_matrix=False
)  # Interpolate Lam/U to get 0/f values

# Expand values for multi-phase
diff_mat = np.zeros(shape=(n_col*n_phase, n_mesh*n_phase))
col_points = np.tile(col_points_local, n_phase)
col_weights = np.tile(col_weights_local, n_phase)
col_points_global = np.empty_like(col_points)
col_points_global_linkeage = np.linspace(-1, 1, n_phase+1)
interp0_col_matrix = np.zeros(shape=(n_col * n_phase, n_phase), dtype=interp0f_col_local.dtype)
interpf_col_matrix = np.zeros_like(interp0_col_matrix)
interp0_mesh_matrix = np.zeros(shape=(n_mesh * n_phase, n_phase), dtype=interp0f_col_local.dtype)
interpf_mesh_matrix = np.zeros_like(interp0_mesh_matrix)
for phase in range(n_phase):
    diff_mat[phase*n_col:(phase+1)*n_col, phase*n_mesh:(phase+1)*n_mesh] = diff_mat_local
    interp0_col_matrix[phase * n_col:(phase + 1) * n_col, phase] = interp0f_col_local[0, :]
    interpf_col_matrix[phase * n_col:(phase + 1) * n_col, phase] = interp0f_col_local[1, :]
    interp0_mesh_matrix[phase * n_mesh:(phase + 1) * n_mesh, phase] = interp0f_mesh_local[0, :]
    interpf_mesh_matrix[phase * n_mesh:(phase + 1) * n_mesh, phase] = interp0f_mesh_local[1, :]

    _middle = 0.5*(col_points_global_linkeage[phase] + col_points_global_linkeage[phase+1])
    _range = 0.5*(col_points_global_linkeage[phase+1] - col_points_global_linkeage[phase])
    col_points_global[phase*n_mesh:(phase+1)*n_mesh] = _middle + _range*col_points_local

idces_collocation = np.concatenate([idces_collocation_local+n_mesh*_phase for _phase in range(n_phase)])
idces_anchor = np.concatenate([idces_anchor_local+n_mesh*_phase for _phase in range(n_phase)])

X_sym = ca.SX.sym('X', nx, n_mesh*n_phase)  # Include initial state
U_sym = ca.SX.sym('U', nu, n_col*n_phase)
s_sym = ca.SX.sym('s')
nx_mesh = nxr*n_mesh*n_phase
nu_mesh = nu*n_col*n_phase

idces_initial = np.arange(0, X_sym.shape[1], n_mesh)

tf_sym = ca.SX.sym('tf')
dt_phase = tf_sym / n_phase
z_sym = ca.vcat((
    X_sym[idces_int_output, 0],
    ca.vec(X_sym[idces_state, :]),
    ca.vec(U_sym),
    tf_sym,
))
if use_continuation:
    z_sym = ca.vcat((z_sym, s_sym))

L_col = path_cost_fun(X_sym[:, idces_collocation], U_sym)
f_col = eom_fun(X_sym[:, idces_collocation], U_sym)
X0i_sym = X_sym @ interp0_mesh_matrix
Xfi_sym = X_sym @ interpf_mesh_matrix

integrated_cost = end_cost_fun(Xfi_sym[:, -1]) + dt_phase/2 * (L_col @ col_weights)
dynamic_constraint = (
                             dt_phase/2*f_col[idces_state, :] - X_sym[idces_state, :] @ diff_mat.T
                     ) * np.tile(col_weights[None, :], (nxr, 1))
if use_continuation:
    # Add continuation into dynamics
    dynamic_constraint[idces_fast, :] = dt_phase/2*f_col[idces_fast, :] - s_sym * (X_sym @ diff_mat.T)[idces_fast, :]

xf_int_sym = X_sym[idces_int_output, 0] + tf_sym/2*f_col[idces_int_output, :] @ col_weights

# if reduce_state:
#     dynamic_constraint = ca.vcat((ca.vec(dynamic_constraint[2, :]), ca.sum2(dynamic_constraint[:2, :])))
# else:
dynamic_constraint = ca.vec(dynamic_constraint)
initial_state_constraint = X0i_sym[:, 0] - x0
phase_linkage_constraint = X0i_sym[idces_state, 1:] - Xfi_sym[idces_state, :-1]
terminal_state_constraint = Xfi_sym[idces_state, -1] - xf[idces_state]
terminal_integral_constraint = xf_int_sym[idces_int_output] - xf[idces_int_output]
boundary_constraints = ca.vcat((
    ca.vec(initial_state_constraint),
    ca.vec(phase_linkage_constraint),
    ca.vec(terminal_state_constraint),
    terminal_integral_constraint
))
if use_continuation:
    boundary_constraints = ca.vcat((boundary_constraints, s_sym - 1.))

nlp = {
    'x': z_sym,  # Unknown variables
    'f': integrated_cost,  # Objective function
    'g': ca.vcat((
        boundary_constraints,
        dynamic_constraint,
    )),  # (In)equality constraints
}
nlp_solver = ca.nlpsol('NLP', 'ipopt', nlp)

# Initial guess (from outer solution)
X_outer = np.empty(shape=X_sym.shape, dtype=float)
X_outer[:2, :] = 0.5*(x0[:2, None] + xf[:2, None]) + 0.5*(xf[:2, None] - x0[:2, None]) * col_points_global[None, :]
X_outer[2, :] = psi_outer
U_outer = np.zeros(shape=U_sym.shape, dtype=float)

z_outer = np.concatenate((
    X_outer[idces_int_output, 0],
    X_outer[idces_state, :].ravel(order='F'),
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
ubz[:n_int_output] = ubx[idces_int_output]
ubz[n_int_output:n_int_output+nx_mesh] = np.tile(ubx[idces_state], n_mesh*n_phase)
ubz[n_int_output+nx_mesh:n_int_output+nx_mesh+nu_mesh] = np.tile(ubu, n_col*n_phase)
ubz[n_int_output+nx_mesh+nu_mesh] = ubtf
lbz = np.empty_like(z_outer)
lbz[:n_int_output] = lbx[idces_int_output]
lbz[n_int_output:n_int_output+nx_mesh] = np.tile(lbx[idces_state], n_mesh*n_phase)
lbz[n_int_output+nx_mesh:n_int_output+nx_mesh+nu_mesh] = np.tile(lbu, n_col*n_phase)
lbz[n_int_output+nx_mesh+nu_mesh] = lbtf

if use_continuation:
    z_outer = np.append(z_outer, 0.)
    lbz = np.append(lbz, 0.)
    ubz = np.append(ubz, 1.)

nlp_sol = nlp_solver(x0=z_outer, lbg=0, ubg=0, lbx=lbz, ubx=ubz)

# Unpack solution
z_nlp = nlp_sol['x'].full().ravel()
X_nlp = np.empty(shape=(nx, n_mesh*n_phase))
X_nlp[idces_int_output, 0] = z_nlp[:n_int_output]
# X_nlp[idces_int_output, -1] = z_nlp[n_int_output:2*n_int_output]
X_nlp[idces_int_output, 1:] = np.nan
X_nlp[idces_state, :] = z_nlp[n_int_output:n_int_output+nx_mesh].reshape((nxr, -1), order='F')

U_nlp = z_nlp[n_int_output+nx_mesh:n_int_output+nx_mesh+nu_mesh].reshape((nu, -1), order='F')
tf_nlp = z_nlp[n_int_output+nx_mesh+nu_mesh]
t_nlp = tf_nlp*(1+col_points_global)/2

adjoints_nlp = nlp_sol['lam_g'].full().ravel()
nu0_nlp = adjoints_nlp[:nx]
nu_linkage_nlp = adjoints_nlp[nx:nx+(n_phase-1)*nxr].reshape((nx, -1), order='F')
nuf_nlp = np.empty_like(nu0_nlp)
nuf_nlp[idces_state] = adjoints_nlp[nx+(n_phase-1)*nxr:nx+n_phase*nxr]
nuf_nlp[idces_int_output] = adjoints_nlp[nx+n_phase*nxr:(n_phase+1)*nx]

lam_nlp = np.empty(shape=(nx, n_col * n_phase))
lam_nlp[idces_int_output, :] = np.nan
lam_nlp[idces_state, :] = adjoints_nlp[(n_phase + 1) * nx:(n_phase + 1) * nx + n_phase * n_col * nxr].reshape((nxr, -1), order='F')
for idx_int_output in idces_int_output:
    lam_nlp[idx_int_output, :] = nuf_nlp[idx_int_output]
# if reduce_state:
#     lam_nlp = np.empty(shape=(nx, n_col*n_phase))
#     for idx, idx_int_output in enumerate(idces_int_output):
#         lam_nlp[idx_int_output, :] = adjoints_nlp[(n_phase+1)*nx+n_phase*n_col + idx]
#     lam_nlp[idces_state, :] = adjoints_nlp[(n_phase+1)*nx:(n_phase+1)*nx+n_phase*n_col*nxr].reshape((1, -1), order='F')
#
# else:
#     lam_nlp = adjoints_nlp[(n_phase + 1) * nx:(n_phase + 1) * nx + n_phase * nx * n_col].reshape((nx, -1), order='F')
# lam_nlp = adjoints_nlp[n_phase * nx:].reshape((nx, -1), order='F')
# nuf_nlp = lam_nlp @ interpf_col_matrix

# Save solution ------------------------------------------------------------------------------------------------------ #
sol_nlp = copy(sol_set[-1])
sol_nlp.t = t_nlp
sol_nlp.x = X_nlp
sol_nlp.lam = np.empty_like(sol_nlp.x)
sol_nlp.lam[:] = np.nan
sol_nlp.lam[:, idces_collocation] = lam_nlp
# sol_nlp.lam[:, idces_initial] = np.hstack((-nu0_nlp[:, None], -nu_linkage_nlp))
sol_nlp.lam[:, idces_initial] = lam_nlp @ interp0_col_matrix
sol_nlp.u = np.empty(shape=(nu, t_nlp.shape[0]), dtype=U_nlp.dtype)
sol_nlp.u[:] = np.nan
sol_nlp.u[:, idces_collocation] = U_nlp
sol_nlp.u[:, idces_initial] = U_nlp @ interp0_col_matrix
sol_nlp.nu0 = nu0_nlp
sol_nlp.nuf = nuf_nlp

# Add jump values
idces_fi = np.append(idces_initial[1:], len(t_nlp))
t_nlp_fi = np.append(t_nlp[idces_fi[:-1]], tf_nlp)
X_nlp_fi = X_nlp @ interpf_mesh_matrix
# lam_nlp_fi = np.hstack((-nu_linkage_nlp, nuf_nlp[:, None]))
if collocation_method in ['lg', 'lgl', 'zlg', 'zlgl']:
    lam_nlp_fi = lam_nlp @ interpf_col_matrix
elif collocation_method in ['lgr', 'zlgr']:
    lam_nlp_fi = np.vstack([
        lam_nlp[:, _phase*n_col:(_phase+1)*n_col] @ (col_weights_local * diff_mat_local[:, -1])
        for _phase in range(n_phase)
    ]).T
else:
    raise ValueError(f'collocation_method=={collocation_method} is not implemented!')

u_nlp_fi = U_nlp @ interpf_col_matrix

sol_nlp.t = np.insert(sol_nlp.t, idces_fi, t_nlp_fi)
sol_nlp.x = np.insert(sol_nlp.x, idces_fi, X_nlp_fi, axis=1)
sol_nlp.lam = np.insert(sol_nlp.lam, idces_fi, lam_nlp_fi, axis=1)
sol_nlp.u = np.insert(sol_nlp.u, idces_fi, u_nlp_fi, axis=1)

with open('sol_nlp.data', 'wb') as f:
    pickle.dump(sol_nlp, f)
