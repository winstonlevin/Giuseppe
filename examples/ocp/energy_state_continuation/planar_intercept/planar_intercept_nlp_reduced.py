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

# Path cost parameters
kf = 1E-2

# Outer solution
dp = xf[:2] - x0[:2]
p_hat = dp / np.linalg.norm(dp)

psi_outer = np.arctan2(dp[1], dp[0])
u_outer = np.array((0.,))
lam_outer = np.array((-p_hat[0], -p_hat[1], 0.))
tf_outer = np.dot(dp, dp)**0.5

# -------------------------------------------------------------------------------------------------------------------- #
# NLP SOLUTION                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #
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
n_col = 8
n_int = 30

collocation_method = 'lgl'

if collocation_method == 'lg':
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lg(n_col + 1)
    # col_weights_local = np.insert(col_weights_local, 0, 0)
    idces_anchor_local = np.arange(0, 1, 1)
    idces_collocation_local = np.arange(1, n_col + 1, 1)

    proj_points_local, proj_weights_local = giuseppe.utils.pseudospectral.lg(n_int+1)
    proj_points_local = proj_points_local[1:]
elif collocation_method == 'lgr':
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points_local = np.append(col_points_local, 1.)
    idces_anchor_local = np.array((n_col,))
    idces_collocation_local = np.arange(0, n_col, 1)

    proj_points_local, proj_weights_local = giuseppe.utils.pseudospectral.lgr(n_int)
elif collocation_method == 'lgl':
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgl(n_col)
    idces_anchor_local = np.empty(shape=(0,), dtype=int)
    idces_collocation_local = np.arange(0, n_col, 1)

    proj_points_local, proj_weights_local = giuseppe.utils.pseudospectral.lgl(n_int)
elif collocation_method == 'zlg':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lg(n_col+1)
    col_points_local = np.sort(np.append(col_points_local[1:], 0))
    idces_anchor_local = np.where(col_points_local == 0)[0]
    idces_collocation_local = np.delete(np.arange(0, n_col+1, 1), idces_anchor_local)

    proj_points_local, proj_weights_local = giuseppe.utils.pseudospectral.lg(n_int+1)
    proj_points_local = proj_points_local[1:]
elif collocation_method == 'zlgr':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points_local = np.sort(np.append(col_points_local, 0))
    idces_anchor_local = np.where(col_points_local == 0)[0]
    idces_collocation_local = np.delete(np.arange(0, n_col+1, 1), idces_anchor_local)

    proj_points_local, proj_weights_local = giuseppe.utils.pseudospectral.lgr(n_int)
elif collocation_method == 'zlgl':
    assert n_col % 2 == 0, f"ZLGL requires an even number of collocation points, but n_col={n_col}!"
    col_points_local, col_weights_local = giuseppe.utils.pseudospectral.lgl(n_col)
    col_points_local = np.sort(np.append(col_points_local, 0))
    idces_anchor_local = np.where(col_points_local == 0)[0]
    idces_collocation_local = np.delete(np.arange(0, n_col+1, 1), idces_anchor_local)

    proj_points_local, proj_weights_local = giuseppe.utils.pseudospectral.lgl(n_int)
else:
    raise ValueError(f'collocation_method=={collocation_method} is not implemented!')

n_mesh = len(col_points_local)
_, diff_mat_local = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local, col_points_local[idces_collocation_local], compute_diff_matrix=True, compute_interp_matrix=False
)
int_mat_local = giuseppe.utils.pseudospectral.integration_matrix(
    col_points_local[idces_collocation_local], col_points_local[idces_collocation_local],
)
interp0f_col_local, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local[idces_collocation_local], np.array((-1, +1)), compute_interp_matrix=True, compute_diff_matrix=False
)  # Interpolate Lam/U to get 0/f values
interp0f_mesh_local, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local, np.array((-1, +1)), compute_interp_matrix=True, compute_diff_matrix=False
)  # Interpolate Lam/U to get 0/f values

projection_matrix_local, projection_diff_matrix_local = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local, proj_points_local, compute_interp_matrix=True, compute_diff_matrix=True
)
projection_col_matrix_local, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points_local[idces_collocation_local], proj_points_local, compute_interp_matrix=True, compute_diff_matrix=False
)

# Expand values for multi-phase
proj_mat = np.zeros(shape=(n_int*n_phase, n_mesh*n_phase))
proj_col_mat = np.zeros(shape=(n_int*n_phase, n_col*n_phase))
proj_diff_mat = np.zeros(shape=(n_int*n_phase, n_mesh*n_phase))
proj_weights = np.zeros(shape=(n_int*n_phase,))

diff_mat = np.zeros(shape=(n_col*n_phase, n_mesh*n_phase))
int_mat = np.zeros(shape=(n_col*n_phase, n_col*n_phase))
col_points = np.tile(col_points_local, n_phase)
col_weights = np.tile(col_weights_local, n_phase)
col_points_global = np.empty_like(col_points)
col_points_global_linkeage = np.linspace(-1, 1, n_phase+1)
interp0_col_matrix = np.zeros(shape=(n_col * n_phase, n_phase), dtype=interp0f_col_local.dtype)
interpf_col_matrix = np.zeros_like(interp0_col_matrix)
interp0_mesh_matrix = np.zeros(shape=(n_mesh * n_phase, n_phase), dtype=interp0f_col_local.dtype)
interpf_mesh_matrix = np.zeros_like(interp0_mesh_matrix)
for phase in range(n_phase):
    proj_mat[phase*n_int:(phase+1)*n_int, phase*n_mesh:(phase+1)*n_mesh] = projection_matrix_local
    proj_col_mat[phase*n_int:(phase+1)*n_int, phase*n_col:(phase+1)*n_col] = projection_col_matrix_local
    proj_diff_mat[phase*n_int:(phase+1)*n_int, phase*n_mesh:(phase+1)*n_mesh] = projection_diff_matrix_local
    proj_weights[phase*n_int:(phase+1)*n_int] = proj_weights_local

    diff_mat[phase*n_col:(phase+1)*n_col, phase*n_mesh:(phase+1)*n_mesh] = diff_mat_local
    int_mat[phase*n_col:(phase+1)*n_col, phase*n_col:(phase+1)*n_col] = int_mat_local  # Diagonal block is current integration
    int_mat[phase*n_col:(phase+1)*n_col, :phase*n_col] = np.tile(col_weights_local[None, :], (n_col, phase,))
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
nx_mesh = nxr*n_mesh*n_phase
nu_mesh = nu*n_col*n_phase

Xproj_sym = X_sym @ proj_mat.T
Uproj_sym = U_sym @ proj_col_mat.T

idces_initial = np.arange(0, X_sym.shape[1], n_mesh)

tf_sym = ca.SX.sym('tf')
dt_phase = tf_sym / n_phase
z_sym = ca.vcat((
    X_sym[idces_int_output, 0],
    ca.vec(X_sym[idces_state, :]),
    ca.vec(U_sym),
    tf_sym,
))

L_proj = path_cost_fun(Xproj_sym, Uproj_sym)
f_proj = eom_fun(Xproj_sym, Uproj_sym)
X0i_sym = X_sym @ interp0_mesh_matrix
Xfi_sym = X_sym @ interpf_mesh_matrix

integrated_cost = end_cost_fun(Xfi_sym[:, -1]) + dt_phase/2 * (L_proj @ proj_weights)
dynamic_residual = (
    dt_phase/2*f_proj[idces_state, :] - X_sym[idces_state, :] @ proj_diff_mat.T
) * np.tile(proj_weights[None, :], (nxr, 1))

xf_int_sym = X_sym[idces_int_output, 0] + dt_phase/2*f_proj[idces_int_output, :] @ proj_weights

# dynamic_constraint = ca.vec(dynamic_residual)
collocated_residual = dynamic_residual @ proj_col_mat
dynamic_constraint = ca.vec(collocated_residual)

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
ubx = np.maximum(x0, xf)
ubx[:2] += 10.
ubx[2] += np.pi/180
lbx = np.minimum(x0, xf)
lbx[:2] -= 10.
lbx[2] -= np.pi

ubu = 1/kf
lbu = -ubu
ubtf = tf_outer * 2.
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

nlp_sol = nlp_solver(x0=z_outer, lbg=0, ubg=0, lbx=lbz, ubx=ubz)

# Unpack solution
z_nlp = nlp_sol['x'].full().ravel()
X_nlp = np.empty(shape=(nx, n_mesh*n_phase))
X_nlp[idces_state, :] = z_nlp[n_int_output:n_int_output+nx_mesh].reshape((nxr, -1), order='F')
X_nlp[idces_int_output, 0] = z_nlp[:n_int_output]
X_nlp[idces_int_output, 1:] = np.nan

U_nlp = z_nlp[n_int_output+nx_mesh:n_int_output+nx_mesh+nu_mesh].reshape((nu, -1), order='F')
tf_nlp = z_nlp[n_int_output+nx_mesh+nu_mesh]
t_nlp = tf_nlp*(1+col_points_global)/2

# Calculate integrated states
f_col_nlp = eom_fun(X_nlp[:, idces_collocation], U_nlp).full()
for idx_int in idces_int_output:
    X_nlp[idx_int, idces_collocation] = X_nlp[idx_int, 0:1] + tf_nlp/(2*n_phase)*f_col_nlp[idx_int, :] @ int_mat.T

# Costate information
adjoints_nlp = nlp_sol['lam_g'].full().ravel()
nu0_nlp = adjoints_nlp[:nx]
nu_linkage_nlp = adjoints_nlp[nx:nx+(n_phase-1)*nxr].reshape((nxr, -1), order='F')
nuf_nlp = np.empty_like(nu0_nlp)
nuf_nlp[idces_state] = adjoints_nlp[nx+(n_phase-1)*nxr:nx+n_phase*nxr]
nuf_nlp[idces_int_output] = adjoints_nlp[nx+n_phase*nxr:nx+n_phase*nxr+n_int_output]

lam_nlp = np.empty(shape=(nx, n_col * n_phase))
lam_nlp[idces_int_output, :] = np.nan
lam_nlp[idces_state, :] = adjoints_nlp[nx+n_phase*nxr+n_int_output:nx+n_phase*nxr+n_int_output + n_col*n_phase*nxr].reshape((nxr, -1), order='F')
for idx_int_output in idces_int_output:
    lam_nlp[idx_int_output, :] = nuf_nlp[idx_int_output]


# Save solution ------------------------------------------------------------------------------------------------------ #
sol_nlp = giuseppe.data_classes.Solution()
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
