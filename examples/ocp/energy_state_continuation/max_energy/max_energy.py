import pickle

import numpy as np
import matplotlib
import casadi as ca

import giuseppe

matplotlib.use('TkAgg', force=True)

# -------------------------------------------------------------------------------------------------------------------- #
# PROBLEM SETUP                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
# Constants -------------------------------------- #
# (from https://doi/org/10.2514/6.1968-877)
mass = 340.1943  # kg
g0 = 9.80665  # free fall acceleration [m/s2]
re = 6_378_000  # Earth's radius [m]

# Aerodynamic constants
Sref = 0.2919  # m2
CLa = 1.5658  # [-]
CD0 = 0.0612  # [-]
CDi = 1.6537  # [-]

# Atmosphere function
rho0 = 1.2  # Sea-level density [kg/m3]
h_ref = 7_500  # Density decay rate [m]

# Boundary conditions
# (initial)
h0 = 40e3  # [m]
lon0 = 0.
lat0 = 0.
V0 = 2e3  # [m/s]
gam0 = 0.  # [rad]
psi0 = 0.  # [rad]

# (terminal)
hf = 0.
lonf = 3. * np.pi/180
latf = 1. * np.pi/180

lamVf = 0.
lamgamf = 0.
lampsif = 0.
# ------------------------------------------------ #

# Symbolic expressions to derive necessary conditions for optimality ------------------------------------------------- #
tf_sym = ca.SX.sym('tf')
tau_sym = ca.SX.sym('tau')

h_sym = ca.SX.sym('h')
lat_sym = ca.SX.sym('lat')
lon_sym = ca.SX.sym('lon')
V_sym = ca.SX.sym('V')
gam_sym = ca.SX.sym('gam')
psi_sym = ca.SX.sym('psi')

alpha_sym = ca.SX.sym('alpha')
sig_sym = ca.SX.sym('sig')

control_sym = ca.vcat((alpha_sym, sig_sym))
state_sym = ca.vcat((h_sym, lat_sym, lon_sym, V_sym, gam_sym, psi_sym))
costate_sym = ca.vcat([ca.SX.sym('lam_' + _x.name()) for _x in ca.vertsplit(state_sym)])

nx = state_sym.shape[0]
nu = control_sym.shape[0]

# Algebraic expressions
R_sym = h_sym + re
g_sym = g0 * (re/R_sym)**2
rho_sym = rho0 * ca.exp(-h_sym/h_ref)
wing_load_sym = (0.5*rho_sym*V_sym**2) * Sref / mass
drag_sym = wing_load_sym * (CD0 + CDi*alpha_sym**2)
lift_sym = wing_load_sym * (CLa * alpha_sym)
Vlat_sym = V_sym * ca.cos(gam_sym)

# Dynamics
eom_state_sym = ca.vcat((
    V_sym * ca.sin(gam_sym),
    ((V_sym/R_sym) * ca.cos(gam_sym)) * ca.sin(psi_sym),
    ((V_sym/R_sym) * ca.cos(gam_sym)) * ca.cos(psi_sym)/ca.cos(lat_sym),
    -drag_sym - g_sym*ca.sin(gam_sym),
    (lift_sym*ca.cos(sig_sym) - (g_sym - V_sym**2/R_sym))/V_sym,
    (lift_sym*ca.sin(sig_sym) - Vlat_sym**2/R_sym * ca.cos(psi_sym)*ca.tan(lat_sym))/Vlat_sym
))

# Path cost
path_cost_sym = -eom_state_sym[3]  # Max Vf - V0

path_cost_fun = ca.Function('L', (state_sym, control_sym), (path_cost_sym,), ('x', 'u'), ('L',))
eom_state_fun = ca.Function('f', (state_sym, control_sym), (eom_state_sym,), ('x', 'u'), ('f',))

# Boundary conditions
# (initial)
initial_state = np.array((h0, lat0, lon0, V0, gam0, psi0))
bc0_sym = state_sym - initial_state

# (terminal)
terminal_pos = np.array((hf, latf, lonf))
bcf_sym = state_sym[:3] - terminal_pos

bc0_fun = ca.Function('BC0', (state_sym,), (bc0_sym,), ('x',), ('BC0',))
bcf_fun = ca.Function('BCf', (state_sym,), (bcf_sym,), ('x',), ('BCf',))

# Path constraints
control_lower_bound = np.array((0., -np.pi))  # Make AoA>=0 to disambiguate sign of AoA/Bank
control_upper_bound = np.array((40*np.pi/180, np.pi))

# -------------------------------------------------------------------------- #
# SCALING PROCEDURE:                                                         #
# Since the cost function is the LSE of the                                  #
# dynamic error, it is imperative that the dynamics are properly scaled,     #
# otherwise states with large magnitudes will dominate the cost. Each state  #
# is scaled so that the dimension state X is written in terms of the non-    #
# dimensional state Xnd, its mean Mx, and its range Rx as:                   #
#                              X = Mx + Rx*Xnd                               #
# Additionally, and more importantly, the dynamic error must be scaled. The  #
# dynamics of the dimensional state are scaled as:
#                              F = Mf + Rf * Fnd
# But the dynamics of the state are:
#                              X' = Rx*Xnd' = Mf + Rf * Fnd
# So that the nondimensionalized dynamic error of the nondimensionalized
# state is:
#                              Xnd' = (Mf + Rf * Fnd)/Rx
#                              Fnd  = (Rx*Xnd' - Mf)/Rf
# And the "Estimated" dynamics error is:
#                              XndEst  = A * phi
#                              XndEst' = A * phi'
# So that the appropriate error term is:
#                              dynErr = (Xnd' - XndEst')*Rx/Rf
use_state_scaling = True
use_control_scaling = True
use_costate_scaling = True

# Scaling
if use_state_scaling:
    # state_bias, state_scale = np.array((
    #     (0.,     abs(hf - h0)),      # h
    #     (0., abs(latf - lat0)),  # Lat
    #     (0., abs(lonf - lon0)),  # Lon
    #     (0.,            V0),                # V
    #     (0.,              0.5*np.pi),           # gam
    #     (0.,              0.5*np.pi),           # psi
    # )).T

    state_bias, state_scale = np.array((
        ((hf + h0)/2,     abs(hf - h0)/2),      # h
        ((latf + lat0)/2, abs(latf - lat0)/2),  # Lat
        ((lonf + lon0)/2, abs(lonf - lon0)/2),  # Lon
        (V0/2,            V0/2),                # V
        (0.,              0.5*np.pi),           # gam
        (0.,              0.5*np.pi),           # psi
    )).T
else:
    state_bias = np.zeros_like(initial_state)
    state_scale = np.ones_like(initial_state)

state_inv_scale = 1 / state_scale
V_scale = state_scale[3]

if use_control_scaling:
    control_bias = 0.5*(control_upper_bound + control_lower_bound)
    control_scale = 0.5*(control_upper_bound - control_lower_bound)
else:
    control_bias = np.zeros_like(control_lower_bound)
    control_scale = np.ones_like(control_lower_bound)


if use_costate_scaling:
    state_dynamics_bias, state_dynamics_scale = np.array((
        (0., V_scale),  # dh/dt
        (0., V_scale/re),  # dLat/dt
        (0., V_scale/re),  # dLon/dt
        (0., g0),  # dV/dt
        (0., g0/V_scale),  # dgam/dt
        (0., g0/V_scale),  # dpsi/dt
    )).T
    path_cost_scale = g0  # L = dV/dt
else:
    state_dynamics_bias = np.zeros_like(initial_state)
    state_dynamics_scale = np.ones_like(initial_state)
    path_cost_scale = 1.

bc0_scale = state_scale
bcf_scale = state_scale[:3]

x0_nd = (initial_state - state_bias) / state_scale
pf_nd = (terminal_pos - state_bias[:3]) / state_scale[:3]

# ----------------------------------------------------------------------------- #
# Discretization of continuous signals                                          #
# ----------------------------------------------------------------------------- #
n_col = 30  # Number of basis functions for state estimate (1 more than costate/control)
n_int = 100  # Number of integration locations


collocation_method = 'lg'

if collocation_method == 'lg':
    col_points, col_weights = giuseppe.utils.pseudospectral.lg(n_col + 1)
    # col_weights = np.insert(col_weights, 0, 0)
    idces_anchor = np.arange(0, 1, 1)
    idces_collocation = np.arange(1, n_col + 1, 1)
    anchor = 'initial'

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
    proj_points = proj_points[1:]
elif collocation_method == 'lgr':
    col_points, col_weights = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points = np.append(col_points, 1.)
    idces_anchor = np.array((n_col,))
    idces_collocation = np.arange(0, n_col, 1)
    anchor = 'initial'

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgr(n_int)
elif collocation_method == 'lgl':
    col_points, col_weights = giuseppe.utils.pseudospectral.lgl(n_col)
    idces_anchor = np.empty(shape=(0,), dtype=int)
    idces_collocation = np.arange(0, n_col, 1)
    anchor = 'none'

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgl(n_int)
elif collocation_method == 'zlg':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points, col_weights = giuseppe.utils.pseudospectral.lg(n_col+1)
    col_points = np.sort(np.append(col_points[1:], 0))
    idces_anchor = np.where(col_points == 0)[0]
    idces_collocation = np.delete(np.arange(0, n_col+1, 1), idces_anchor)
    anchor = 'middle'

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
    proj_points = proj_points[1:]
elif collocation_method == 'zlgr':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points, col_weights = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points = np.sort(np.append(col_points, 0))
    idces_anchor = np.where(col_points == 0)[0]
    idces_collocation = np.delete(np.arange(0, n_col+1, 1), idces_anchor)
    anchor = 'middle'

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgr(n_int)
elif collocation_method == 'zlgl':
    assert n_col % 2 == 0, f"ZLGL requires an even number of collocation points, but n_col={n_col}!"
    col_points, col_weights = giuseppe.utils.pseudospectral.lgl(n_col)
    col_points = np.sort(np.append(col_points, 0))
    idces_anchor = np.where(col_points == 0)[0]
    idces_collocation = np.delete(np.arange(0, n_col+1, 1), idces_anchor)
    anchor = 'middle'

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgl(n_int)
else:
    raise ValueError(f'collocation_method=={collocation_method} is not implemented!')

n_mesh = len(col_points)
proj_mat, proj_diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points, proj_points, compute_interp_matrix=True, compute_diff_matrix=True
)
proj_col_mat, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points[idces_collocation], proj_points, compute_interp_matrix=True, compute_diff_matrix=False
)
interp0f_mesh, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points, np.array((-1, +1)), compute_interp_matrix=True, compute_diff_matrix=False
)  # Interpolate Lam/U to get 0/f values

X_sym = ca.SX.sym('X', nx, n_mesh)  # Include initial state
U_sym = ca.SX.sym('U', nu, n_col)
nx_mesh = nx*n_mesh
nu_col = nu * n_col

Xproj_sym = state_bias[:, None] + state_scale[:, None] * (X_sym @ proj_mat.T)
DXproj_sym = state_scale[:, None] * (X_sym @ proj_diff_mat.T)
Uproj_sym = control_bias[:, None] + control_scale[:, None] * (U_sym @ proj_col_mat.T)

tf_sym = ca.SX.sym('tf')
z_sym = ca.vcat((
    ca.vec(X_sym),
    ca.vec(U_sym),
    tf_sym,
))

L_proj_sym = path_cost_fun(Xproj_sym, Uproj_sym) / path_cost_scale
f_proj_sym = eom_state_fun(Xproj_sym, Uproj_sym)
X0i_sym = X_sym @ interp0f_mesh[0]
Xfi_sym = X_sym @ interp0f_mesh[1]

integrated_cost_sym = tf_sym / 2 * (L_proj_sym @ proj_weights)
dynamic_residual_sym = (
    tf_sym / 2 * f_proj_sym - DXproj_sym
) * np.tile(proj_weights[None, :], (nx, 1))
dynamic_residual_sym /= state_dynamics_scale[:, None]
collocated_residual_sym = dynamic_residual_sym @ proj_col_mat
dynamic_constraint_sym = ca.vec(collocated_residual_sym)

bc0_sym = X0i_sym - x0_nd
bcf_sym = Xfi_sym[:3] - pf_nd
boundary_constraints = ca.vcat((bc0_sym, bcf_sym))

nlp = {
    'x': z_sym,  # Unknown variables
    'f': integrated_cost_sym,  # Objective function
    'g': ca.vcat((boundary_constraints, dynamic_constraint_sym))  # Equality constraints
}
nlp_solver = ca.nlpsol('NLP', 'ipopt', nlp)

# Initial guess ------------------------------------------- #
xnd0_guess = x0_nd
xndf_guess = np.empty_like(xnd0_guess)
xndf_guess[:3] = pf_nd
xndf_guess[3:] = x0_nd[3:]
xnd_guess = 0.5*((xnd0_guess + xndf_guess)[:, None] + (xndf_guess - xnd0_guess)[:, None] * col_points[None, :])

und_guess = np.empty(shape=U_sym.shape, dtype=xnd0_guess.dtype)
und_guess[:] = -(control_bias / control_scale)[:, None]  # u = 0

# For final time, guess based on boundary conditions and scales
tf_min = np.linalg.norm(np.array((1., re, re))*(terminal_pos - initial_state[:3]) / initial_state[3])
tf_guess = tf_min

z_guess = np.concatenate((
    xnd_guess.ravel(order='F'),
    und_guess.ravel(order='F'),
    (tf_guess,),
))

lb_state = np.empty_like(initial_state)
ub_state = np.empty_like(initial_state)
lb_state[0] = -1_000.  # Altitude
ub_state[0] = 100_000.
lb_state[1:3] = initial_state[1:3]  # Lat/lon
ub_state[1:3] = terminal_pos[1:3]
lb_state[3] = 10.
ub_state[3] = 2*initial_state[3]
lb_state[4] = -85*np.pi/180  # FPA
ub_state[4] = 85*np.pi/180
lb_state[5] = -np.pi  # Heading
ub_state[5] = np.pi

lbx = (lb_state - state_bias) / state_scale
ubx = (ub_state - state_bias) / state_scale
lbu = (control_lower_bound - control_bias) / control_scale
ubu = (control_upper_bound - control_bias) / control_scale

lbtf = tf_min
ubtf = tf_min*3.

lbz = np.empty_like(z_guess)
lbz[:nx_mesh] = np.tile(lbx, n_mesh)
lbz[nx_mesh:nx_mesh + nu_col] = np.tile(lbu, n_col)
lbz[nx_mesh + nu_col] = lbtf
ubz = np.empty_like(z_guess)
ubz[:nx_mesh] = np.tile(ubx, n_mesh)
ubz[nx_mesh:nx_mesh + nu_col] = np.tile(ubu, n_col)
ubz[nx_mesh + nu_col] = ubtf

nlp_sol = nlp_solver(x0=z_guess, lbg=0, ubg=0, lbx=lbz, ubx=ubz)


def unpack_solution(_z_nlp, _adjoints_nlp=None):
    # Primal information
    X_nlp = state_bias[:, None] + state_scale[:, None] * _z_nlp[:nx_mesh].reshape((nx, -1), order='F')

    U_nlp = np.empty(shape=(nu, n_mesh))
    U_nlp[:, idces_anchor] = np.nan
    U_nlp[:, idces_collocation] = control_bias[:, None] \
        + control_scale[:, None] * _z_nlp[nx_mesh:nx_mesh + nu_col].reshape((nu, -1), order='F')

    # Unwrap angles
    sig_unwrapped = np.unwrap(U_nlp[1, idces_collocation], period=np.pi)
    flip_sign = np.zeros(shape=(n_mesh,), dtype=bool)
    flip_sign[idces_collocation] = np.not_equal(np.sign(sig_unwrapped), np.sign(U_nlp[1, idces_collocation]))
    U_nlp[0, flip_sign] *= -1
    U_nlp[1, idces_collocation] = sig_unwrapped

    tf_nlp = _z_nlp[nx_mesh + nu_col]
    t_nlp = tf_nlp*(1+col_points)/2

    if _adjoints_nlp is not None:
        # Costate information
        nu0_nlp = _adjoints_nlp[:nx]
        nuf_nlp = _adjoints_nlp[nx:nx + bcf_sym.numel()]

        lam_nlp = np.empty(shape=(nx, n_mesh))
        lam_nlp[:, idces_anchor] = np.nan
        lam_nlp[:, idces_collocation] = _adjoints_nlp[nx + bcf_sym.numel():].reshape((nx, -1), order='F') \
            * path_cost_scale / state_dynamics_scale[:, None]

        lam_nlp[:, 0] = -nu0_nlp / bc0_scale  # Import closure conditions
        lam_nlp[:3, -1] = nuf_nlp / bcf_scale
    else:
        nu0_nlp = np.empty_like(initial_state)
        nu0_nlp[:] = np.nan
        nuf_nlp = np.empty_like(terminal_pos)
        nuf_nlp[:] = np.nan
        lam_nlp = np.empty(shape=(nx, n_mesh))
        lam_nlp[:, idces_anchor] = np.nan

    # Append terminal condition
    t_nlp = np.append(t_nlp, tf_nlp)
    X_nlp = np.append(X_nlp, (X_nlp @ interp0f_mesh[1])[:, None], axis=-1)
    U_nlp = np.append(U_nlp, np.nan*U_nlp[:, -1:], axis=-1)
    lam_nlp = np.append(lam_nlp, np.nan*lam_nlp[:, -1:], axis=-1)

    # Compile solution
    _sol_nlp = giuseppe.data_classes.Solution()
    _sol_nlp.t = t_nlp
    _sol_nlp.x = X_nlp
    _sol_nlp.lam = lam_nlp
    _sol_nlp.u = U_nlp
    _sol_nlp.nu0 = nu0_nlp
    _sol_nlp.nuf = nuf_nlp
    return _sol_nlp


guess_nlp = unpack_solution(z_guess)
sol_nlp = unpack_solution(nlp_sol['x'].full().ravel(), nlp_sol['lam_g'].full().ravel())
with open('guess_nlp.data', 'wb') as f:
    pickle.dump(guess_nlp, f)
with open('sol_nlp.data', 'wb') as f:
    pickle.dump(sol_nlp, f)

# ----------------------------------------------------------------------------------------------------- #
# INDIRECT SOLUTION                                                                                     #
# ----------------------------------------------------------------------------------------------------- #
# Scalar signals -------------------------- #
hamiltonian_sym = path_cost_sym + ca.dot(costate_sym, eom_state_sym)

eom_costate_sym = -ca.jacobian(hamiltonian_sym, state_sym).T
control_law_sym = ca.jacobian(hamiltonian_sym, control_sym).T

hamiltonian_fun = ca.Function(
    'H', (state_sym, costate_sym, control_sym), (hamiltonian_sym,), ('x', 'lam', 'u'), ('H',)
)
eom_costate_fun = ca.Function(
    'nHx', (state_sym, costate_sym, control_sym), (eom_costate_sym,), ('x', 'lam', 'u'), ('nHx',)
)
control_law_fun = ca.Function(
    'Hu', (state_sym, costate_sym, control_sym), (control_law_sym,), ('x', 'lam', 'u'), ('Hu',)
)
# ----------------------------------------- #


# Update scales --------------------------- #
def gen_scales(_y, _tol: float = 1E-3):
    _y_max = _y.max(initial=-np.inf, axis=-1)
    _y_min = _y.min(initial=np.inf, axis=-1)
    return 0.5*(_y_max + _y_min), np.maximum(0.5*(_y_max - _y_min), _tol)


if use_state_scaling:
    state_bias, state_scale = gen_scales(sol_nlp.x)

    x0_nd = (initial_state - state_bias) / state_scale
    pf_nd = (terminal_pos - state_bias[:3]) / state_scale[:3]
if use_control_scaling:
    control_bias, control_scale = gen_scales(sol_nlp.u[:, 1:-1])
if use_costate_scaling:
    costate_bias, costate_scale = gen_scales(sol_nlp.lam[:, 1:-1])

    # Dynamics scaling
    hu_vals_nlp = control_law_fun(
        sol_nlp.x[:, idces_collocation], sol_nlp.lam[:, idces_collocation], sol_nlp.u[:, idces_collocation]
    ).full()
    dxdt_vals_nlp = eom_state_fun(
        sol_nlp.x[:, idces_collocation], sol_nlp.u[:, idces_collocation]
    ).full()
    dlamdt_vals_nlp = eom_costate_fun(
        sol_nlp.x[:, idces_collocation], sol_nlp.lam[:, idces_collocation], sol_nlp.u[:, idces_collocation]
    ).full()

    # Scaling for dynamics
    g_bias, g_scale = gen_scales(hu_vals_nlp)
    f_bias, f_scale = gen_scales(np.vstack((dxdt_vals_nlp, dlamdt_vals_nlp)))
else:
    costate_bias = np.zeros_like(initial_state)
    costate_scale = np.ones_like(initial_state)

    g_bias = np.zeros_like(control_lower_bound)
    g_scale = np.ones_like(g_bias)
    f_bias = np.zeros(shape=(2*nx,), dtype=initial_state.dtype)
    f_scale = np.ones_like(f_bias)

Lam_sym = ca.SX.sym('Lam', X_sym.shape)

# Update scaling
Xproj_sym = state_bias[:, None] + state_scale[:, None] * (X_sym @ proj_mat.T)
DXproj_sym = state_scale[:, None] * (X_sym @ proj_diff_mat.T)
Uproj_sym = control_bias[:, None] + control_scale[:, None] * (U_sym @ proj_col_mat.T)

Lamproj_sym = costate_bias[:, None] + costate_scale[:, None] * (X_sym @ proj_mat.T)
DLamproj_sym = costate_scale[:, None] * (X_sym @ proj_diff_mat.T)
DZproj_sym = ca.vcat((DXproj_sym, DLamproj_sym))

# The objective is the integrated error of the dynamics and algebraic signals
f_proj_sym = ca.vcat((
    eom_state_fun(Xproj_sym, Uproj_sym),
    eom_costate_fun(Xproj_sym, Lamproj_sym, Uproj_sym),
))
g_proj_sym = control_law_fun(Xproj_sym, Lamproj_sym, Uproj_sym)

dynamic_residual_sym = (
    tf_sym/2*f_proj_sym - DZproj_sym
) * np.tile(proj_weights[None, :], (2*nx, 1))
dynamic_residual_sym /= f_scale[:, None]
algebraic_residual = g_proj_sym * (proj_weights[None, :] / g_scale[:, None])

error_proj = ca.sum1(0.5 * dynamic_residual_sym ** 2) + ca.sum1(0.5 * algebraic_residual ** 2)
integrated_error = ca.sum2(error_proj)

# Boundary conditions
X0i_sym = X_sym @ interp0f_mesh[0]
Xfi_sym = X_sym @ interp0f_mesh[1]
Lamfi_sym = Lam_sym @ interp0f_mesh[1]

bc0_sym = X0i_sym - x0_nd  # Initial state fixed
bcf_sym = ca.vcat((Xfi_sym[:3] - pf_nd, Lamfi_sym[3:]))  # Initial pos fixed, term. vel free -> term. vel. costate = 0
boundary_constraints = ca.vcat((bc0_sym, bcf_sym))

z_sym = ca.vcat((
    ca.vec(X_sym),
    ca.vec(Lam_sym),
    ca.vec(U_sym),
    tf_sym,
))
if col_points[-1] == 1:
    # Terminal point IS collocated
    z_guess = np.concatenate((
        sol_nlp.x.ravel(order='F'),
        sol_nlp.lam.ravel(order='F'),
        sol_nlp.u[:, idces_collocation].ravel(order='F'),
        sol_nlp.t[-1:],
    ))
else:
    # Terminal point was extrapolated
    z_guess = np.concatenate((
        sol_nlp.x[:, :-1].ravel(order='F'),
        sol_nlp.lam[:, :-1].ravel(order='F'),
        sol_nlp.u[:, idces_collocation].ravel(order='F'),
        sol_nlp.t[-1:],
    ))

indirect_nlp = {
    'x': z_sym,  # Unknown variables
    'f': integrated_error,  # Objective function
    'g': boundary_constraints  # Equality constraints
}
indirect_nlp_solver = ca.nlpsol('INLP', 'ipopt', indirect_nlp)

# Bound guess with updated scales
lbx = (lb_state - state_bias) / state_scale
ubx = (ub_state - state_bias) / state_scale
lbu = (control_lower_bound - control_bias) / control_scale
ubu = (control_upper_bound - control_bias) / control_scale

# Allow 100% increase/decrease in costate bounds
lblam = np.empty_like(lbx)
lblam[:] = -2.
ublam = np.empty_like(ubx)
ublam[:] = 2.

lbz = np.empty_like(z_guess)
lbz[:nx_mesh] = np.tile(lbx, n_mesh)
lbz[nx_mesh:2*nx_mesh] = np.tile(lblam, n_mesh)
lbz[2*nx_mesh:-1] = np.tile(lbu, n_col)
lbz[-1] = lbtf
ubz = np.empty_like(z_guess)
ubz[:nx_mesh] = np.tile(ubx, n_mesh)
ubz[nx_mesh:2*nx_mesh] = np.tile(ublam, n_mesh)
ubz[2*nx_mesh:-1] = np.tile(ubu, n_col)
ubz[-1] = ubtf

indirect_nlp_sol = indirect_nlp_solver(x0=z_guess, lbg=0, ubg=0, lbx=lbz, ubx=ubz)

# Unpack indirect solution
z_indirect = indirect_nlp_sol['x'].full().ravel()
adjoints_indirect = indirect_nlp_sol['lam_g'].full().ravel()

X_indirect = np.empty_like(sol_nlp.x)
X_indirect[:] = np.nan
X_indirect[:, :-1] = z_indirect[:nx_mesh].reshape((nx, -1), order='F')
X_indirect[:, -1] = X_indirect[:, :-1] @ interp0f_mesh[1]
X_indirect = state_bias[:, None] + state_scale[:, None] * X_indirect
Lam_indirect = np.empty_like(sol_nlp.lam)
Lam_indirect[:] = np.nan
Lam_indirect[:, :-1] = z_indirect[nx_mesh:2*nx_mesh].reshape((nx, -1), order='F')
Lam_indirect[:, -1] = Lam_indirect[:, :-1] @ interp0f_mesh[1]
Lam_indirect = costate_bias[:, None] + costate_scale[:, None] * Lam_indirect
U_indirect = np.empty_like(sol_nlp.u)
U_indirect[:] = np.nan
U_indirect[:, idces_collocation] = z_indirect[2*nx_mesh:-1].reshape((nu, -1), order='F')
U_indirect = control_bias[:, None] + control_scale[:, None] * U_indirect
tf_indirect = z_indirect[-1]
t_indirect = np.empty_like(sol_nlp.t)
t_indirect[:-1] = tf_indirect * 0.5*(col_points + 1)
t_indirect[-1] = tf_indirect
sol_indirect = giuseppe.data_classes.Solution(
    t=t_indirect,
    x=X_indirect,
    lam=Lam_indirect,
    u=U_indirect,
    nu0=-Lam_indirect[:, 0],
    nuf=Lam_indirect[:3, -1],
)

with open('sol_indirect.data', 'wb') as f:
    pickle.dump(sol_indirect, f)

# z_sym = ca.vcat((
#     ca.vec(X_sym),
#     ca.vec(Lam_sym),
#     ca.vec(U_sym),
#     tf_sym,
# ))

# TODO - see if indirect method improves accuracy with same dimension
