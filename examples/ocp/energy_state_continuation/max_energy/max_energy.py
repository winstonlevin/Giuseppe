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
lonf = 5. * np.pi/180
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

# Scaling
state_bias, state_scale = np.array((
    ((hf + h0)/2,     abs(hf - h0)/2),      # h
    ((latf + lat0)/2, abs(latf - lat0)/2),  # Lat
    ((lonf + lon0)/2, abs(lonf - lon0)/2),  # Lon
    (V0/2,            V0/2),                # V
    (0.,              0.5*np.pi),           # gam
    (0.,              0.5*np.pi),           # psi
)).T
state_inv_scale = 1 / state_scale
V_scale = state_scale[3]

state_dynamics_bias, state_dynamics_scale = np.array((
    (0., V_scale),  # dh/dt
    (0., V_scale/re),  # dLat/dt
    (0., V_scale/re),  # dLon/dt
    (0., g0),  # dV/dt
    (0., g0/V_scale),  # dgam/dt
    (0., g0/V_scale),  # dpsi/dt
)).T
control_bias, control_scale = np.array((
    (0., 40.*np.pi/180),
    (0., np.pi),
)).T

path_cost_scale = g0  # L = dV/dt
costate_bias, costate_scale = np.zeros_like(state_dynamics_scale), 1 / state_dynamics_scale

bc0_scale = state_scale
bcf_scale = state_scale[:3]

x0_nd = (initial_state - state_bias) / state_scale
pf_nd = (terminal_pos - state_bias[:3]) / state_scale[:3]

# ----------------------------------------------------------------------------- #
# Discretization of continuous signals                                          #
# ----------------------------------------------------------------------------- #
n_col = 8  # Number of basis functions for state estimate (1 more than costate/control)
n_int = 30  # Number of integration locations


collocation_method = 'lg'

if collocation_method == 'lg':
    col_points, col_weights = giuseppe.utils.pseudospectral.lg(n_col + 1)
    # col_weights = np.insert(col_weights, 0, 0)
    idces_anchor = np.arange(0, 1, 1)
    idces_collocation = np.arange(1, n_col + 1, 1)

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
    proj_points = proj_points[1:]
elif collocation_method == 'lgr':
    col_points, col_weights = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points = np.append(col_points, 1.)
    idces_anchor = np.array((n_col,))
    idces_collocation = np.arange(0, n_col, 1)

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgr(n_int)
elif collocation_method == 'lgl':
    col_points, col_weights = giuseppe.utils.pseudospectral.lgl(n_col)
    idces_anchor = np.empty(shape=(0,), dtype=int)
    idces_collocation = np.arange(0, n_col, 1)

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgl(n_int)
elif collocation_method == 'zlg':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points, col_weights = giuseppe.utils.pseudospectral.lg(n_col+1)
    col_points = np.sort(np.append(col_points[1:], 0))
    idces_anchor = np.where(col_points == 0)[0]
    idces_collocation = np.delete(np.arange(0, n_col+1, 1), idces_anchor)

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
    proj_points = proj_points[1:]
elif collocation_method == 'zlgr':
    assert n_col % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
    col_points, col_weights = giuseppe.utils.pseudospectral.lgr(n_col)
    col_points = np.sort(np.append(col_points, 0))
    idces_anchor = np.where(col_points == 0)[0]
    idces_collocation = np.delete(np.arange(0, n_col+1, 1), idces_anchor)

    proj_points, proj_weights = giuseppe.utils.pseudospectral.lgr(n_int)
elif collocation_method == 'zlgl':
    assert n_col % 2 == 0, f"ZLGL requires an even number of collocation points, but n_col={n_col}!"
    col_points, col_weights = giuseppe.utils.pseudospectral.lgl(n_col)
    col_points = np.sort(np.append(col_points, 0))
    idces_anchor = np.where(col_points == 0)[0]
    idces_collocation = np.delete(np.arange(0, n_col+1, 1), idces_anchor)

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
interp0f_mesh_local, _ = giuseppe.utils.pseudospectral.lagrange_matrices(
    col_points, np.array((-1, +1)), compute_interp_matrix=True, compute_diff_matrix=False
)  # Interpolate Lam/U to get 0/f values

X_sym = ca.SX.sym('X', nx, n_mesh)  # Include initial state
U_sym = ca.SX.sym('U', nu, n_col)
nx_mesh = nx*n_mesh
nu_mesh = nu*n_col

Xproj_sym = state_bias[:, None] + state_scale[:, None] * (X_sym @ proj_mat.T)
Uproj_sym = control_bias[:, None] + control_scale[:, None] * (U_sym @ proj_col_mat.T)

tf_sym = ca.SX.sym('tf')
z_sym = ca.vcat((
    ca.vec(X_sym),
    ca.vec(U_sym),
    tf_sym,
))

L_proj = path_cost_fun(Xproj_sym, Uproj_sym) / path_cost_scale
f_proj = eom_state_fun(Xproj_sym, Uproj_sym)
X0i_sym = X_sym @ interp0f_mesh_local[0]
Xfi_sym = X_sym @ interp0f_mesh_local[1]

integrated_cost = tf_sym/2 * (L_proj @ proj_weights)
dynamic_residual = (
    tf_sym/2*f_proj - X_sym @ proj_diff_mat.T
) * np.tile(proj_weights[None, :], (nx, 1))
dynamic_residual /= state_dynamics_scale[:, None]
collocated_residual = dynamic_residual @ proj_col_mat
dynamic_constraint = ca.vec(collocated_residual)

bc0_sym = X0i_sym - x0_nd
bcf_sym = Xfi_sym[:3] - pf_nd
boundary_constraints = ca.vcat((bc0_sym, bcf_sym))

nlp = {
    'x': z_sym,  # Unknown variables
    'f': integrated_cost,  # Objective function
    'g': ca.vcat((boundary_constraints, dynamic_constraint))  # Equality constraints
}
nlp_solver = ca.nlpsol('NLP', 'ipopt', nlp)

# Initial guess ------------------------------------------- #
xnd0_guess = x0_nd
xndf_guess = np.empty_like(xnd0_guess)
xndf_guess[:3] = pf_nd
xndf_guess[3:] = x0_nd[3:]
xnd_guess = ((xnd0_guess + xndf_guess)/2)[:, None] + (xndf_guess - xnd0_guess)[:, None] * col_points[None, :]

und_guess = np.zeros(shape=U_sym.shape, dtype=xnd0_guess.dtype)

# For final time, guess based on boundary conditions and scales
tf_guess = 0.5*np.linalg.norm((terminal_pos - initial_state[:3]) / state_dynamics_scale[:3])

z_guess = np.concatenate((
    (tf_guess,),
    xnd_guess.ravel(order='F'),
    und_guess.ravel(order='F')
))

# TODO - set bounds based on state constraints
# Bounds -- Assuming the problem is well-scaled, we should have coefficients O(1)
# so I set bounds liberally at O(100). For tf, we know dE/dt < 0, so I set the initial value as its maximum
lbz = np.empty_like(z_guess)
lbz[0] = 0.  # tf
lbz[1:1+nx*n_col] = -1E3  # Coefficients of signals

ubz = np.empty_like(z_guess)
ubz[0] = 2*tf_guess  # tf
ubz[1:] = 1E3  # Coefficients of signals

nlp_sol = nlp_solver(x0=z_guess, lbg=0., ubg=0., lbx=lbz, ubx=ubz)

# Unpack solution
tf = nlp_sol['x'][0].full()[0, 0]
state_bases_cat = nlp_sol['x'][1:1+len(state_bases_cat_guess)].full().ravel()
control_bases_cat = nlp_sol['x'][1+len(state_bases_cat_guess):].full().ravel()
state_bases = []
idx0 = 0
for n in state_order:
    state_bases.append(state_bases_cat[idx0:idx0+n])
    idx0 += n
control_bases = []
idx0 = 0
for n in control_order:
    control_bases.append(control_bases_cat[idx0:idx0+n])
    idx0 += n

nu0 = nlp_sol['lam_g'][:bc0_sym.shape[0]].full().ravel()
nuf = nlp_sol['lam_g'][bc0_sym.shape[0]:bc0_sym.shape[0]+bcf_sym.shape[0]].full().ravel()
costate_bases_cat = nlp_sol['lam_g'][bc0_sym.shape[0]+bcf_sym.shape[0]:].full().ravel()
costate_bases = []
idx0 = 0
for n in costate_order:
    costate_bases.append(costate_bases_cat[idx0:idx0+n])
    idx0 += n

# Save solution
x_poly_guess = [np.polynomial.legendre.Legendre(_b) for _b in state_bases_guess]
u_poly_guess = [np.polynomial.legendre.Legendre(_b) for _b in control_bases_guess]

x_poly_sol = [np.polynomial.legendre.Legendre(_b) for _b in state_bases]
u_poly_sol = [np.polynomial.legendre.Legendre(_b) for _b in control_bases]
lam_poly_sol = [np.polynomial.legendre.Legendre(_b) for _b in costate_bases]

sol_dict = {
    'tf_guess': tf_guess,
    'x_guess': x_poly_guess,
    'u_guess': u_poly_guess,

    'tf': tf,
    'x': x_poly_sol,
    'u': u_poly_sol,
    'lam': lam_poly_sol,
    'nu0': nu0,
    'nuf': nuf,

    'xb': state_bias,
    'xr': state_scale,
    'ub': control_bias,
    'ur': control_scale,
    'lamb': costate_bias,
    'lamr': np.ones_like(costate_scale),
}

with open('sol_nlp.data', 'wb') as f:
    pickle.dump(sol_dict, f)
