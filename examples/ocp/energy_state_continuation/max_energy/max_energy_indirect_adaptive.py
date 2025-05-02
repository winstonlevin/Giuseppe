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

# State Dynamics
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

# Costate Dynamics
hamiltonian_sym = path_cost_sym + ca.dot(costate_sym, eom_state_sym)
eom_costate_sym = -ca.jacobian(hamiltonian_sym, state_sym).T
control_law_sym = ca.jacobian(hamiltonian_sym, control_sym).T

# Functions
path_cost_fun = ca.Function('L', (state_sym, control_sym), (path_cost_sym,), ('x', 'u'), ('L',))
eom_state_fun = ca.Function('f', (state_sym, control_sym), (eom_state_sym,), ('x', 'u'), ('f',))
eom_costate_fun = ca.Function(
    'fLam', (state_sym, costate_sym, control_sym), (eom_costate_sym,), ('x', 'lam', 'u'), ('fLam',)
)
hamiltonian_function = ca.Function(
    'H', (state_sym, costate_sym, control_sym), (hamiltonian_sym,), ('x', 'lam', 'u'), ('H',)
)
control_law_function = ca.Function(
    'Hu', (state_sym, costate_sym, control_sym), (control_law_sym,), ('x', 'lam', 'u'), ('Hu',)
)

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
control_lower_bound = np.array((-40*np.pi, -3*np.pi))  # Make AoA>=0 to disambiguate sign of AoA/Bank
control_upper_bound = np.array((40*np.pi/180, 3*np.pi))

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

    # scale(H) = scale(lam) * scale(f) -> scale(lam) = scale(H) / scale(f)
    costate_bias = np.zeros_like(state_bias)
    costate_scale = path_cost_scale / state_dynamics_scale

    # d(lam)/dt = -Hx -> scale(d(lam)/dt) = scale(H) / scale(x)
    costate_dynamics_bias = np.zeros_like(costate_bias)
    costate_dynamics_scale = path_cost_scale / state_scale

    # g = Hu -> scale(g) = scale(H) / scale(u)
    control_function_bias = np.zeros_like(control_scale)
    control_function_scale = path_cost_scale / control_scale
else:
    state_dynamics_bias = np.zeros_like(initial_state)
    state_dynamics_scale = np.ones_like(initial_state)
    path_cost_scale = 1.

    # scale(H) = scale(lam) * scale(f) -> scale(lam) = scale(H) / scale(f)
    costate_bias = np.zeros_like(state_bias)
    costate_scale = np.ones_like(costate_bias)

    # d(lam)/dt = -Hx -> scale(d(lam)/dt) = scale(H) / scale(x)
    costate_dynamics_bias = np.zeros_like(costate_bias)
    costate_dynamics_scale = np.ones_like(costate_dynamics_bias)

    # g = Hu -> scale(g) = scale(H) / scale(u)
    control_function_bias = np.zeros_like(control_scale)
    control_function_scale = np.ones_like(control_function_bias)

bc0_scale = state_scale
bcf_scale = state_scale[:3]

x0_nd = (initial_state - state_bias) / state_scale
pf_nd = (terminal_pos - state_bias[:3]) / state_scale[:3]


# ----------------------------------------------------------------------------- #
# Discretization of continuous signals                                          #
# ----------------------------------------------------------------------------- #
# Legendre basis
bases = []


def update_legendre_order(max_order):
    if len(bases) < max_order:
        bases.extend([np.polynomial.Legendre.basis(_o) for _o in range(len(bases), max_order)])


def eval_legendre(eval_points, deriv: int = 0):
    if deriv == 0:
        return np.vstack([_b(eval_points) for _b in bases]).T
    else:
        return np.vstack([_b.deriv(deriv)(eval_points) for _b in bases]).T


def coefficients_from_samples(tau, x, x_bias=None, x_scale=None):
    # Non-dimensionalize x
    xnd = x.copy()
    if x_bias is not None:
        xnd -= x_bias[:, None]
    if x_scale is not None:
        xnd /= x_scale[:, None]

    # Generate projection matrix
    order_x = xnd.shape[1]
    update_legendre_order(order_x)
    X = eval_legendre(tau)
    return np.linalg.solve(X.T[:order_x], xnd.T).T


def integration_values(n_int: int = 10, integration_method: str ='lgl'):
    # Generate integrator (ti, wi)
    if integration_method == 'lg':
        int_points, int_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
        int_points = int_points[1:]
    elif integration_method == 'lgr':
        int_points, int_weights = giuseppe.utils.pseudospectral.lgr(n_int)
    elif integration_method == 'lgl':
        int_points, int_weights = giuseppe.utils.pseudospectral.lgl(n_int)
    elif integration_method == 'zlg':
        assert n_int % 2 == 0, f"ZLG requires an even number of collocation points, but n_int={n_int}!"
        int_points, int_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
        int_points = int_points[1:]
    elif integration_method == 'zlgr':
        assert n_int % 2 == 0, f"ZLG requires an even number of collocation points, but n_int={n_int}!"
        int_points, int_weights = giuseppe.utils.pseudospectral.lgr(n_int)
    elif integration_method == 'zlgl':
        assert n_int % 2 == 0, f"ZLGL requires an even number of collocation points, but n_int={n_int}!"
        int_points, int_weights = giuseppe.utils.pseudospectral.lgl(n_int)
    else:
        raise ValueError(f'integration_method=={integration_method} is not implemented!')

    return int_points, int_weights


def solve_indirect(
        guess, x_order: list, lam_order: list, u_order: list, int_points, int_weights
):
    # Generate basis functions --------------------------------------------------------------------------------------- #
    n_int = int_points.size
    max_order = np.concatenate((x_order, lam_order, u_order)).max(initial=0)
    assert n_int >= max_order, "The order of the integrator must be at least as high as the highest order signal!"
    update_legendre_order(max_order)

    # Evaluation matrices associated with integrator
    proj_mat = eval_legendre(int_points)
    proj_diff_mat = eval_legendre(int_points, deriv=1)
    interp0f_mesh = eval_legendre(np.array((-1, +1.)))

    # Coefficient matrices associated with signals
    X_sym = ca.SX.zeros(nx, max_order)
    Lam_sym = ca.SX.zeros(nx, max_order)
    U_sym = ca.SX.zeros(nu, max_order)

    x_params = [ca.SX.sym('C' + _x.name(), order) for (_x, order) in zip(ca.vertsplit(state_sym), x_order)]
    lam_params = [ca.SX.sym('C' + _x.name(), order) for (_x, order) in zip(ca.vertsplit(costate_sym), lam_order)]
    u_params = [ca.SX.sym('C' + _x.name(), order) for (_x, order) in zip(ca.vertsplit(control_sym), u_order)]

    for idx, x_param in enumerate(x_params):
        X_sym[idx, :x_param.numel()] = x_param.T
    for idx, lam_param in enumerate(lam_params):
        Lam_sym[idx, :lam_param.numel()] = lam_param.T
    for idx, u_param in enumerate(u_params):
        U_sym[idx, :u_param.numel()] = u_param.T

    Xproj_sym = state_bias[:, None] + state_scale[:, None] * (X_sym @ proj_mat.T)
    DXproj_sym = state_scale[:, None] * (X_sym @ proj_diff_mat.T)
    Lamproj_sym = costate_bias[:, None] + costate_scale[:, None] * (Lam_sym @ proj_mat.T)
    DLamproj_sym = costate_scale[:, None] * (Lam_sym @ proj_diff_mat.T)
    Uproj_sym = control_bias[:, None] + control_scale[:, None] * (U_sym @ proj_mat.T)

    tf_sym = ca.SX.sym('tf')
    z_sym = ca.vcat((ca.vcat(x_params), ca.vcat(lam_params), ca.vcat(u_params), tf_sym))

    # Discretized dynamic residuals
    f_proj_sym = eom_state_fun(Xproj_sym, Uproj_sym)
    flam_proj_sym = eom_costate_fun(Xproj_sym, Lamproj_sym, Uproj_sym)
    X0ind_sym = X_sym @ interp0f_mesh[0]
    Xfind_sym = X_sym @ interp0f_mesh[1]
    Xfi_sym = state_bias[:, None] + state_scale[:, None] * Xfind_sym
    Lamfi_sym = costate_bias[:, None] + costate_scale[:, None] * (Lam_sym @ interp0f_mesh[1])
    Ufi_sym = control_bias[:, None] + control_scale[:, None] * (U_sym @ interp0f_mesh[1])

    dyn_res_x = (tf_sym/2 * f_proj_sym - DXproj_sym) * np.tile(int_weights[None, :], (nx, 1))
    dyn_res_x /= state_dynamics_scale[:, None]
    dyn_res_lam = (tf_sym/2 * flam_proj_sym - DLamproj_sym) * np.tile(int_weights[None, :], (nx, 1))
    dyn_res_lam /= costate_dynamics_scale[:, None]

    # Discretized algebraic residual
    alg_res_u = control_law_function(Xproj_sym, Lamproj_sym, Uproj_sym) / control_function_scale[:, None]

    # Total residual
    residual = ca.vcat((ca.vec(dyn_res_x), ca.vec(dyn_res_lam), ca.vec(alg_res_u)))
    cost_res = 0.5 * ca.dot(residual, residual)

    # Boundary constraints
    bc0_sym = X0ind_sym - x0_nd
    bcf_sym = Xfind_sym[:3] - pf_nd
    bcf_adj_sym = Lamfi_sym[3:]  # Vf free -> LamVf = 0
    bcf_t_sym = hamiltonian_function(Xfi_sym, Lamfi_sym, Ufi_sym)
    boundary_constraints = ca.vcat((bc0_sym, bcf_sym, bcf_adj_sym, bcf_t_sym))

    # Define NLP
    nlp = {
        'x': z_sym,  # Unknown variables
        'f': cost_res,  # Objective function
        'g': boundary_constraints  # Equality constraints
    }
    nlp_solver = ca.nlpsol('NLP', 'ipopt', nlp)

    # Initial guess and bounds --------------------------------------------- #
    # Pad guesses with 0 coeff for increased order
    cx_guesses = guess['x']
    for cx_guess, cx_order in zip(cx_guesses, x_order):
        n_pad = cx_order - len(cx_guess)
        cx_guess.extend(0. for _ in range(n_pad))

    clam_guesses = guess['lam']
    for clam_guess, clam_order in zip(clam_guesses, lam_order):
        n_pad = clam_order - len(clam_guess)
        clam_guess.extend(0. for _ in range(n_pad))

    cu_guesses = guess['u']
    for cu_guess, cu_order in zip(cu_guesses, u_order):
        n_pad = cu_order - len(cu_guess)
        cu_guess.extend(0. for _ in range(n_pad))

    tf_guess = guess['tf']

    z_guess = np.concatenate((*cx_guesses, *clam_guesses, *cu_guesses, (tf_guess,)))

    # Bounds (permit coefficients to change by +- 10)
    lbz = z_guess.copy()
    lbz[:-1] -= 10
    lbz[-1] = tf_guess/3

    ubz = z_guess.copy()
    ubz[:-1] += 10
    ubz[-1] = tf_guess*3

    return nlp_solver(x0=z_guess, lbg=0, ubg=0, lbx=lbz, ubx=ubz), int_points


def unpack_sol(_z_nlp, x_order, lam_order, u_order, int_points):
    # Unpack NLP coefficients ------------------------- #
    n_bases = len(bases)
    idx0 = 0  # Iterate through _z_nlp elements
    Xnd_nlp = np.zeros(shape=(nx, n_bases))
    for _c, _order in zip(Xnd_nlp, x_order):
        _c[:_order] = _z_nlp[idx0:idx0+_order]
        idx0 += _order
    Lamnd_nlp = np.zeros(shape=(nx, n_bases))
    for _c, _order in zip(Lamnd_nlp, lam_order):
        _c[:_order] = _z_nlp[idx0:idx0+_order]
        idx0 += _order
    Und_nlp = np.zeros(shape=(nu, n_bases))
    for _c, _order in zip(Und_nlp, u_order):
        _c[:_order] = _z_nlp[idx0:idx0+_order]
        idx0 += _order
    tf_nlp = _z_nlp[idx0]

    # Project values onto interpolation points --------------------------- #
    # Ensure initial/terminal conditions are interpolated
    if int_points[0] != -1:
        int_points = np.concatenate(((-1.,), int_points))
    if int_points[-1] != 1:
        int_points = np.append(int_points, +1.)
    proj_mat = eval_legendre(int_points)

    t_nlp = tf_nlp * 0.5 * (1. + int_points)
    X_nlp = state_bias[:, None] + state_scale[:, None] * (Xnd_nlp @ proj_mat.T)
    lam_nlp = costate_bias[:, None] + costate_scale[:, None] * (Lamnd_nlp @ proj_mat.T)
    U_nlp = control_bias[:, None] + control_scale[:, None] * (Und_nlp @ proj_mat.T)

    # Enforce closure conditions:
    nu0_nlp = -lam_nlp[:, 0]  # nu0 + lam0 = 0
    nuf_nlp = lam_nlp[:3, -1]  # nuf - lamPf = 0

    # Save coefficient info
    coef_info = np.concatenate((x_order, lam_order, u_order, _z_nlp[:-1]))

    return giuseppe.data_classes.Solution(
        t=t_nlp,
        x=X_nlp,
        lam=lam_nlp,
        u=U_nlp,
        nu0=nu0_nlp,
        nuf=nuf_nlp,
        k=coef_info
    )


# TODO - solve using NLP solution as initial guess
def load_sol(file_name):
    with open(file_name, 'rb') as f:
        sol = pickle.load(f)
    return sol


try:
    sol_nlp = load_sol('sol_nlp.data')
except FileNotFoundError:
    from max_energy import sol_nlp

x_order = np.sum(sol_nlp.k > 0)  # In NLP sol, all are same order
u_order = np.sum(sol_nlp.k > 1)
lam_order = np.sum(u_order)

n_int = np.max([x_order, u_order, lam_order])

x_order = [x_order for i in range(nx)]
u_order = [u_order for i in range(nu)]
lam_order = [lam_order for i in range(nx)]

# Legendre basis
if len(bases) < n_int:
    bases.extend([np.polynomial.Legendre.basis(_o) for _o in range(len(bases), n_int)])

# Coefficients from x samples
t_vals = sol_nlp.t
t0, tf = t_vals[0], t_vals[-1]
tau_vals = 0.5*(t0 + tf) + 0.5*(tf - t0)*t_vals

# Compute as arrays
coeffs_x = coefficients_from_samples(tau_vals[sol_nlp.k > 0], sol_nlp.x[:, sol_nlp.k > 0], state_bias, state_scale)
coeffs_lam = coefficients_from_samples(tau_vals[sol_nlp.k > 1], sol_nlp.lam[:, sol_nlp.k > 1], costate_bias, costate_scale)
coeffs_u = coefficients_from_samples(tau_vals[sol_nlp.k > 1], sol_nlp.u[:, sol_nlp.k > 1], control_bias, control_scale)

# Convert to list of lists b/c in general not square
coeffs_x = [list(_cx) for _cx in coeffs_x]
coeffs_lam = [list(_cx) for _cx in coeffs_lam]
coeffs_u = [list(_cx) for _cx in coeffs_u]

guess = {
    'x': coeffs_x,
    'u': coeffs_u,
    'lam': coeffs_lam,
    'tf': tf
}

int_points, int_weights = integration_values(n_int, 'lgl')
nlp_sol = solve_indirect(guess, x_order, lam_order, u_order, int_points, int_weights)
sol_indirect = unpack_sol(nlp_sol['x'].full().ravel(), x_order, lam_order, u_order, int_points)

with open('sol_indirect.data', 'wb') as f:
    pickle.dump(sol_indirect, f)

# TODO - develop adaptation of state orders
