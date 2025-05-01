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
def generate_residual(x_order: list, lam_order: list, u_order: list, n_int: int = 10, integration_method: str ='lgl'):
    # Generate integrator (ti, wi)
    if integration_method == 'lg':
        proj_points, proj_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
        proj_points = proj_points[1:]
    elif integration_method == 'lgr':
        proj_points, proj_weights = giuseppe.utils.pseudospectral.lgr(n_int)
    elif integration_method == 'lgl':
        proj_points, proj_weights = giuseppe.utils.pseudospectral.lgl(n_int)
    elif integration_method == 'zlg':
        assert n_int % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
        proj_points, proj_weights = giuseppe.utils.pseudospectral.lg(n_int+1)
        proj_points = proj_points[1:]
    elif integration_method == 'zlgr':
        assert n_int % 2 == 0, f"ZLG requires an even number of collocation points, but n_col={n_col}!"
        proj_points, proj_weights = giuseppe.utils.pseudospectral.lgr(n_int)
    elif integration_method == 'zlgl':
        assert n_int % 2 == 0, f"ZLGL requires an even number of collocation points, but n_col={n_col}!"
        proj_points, proj_weights = giuseppe.utils.pseudospectral.lgl(n_int)
    else:
        raise ValueError(f'integration_method=={integration_method} is not implemented!')

    # Generate basis functions --------------------------------------------------------------------------------------- #
    max_order = np.concatenate((x_order, lam_order, u_order)).max(initial=0)

    # Legendre basis
    bases = [np.polynomial.Legendre.basis(_o) for _o in range(max_order + 1)]

    # Evaluation matrices associated with integrator
    proj_mat = np.vstack([_b(proj_points) for _b in bases]).T
    proj_diff_mat = np.vstack([_b.deriv()(proj_points) for _b in bases]).T
    interp0f_mesh = np.vstack([_b(np.array((-1, +1.))) for _b in bases]).T

    # Coefficient matrices associated with signals
    X_sym = ca.SX.zeros(nx, max_order)
    Lam_sym = ca.SX.zeros(nx, max_order)
    U_sym = ca.SX.zeros(nu, max_order)

    # TODO - generate parameter vector based on order, finish implementation of indirect method using IPOPT to optimize
    # TODO - residual

# TODO - solve using NLP solution as initial guess

# TODO - develop adaptation of state orders
