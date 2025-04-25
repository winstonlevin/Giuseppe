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

n_state = state_sym.shape[0]
m_control = control_sym.shape[0]

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
path_cost_sym = -eom_state_sym[0]  # Max Vf - V0

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

# ----------------------------------------------------------------------------- #
# Discretization of continuous signals                                          #
# ----------------------------------------------------------------------------- #
n_int = 10  # Where the dynamic cost is numerically integrated
n_basis_max = 3  # Number of basis functions for state approximation

state_order = np.empty(shape=initial_state.shape, dtype=int)
state_order[:3] = n_basis_max  # 2 bc -> highest order
state_order[3:] = n_basis_max - 1  # Only 1 bc -> let costate order be 1 higher
costate_order = state_order - 1  # Eliminate 1 order for each specified boundary condition
costate_order[:3] -= 1
control_order = np.empty(shape=(m_control,), dtype=int)
control_order[:] = state_order[3] - 1

# Generate Basis Functions and Discretize Signals ------------------------------------------- #
legendre_polys_sym = [1., tau_sym]
if n_basis_max > 2:
    # Generate Legendre polynomials via recurrence relation
    for n in range(1, n_basis_max-1):
        legendre_polys_sym.append(
            ((2*n+1)*tau_sym*legendre_polys_sym[-1] - n*legendre_polys_sym[-2])/(n+1)
        )
else:
    legendre_polys_sym = legendre_polys_sym[:n_basis_max]

legendre_polys_sym = ca.vcat(legendre_polys_sym)

state_bases = [ca.SX.sym('C' + _x.name(), _nb) for (_x, _nb) in zip(ca.vertsplit(state_sym), state_order)]
costate_bases = [ca.SX.sym('C' + _x.name(), _nb) for (_x, _nb) in zip(ca.vertsplit(costate_sym), costate_order)]
control_bases = [ca.SX.sym('C' + _x.name(), _nb) for (_x, _nb) in zip(ca.vertsplit(control_sym), costate_order)]

state_bases_cat = ca.vcat(state_bases)
control_bases_cat = ca.vcat(control_bases)

state_signal_sym = ca.vcat([ca.dot(_c, legendre_polys_sym[:_c.shape[0]]) for _c in state_bases])
control_signal_sym = ca.vcat([ca.dot(_c, legendre_polys_sym[:_c.shape[0]]) for _c in control_bases])

# Integration points and weights
col_points, col_weights = giuseppe.utils.pseudospectral.lgl(n_int)

path_cost_signal_sym = path_cost_fun(
    state_signal_sym, control_signal_sym
) / path_cost_scale
state_dynamic_function_signal_sym = eom_state_fun(
    state_signal_sym, control_signal_sym
)
state_dynamic_signal_sym = ca.jacobian(state_dynamic_function_signal_sym, tau_sym)
dynamic_residual_signal_sym = ca.vec(legendre_polys_sym @ (
        (tf_sym/2)*state_dynamic_function_signal_sym - state_dynamic_signal_sym
).T)  # TODO - change to order of costates


bc0_signal_sym = ca.substitute(bc0_fun(state_signal_sym), tau_sym, -1.) / bc0_scale
bcf_signal_sym = ca.substitute(bcf_fun(state_signal_sym), tau_sym, +1.) / bcf_scale

# Turn signals into callable functions
path_cost_signal_fun = ca.Function(
    'Ladj', (tau_sym, state_bases_cat, control_bases_cat),
    (path_cost_signal_sym,),
    ('tau', 'Cx', 'Cu'), ('Ladj',)
)
dynamic_residual_signal_fun = ca.Function(
    'fres', (tau_sym, state_bases_cat, control_bases_cat),
    (dynamic_residual_signal_sym,),
    ('tau', 'Cx', 'Cu'), ('fres',)
)

integral_cost = tf_sym * ca.sum1(ca.vcat([
    _wi * path_cost_signal_fun(_taui, state_bases_cat, control_bases_cat)
    for (_taui, _wi) in zip(col_points, col_weights*0.5)
]))
dyn_res_sym = ca.sum2(dynamic_residual_signal_fun(col_points[None, :], state_bases_cat, control_bases_cat))

z_sym = ca.vcat((tf_sym, state_bases_cat, control_bases_cat))

nlp = {
    'x': z_sym,  # Unknown variables
    'f': integral_cost,  # Objective function
    'g': ca.vcat((bc0_signal_sym, bcf_signal_sym, dyn_res_sym))  # Equality constraints
}
nlp_solver = ca.nlpsol('NLP', 'ipopt', nlp)

# Initial guess ------------------------------------------- #
# The first two basis functions are:
# [1, x]
# So I set states as:
# C0 = x0       [For initial state constraint]
# C1 = xf - x0  [For terminal state constraint]
# Ci = 0        [Otherwise]
state_basis_matrix_guess = np.zeros(shape=state_basis_matrix.shape, dtype=float)
state_basis_matrix_guess[:, 0] = (initial_state - state_bias)/state_scale
state_basis_matrix_guess[:3, 1] = (terminal_pos - initial_state[:3] - state_bias[:3])/state_scale[:3]
control_basis_matrix_guess = np.zeros(shape=control_basis_matrix.shape, dtype=float)

# For final time, guess based on boundary conditions and scales
tf_guess = 0.5*np.linalg.norm((terminal_pos - initial_state[:3]) / state_dynamics_scale[:3])

z_guess = np.concatenate((
    (tf_guess,),
    state_basis_matrix_guess.ravel(order='F'),
    control_basis_matrix_guess.ravel(order='F')
))

# Bounds -- Assuming the problem is well-scaled, we should have coefficients O(1)
# so I set bounds liberally at O(100). For tf, we know dE/dt < 0, so I set the initial value as its maximum
lbz = np.empty_like(z_guess)
lbz[0] = 0.  # tf
lbz[1:] = -3.  # Coefficients of signals

ubz = np.empty_like(z_guess)
ubz[0] = 2*tf_guess  # tf
ubz[1:] = 3.  # Coefficients of signals

nlp_sol = nlp_solver(x0=z_guess, lbg=0., ubg=0., lbx=lbz, ubx=ubz)
