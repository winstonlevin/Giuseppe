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

n = state_sym.shape[0]
m = control_sym.shape[0]

# Algebraic expressions
R_sym = h_sym + re
g_sym = g0 * (re/R_sym)**2
rho_sym = rho0 * ca.exp(-h_sym/h_ref)
wing_load_sym = (0.5*rho_sym*V_sym**2) * Sref / mass
drag_sym = wing_load_sym * (CD0 + CDi*alpha_sym**2)
lift_sym = wing_load_sym * (CLa * alpha_sym)
Vlat_sym = V_sym * ca.cos(gam_sym)

eom_state_sym = ca.vcat((
    V_sym * ca.sin(gam_sym),
    ((V_sym/R_sym) * ca.cos(gam_sym)) * ca.sin(psi_sym),
    ((V_sym/R_sym) * ca.cos(gam_sym)) * ca.cos(psi_sym)/ca.cos(lat_sym),
    -drag_sym - g_sym*ca.sin(gam_sym),
    (lift_sym*ca.cos(sig_sym) - (g_sym - V_sym**2/R_sym))/V_sym,
    (lift_sym*ca.sin(sig_sym) - Vlat_sym**2/R_sym * ca.cos(psi_sym)*ca.tan(lat_sym))/Vlat_sym
))

path_cost_sym = -eom_state_sym[0]  # Max Vf - V0

hamiltonian_sym = path_cost_sym + ca.dot(costate_sym, eom_state_sym)
eom_costate_sym = -ca.jacobian(hamiltonian_sym, state_sym).T
hu_sym = ca.jacobian(hamiltonian_sym, control_sym).T

# Differential algebraic system
eom_sym = ca.vcat((eom_state_sym, eom_costate_sym))
alg_sym = hu_sym

# Boundary conditions
# (initial)
bc0 = ca.vcat((
    h_sym - h0,
    lat_sym - lat0,
    lon_sym - lon0,
    V_sym - V0,
    gam_sym - gam0,
    psi_sym - psi0
))

# (terminal)
bcf = ca.vcat((
    h_sym - hf,
    lat_sym - latf,
    lon_sym - lonf,
    costate_sym[3],  # V, gam, psi Free -> lamf = 0
    costate_sym[4],  # (B/C path cost, not terminal cost formulation)
    costate_sym[5],
    hamiltonian_sym,  # Final time should make terminal Hamiltonian 0
))

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

hamiltonian_scale = g0  # L = dV/dt
costate_scale = hamiltonian_scale / state_dynamics_scale
costate_dynamics_scale = costate_scale / state_scale
control_law_scale = hamiltonian_scale / control_scale

# ----------------------------------------------------------------------------- #
# Discretization of continuous signals                                          #
# ----------------------------------------------------------------------------- #
n_int = 10  # Where the dynamic cost is numerically integrated
n_basis = 3  # Number of basis functions for state approximation

# Generate Basis Functions --------------------------------- #
legendre_polys_sym = [1., tau_sym]
if n_basis > 2:
    # Generate Legendre polynomials via recurrence relation
    for n in range(1, n_basis-1):
        legendre_polys_sym.append(
            ((2*n+1)*tau_sym*legendre_polys_sym[-1] - n*legendre_polys_sym[-2])/(n+1)
        )
else:
    legendre_polys_sym = legendre_polys_sym[:n_basis]

legendre_polys_sym = ca.vcat(legendre_polys_sym)

state_basis_matrix = ca.SX.sym('CX', n, n_basis)
costate_basis_matrix = ca.SX.sym('CLam', n, n_basis)
control_basis_matrix = ca.SX.sym('CU', m, n_basis)

state_signal_sym = state_basis_matrix @ legendre_polys_sym
costate_signal_sym = costate_basis_matrix @ legendre_polys_sym
control_signal_sym = control_basis_matrix @ legendre_polys_sym

# Integration points and weights
col_points, col_weights = giuseppe.utils.pseudospectral.lgl(n_int)

path_cost_signal_sym = path_cost_fun(
    state_signal_sym, control_signal_sym
)
state_dynamic_function_signal_sym = eom_state_fun(
    state_signal_sym, control_signal_sym
)
state_dynamic_signal_sym = ca.jacobian(state_dynamic_function_signal_sym, tau_sym)

adjoined_path_cost_signal_sym = path_cost_signal_sym + ca.dot(
    costate_signal_sym, state_dynamic_function_signal_sym - state_dynamic_signal_sym
)

# Turn signals into callable functions
adjoined_path_cost_signal_fun = ca.Function(
    'Ladj', (tau_sym, state_basis_matrix, costate_basis_matrix, control_basis_matrix),
    (adjoined_path_cost_signal_sym,),
    ('tau', 'Cx', 'CLam', 'Cu')
)


