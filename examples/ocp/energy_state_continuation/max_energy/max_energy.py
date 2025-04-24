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
# (from https://doi/org/10.2514/6.1968-877)
mass = 340.1943  # kg
g0 = 9.80665  # free fall acceleration [m/s2]
re = 6_378_000  # Earth's radius [m]

ocp.add_constant('m', mass)
ocp.add_constant('g0', g0)
ocp.add_constant('re', re)
ocp.add_expression('r', 're + h')
ocp.add_expression('g', 'g0 * (re/r)**2')

# Aerodynamic constants
Sref = 0.2919  # m2
CLa = 1.5658  # [-]
CD0 = 0.0612  # [-]
CDi = 1.6537  # [-]
CL_max_ld = CLa * (CD0/CDi)**0.5
ocp.add_constant('Sref', Sref)
ocp.add_constant('CLa', CLa)  # 1/rad
ocp.add_constant('CD0', CD0)  # [-]
ocp.add_constant('CDi', CDi)  # [-]

# State and control variables
ocp.add_control('CL')
ocp.add_control('sig')
ocp.add_expression('Qdyn', '0.5*rho*V*V')
ocp.add_expression('L', 'Qdyn*Sref*CL')
ocp.add_expression('CD', 'CD0 + CDi/CLa**2 * CL*CL')
ocp.add_expression('D', 'Qdyn*Sref*CD')

# Atmosphere function
rho0 = 1.2  # Sea-level density [kg/m3]
h_ref = 7_500  # Density decay rate [m]
ocp.add_expression('rho', 'rho0 * exp(-h/h_ref)')
ocp.add_constant('rho0', rho0)  # slug/ft3
ocp.add_constant('h_ref', h_ref)  # ft

# Dynamics
ocp.add_state('h', 'V*sin(gam)')
ocp.add_state('lon', 'V/r * cos(gam)*cos(psi)/cos(lat)')
ocp.add_state('lat', 'V/r * cos(gam)*sin(psi)')
ocp.add_state('V', '-D/m - g * sin(gam)')
ocp.add_state('gam', 'L*cos(sig)/(m*V) + (V/r - g/V) * cos(gam)')
ocp.add_state('psi', 'L*sin(sig)/(m*V) - V/r * cos(gam) * cos(psi) * tan(lat)')

# Cost (Max energy)
ocp.set_cost('0', '-V*D/m', '0')

# Boundary conditions (initial)
ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'lon')
ocp.add_constraint('initial', 'lat')
V0 = 2e3  # [m/s]
h0 = 40e3  # [m]
gam0 = 0.  # [rad]
psi0 = 0.  # [rad]
ocp.add_constant('h0', h0)
ocp.add_constraint('initial', 'h - h0')
ocp.add_constant('V0', V0)
ocp.add_constraint('initial', 'V - V0')
ocp.add_constant('gam0', gam0)
ocp.add_constraint('initial', 'gam - gam0')
ocp.add_constant('psi0', psi0)
ocp.add_constraint('initial', 'psi - psi0')

# (terminal)
hf = 0.
lonf = 5. * np.pi/180
latf = 1. * np.pi/180
ocp.add_constant('hf', hf)
ocp.add_constraint('terminal', 'h - hf')
ocp.add_constant('lonf', lonf)
ocp.add_constraint('terminal', 'h - hf')
ocp.add_constant('latf', latf)
ocp.add_constraint('terminal', 'h - hf')

# -------------------------------------------------------------------------------------------------------------------- #
# NUMERICAL SOLUTION TO FULL FIDELITY OCP                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
with giuseppe.utils.Timer('Setup Time: '):
    comp_dual = giuseppe.problems.SymDual(ocp, control_method='differential').compile(use_jit_compile=True)
    solver = giuseppe.numeric_solvers.SciPySolver(comp_dual, verbose=2)
    guess = giuseppe.guess_generation.auto_propagate_guess(comp_dual, control=(CL_max_ld, 0.), t_span=30)

with giuseppe.utils.Timer('Solve Time: '):
    cont = giuseppe.continuation.ContinuationHandler(solver, guess)
    # cont.add_linear_series(100, {'hEf': hEf})
    sol_set = cont.run_continuation()
