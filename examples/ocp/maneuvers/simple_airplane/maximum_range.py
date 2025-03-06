import pickle
from copy import deepcopy

import math
import numpy as np
from scipy import optimize

import giuseppe

ocp = giuseppe.problems.symbolic.StrInputProb()

# -------------------------------------------------------------------------------------------------------------------- #
# PROBLEM SETUP                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
# Independent Variables
ocp.set_independent('t')

# Constants
# (from Airplane 1 at subsonic speeds:
# https://doi/org/10.2514/6.1968-877)
weight = 42_000  # lb
g = 32.2  # gravitational acceleration [ft/s2]
ocp.add_constant('W', weight)
ocp.add_constant('g', g)
ocp.add_expression('m', 'W/g')  # Mass [slug]

# Aerodynamic constants
Sref = 530.
CLa = 3.44
CD0 = 0.013
eta = 0.54
CL_max_ld = (CLa*CD0/eta)**0.5
ocp.add_constant('Sref', Sref)  # ft2
ocp.add_constant('CLa', CLa)  # 1/rad
ocp.add_constant('CD0', CD0)  # [-]
ocp.add_constant('eta', eta)  # [-]

# State and control variables
ocp.add_expression('h', 'hg - V*V/(2*g)')
ocp.add_state('hg', '-V*D/W')
ocp.add_state('V', '-D/m - g * sin(gam)')
ocp.add_state('gam', 'L/(m*V) - g/V*cos(gam)')

ocp.add_control('CL')
ocp.add_expression('Qdyn', '0.5*rho*V*V')
ocp.add_expression('L', 'Qdyn*Sref*CL')
ocp.add_expression('CD', 'CD0 + eta/CLa * CL*CL')
ocp.add_expression('D', 'Qdyn*Sref*CD')

# Atmosphere function
rho0 = 0.002378
h_ref = 23_800
ocp.add_expression('rho', 'rho0 * exp(-h/h_ref)')
ocp.add_constant('rho0', rho0)  # slug/ft3
ocp.add_constant('h_ref', h_ref)  # ft

# Cost
ocp.set_cost('0', '-V * cos(gam)', '0')

# Boundary conditions (initial)
ocp.add_constraint('initial', 't')
ocp.add_constant('hg0', 80_000.)
ocp.add_constraint('initial', 'hg - hg0')
ocp.add_constant('V0', 1_000.)
ocp.add_constraint('initial', 'V - V0')
ocp.add_constant('gam0', 0.)
ocp.add_constraint('initial', 'gam - gam0')

# (terminal)
hgf = 20_000.
ocp.add_constant('hgf', hgf)
ocp.add_constraint('terminal', 'hg - hgf')

# -------------------------------------------------------------------------------------------------------------------- #
# NUMERICAL SOLUTION TO FULL FIDELITY OCP                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
with giuseppe.utils.Timer('Setup Time: '):
    comp_dual = giuseppe.problems.SymDual(ocp, control_method='differential').compile(use_jit_compile=True)
    solver = giuseppe.numeric_solvers.SciPySolver(comp_dual, verbose=2)
    guess = giuseppe.guess_generation.auto_propagate_guess(comp_dual, control=(CL_max_ld,), t_span=10)

with giuseppe.utils.Timer('Solve Time: '):
    cont = giuseppe.continuation.ContinuationHandler(solver, guess)
    cont.add_linear_series(100, {'hgf': hgf})
    sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
sol = sol_set[-1]


# -------------------------------------------------------------------------------------------------------------------- #
# NUMERICAL SOLUTION TO OUTER SOLUTION                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
def outer_indirect_control(_v, _hg):
    # Control input
    _h = _hg - _v*_v/(2*g)
    _Qdyn = 0.5 * rho0 * math.exp(-_h/h_ref)
    _u = weight / (_Qdyn * Sref)  # enforces d(gam)/dt = 0

    # Outer costates
    _path_cost = -_v
    _cd = CD0 + eta/CLa * _u*_u
    _cdu = 2 * eta/CLa * _u
    _drag = _Qdyn * Sref * _cd
    _dhg_dt = -_v * _drag / weight
    _lam_hg = -_path_cost / _dhg_dt
    _lam_V = 0.  # enforces H_gam = 0.
    _lam_gam = _v * _cdu * (_v/g * _lam_hg + _lam_V)  # enforces Hu = 0
    _lam = np.array((_lam_hg, _lam_V, _lam_gam))

    return _u, _lam


def outer_residual(_v, _hg):
    # z vector contains V, lam_hg, lam_V, lam_gam, u
    _gam = 0.  # enforces d(h)/dt = 0 -> d(V)/dt = 0
    _x = np.array((_hg, _v, _gam))  # gamma = 0
    _u, _lam = outer_indirect_control(_v, _hg)

    # Residual
    _h_x = comp_dual.compute_costate_dynamics(0., _x, _lam, np.array((_u,)), sol.p, sol.k)
    return _h_x[1]  # enforces H_V = 0


sol_outer = deepcopy(sol)
sol_outer.x[2, :] = 0.  # Set gam = 0

# At first index, use "true" solution as initial guess
idx = 0
v0 = 870.
v1 = 880.
hg = sol_outer.x[0, idx]
sol_root = optimize.root_scalar(lambda _v: outer_residual(_v, hg), x0=v0, x1=v1)
sol_outer.x[1, idx] = sol_root.root
sol_outer.x[2, idx] = 0.
sol_outer.u[0, idx], sol_outer.lam[:, idx] = outer_indirect_control(sol_outer.x[1, 0], sol_outer.x[0, 0])
v0 = sol_outer.x[1, idx] - 10.
v1 = sol_outer.x[1, idx] + 10.

for idx, hg in enumerate(sol_outer.x[0, 1:], start=1):
    # Cycle through energy height values, using full fidelity values as initial guess
    sol_root = optimize.root_scalar(lambda _v: outer_residual(_v, hg), x0=v0, x1=v1)
    sol_outer.x[1, idx] = sol_root.root
    sol_outer.x[2, idx] = 0.
    sol_outer.u[0, idx], sol_outer.lam[:, idx] = outer_indirect_control(sol_outer.x[1, 0], sol_outer.x[0, 0])
    v0 = sol_outer.x[1, idx] - 10.
    v1 = sol_outer.x[1, idx] + 10.

with open('sol_outer.data', 'wb') as f:
    pickle.dump(sol_outer, f)
