import numpy as np
import pickle

import giuseppe

d2r = np.pi / 180
r2d = 180 / np.pi

# PROBLEM DEFINITION ---------------------------------------------------------------------------------------------------
hl20 = giuseppe.problems.symbolic.StrInputProb()

# Independent Variable
hl20.set_independent('t')

# Controls
hl20.add_control('alpha')

# States and Dynamics
hl20.add_state('h', 'v * sin(gam)')  # Altitude [m]
hl20.add_state('theta', 'v * cos(gam) / r')  # Downrange angle [rad]
hl20.add_state('v', '-drag/mass - g * sin(gam)')  # Velocity [m/s]
hl20.add_state('gam', 'lift / (mass * v) + (v/r - g/v) * cos(gam)')  # Flight path angle [rad]

# Expressions
hl20.add_expression('g', 'mu/r**2')  # Gravitational acceleration [m/s**2]
hl20.add_expression('drag', 'qdyn * s_ref * CD')  # Drag [N]
hl20.add_expression('lift', 'qdyn * s_ref * CL')  # Lift [N]
hl20.add_expression('CD', 'CD0 + CD1 * alpha + CD2 * alpha**2')  # Drag Coefficient [-]
hl20.add_expression('CL', 'CL0 + CL1 * alpha')  # Lift coefficient [-]
hl20.add_expression('rho', 'rho0 * exp(-h / h_ref)')  # Density [kg/m**3]
hl20.add_expression('qdyn', '0.5 * rho * v ** 2')  # Dynamic Pressure [N/m**2]
hl20.add_expression('r', 'rm + h')  # Radius [m]

# Constants
hl20.add_constant('mu', 42828.371901e9)  # Mar's Gravitational Parameter [m**3/s**2]
hl20.add_constant('rm', 3397.e3)  # Mar's radius [m]
hl20.add_constant('h_ref', 11.1e3)  # Density reference altitude [m]
hl20.add_constant('rho0', 0.02)  # Mars sea-level density

hl20.add_constant('CL0', -0.1232)
hl20.add_constant('CL1', 0.0368 / d2r)
hl20.add_constant('CD0', 0.075)
hl20.add_constant('CD1', -0.0029 / d2r)
hl20.add_constant('CD2', 5.5556e-4 / d2r**2)
hl20.add_constant('s_ref', 26.6)  # Reference area [m**2]
hl20.add_constant('mass', 11000.)  # Mass [kg]

# BOUNDARY CONDITIONS
# Initial Conditions
hl20.add_constant('h0', 80e3)
hl20.add_constant('theta0', 0.)
hl20.add_constant('v0', 4e3)
hl20.add_constant('gam0', -5 * d2r)

hl20.add_constraint('initial', 't')
hl20.add_constraint('initial', 'h - h0')
hl20.add_constraint('initial', 'theta - theta0')
hl20.add_constraint('initial', 'v - v0')
hl20.add_constraint('initial', 'gam - gam0')

# Terminal conditions
hl20.add_constant('hf', 10e3)
hl20.add_constant('thetaf', 1. * d2r)
hl20.add_constant('gamf', 0. * d2r)

hl20.add_constraint('terminal', 'h - hf')
hl20.add_constraint('terminal', 'theta - thetaf')
hl20.add_constraint('terminal', 'gam - gamf')

# CONSTRAINTS
# Control Constraint - alpha
alpha_reg_method = 'atan'
alpha_max = 30 * d2r
alpha_min = -alpha_max
eps_alpha = 1e-2

hl20.add_constant('alpha_max', alpha_max)
hl20.add_constant('eps_alpha', eps_alpha)
hl20.add_inequality_constraint(
    'control', 'alpha', '-alpha_max', 'alpha_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_alpha', method=alpha_reg_method)
)

# COST FUNCTIONAL
# Minimum terminal energy
# hl20.set_cost('-v', '0', 'v')  # Formulation with terminal cost: min{Vf - V0}
hl20.set_cost('0', '-drag/mass - g * sin(gam)', '0')  # Formulation with path cost: min{int(dV/dt)}

# COMPILATION ----------------------------------------------------------------------------------------------------------
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    hl20_dual = giuseppe.problems.symbolic.SymDual(hl20).compile(use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.SciPySolver(hl20_dual, verbose=False, max_nodes=100, node_buffer=10)

# SOLUTION -------------------------------------------------------------------------------------------------------------
# Generate convergent guess
alpha0 = 5 * d2r
if alpha_reg_method in ['trig', 'sin']:
    alpha_reg0 = np.asin(2/(alpha_max - alpha_min) * (alpha0 - 0.5*(alpha_max + alpha_min)))
elif alpha_reg_method in ['atan', 'arctan']:
    alpha_reg0 = eps_alpha * np.tan(0.5 * (2*alpha0 - alpha_max - alpha_min) * np.pi / (alpha_max - alpha_min))
else:
    alpha_reg0 = alpha0

guess = giuseppe.guess_generation.auto_propagate_guess(hl20_dual, control=alpha_reg0, t_span=30)

with open('guess_hl20.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('seed_sol_hl20.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Use continuations to achieve desired terminal conditions
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
# cont.add_linear_series(100, {'hf': 10e3, 'thetaf': 10 * d2r})
cont.add_logarithmic_series(100, {'eps_alpha': 5})
cont.add_linear_series(100, {'thetaf': 2.25*d2r})
sol_set = cont.run_continuation()

# Save Solution
sol_set.save('sol_set_hl20.data')
