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
hl20.add_state('h', 'vn')  # Altitude [m]
hl20.add_state('theta', 'vx / r')  # Downrange angle [rad]
hl20.add_state('vx', '-rho*v*s_ref/(2*mass) * (CD*vx + CL*vn) - vx*vn/r')  # Horizontal velocity [m/s]
hl20.add_state('vn', 'rho*v*s_ref/(2*mass)  * (CL*vx - CD*vn) + vx**2/r - g')  # Vertical velocity [m/s]

# Expressions
hl20.add_expression('v', '(vx**2 + vn**2) ** 0.5')
hl20.add_expression('qdyn', '0.5 * rho * v**2')
hl20.add_expression('lift', 'qdyn * s_ref * CL')
hl20.add_expression('drag', 'qdyn * s_ref * CD')
hl20.add_expression('weight0', 'mass * mu / rm**2')
hl20.add_expression('g', 'mu/r**2')  # Gravitational acceleration [m/s**2]
hl20.add_expression('CD', 'CD0 + CD1 * alpha + CD2 * alpha**2')  # Drag Coefficient [-]
hl20.add_expression('CL', 'CL0 + CL1 * alpha')  # Lift coefficient [-]
hl20.add_expression('rho', 'rho0 * exp(-h / h_ref)')  # Density [kg/m**3]
hl20.add_expression('r', 'rm + h')  # Radius [m]

# Constants
rm = 3397.e3
mu = 42828.371901e9
g0 = mu / rm**2

hl20.add_constant('mu', mu)  # Mar's Gravitational Parameter [m**3/s**2]
hl20.add_constant('rm', rm)  # Mar's radius [m]
hl20.add_constant('h_ref', 11.1e3)  # Density reference altitude [m]
hl20.add_constant('rho0', 0.02)  # Mars sea-level density

CL0 = -0.1232
CL1 = 0.0368 / d2r
CD0 = 0.075
CD1 = -0.0029 / d2r
CD2 = 5.5556e-4 / d2r**2
mass = 11000.

hl20.add_constant('CL0', CL0)
hl20.add_constant('CL1', CL1)
hl20.add_constant('CD0', CD0)
hl20.add_constant('CD1', CD1)
hl20.add_constant('CD2', CD2)
hl20.add_constant('s_ref', 26.6)  # Reference area [m**2]
hl20.add_constant('mass', mass)  # Mass [kg]

# BOUNDARY CONDITIONS
# Scaling Factors
hl20.add_constant('t_scale', 1.)
hl20.add_constant('h_scale', 1.)
hl20.add_constant('theta_scale', 1.)
hl20.add_constant('v_scale', 1.)
hl20.add_constant('alpha_scale', 1.)
# hl20.add_constant('h_scale', 1e4)
# hl20.add_constant('theta_scale', r2d)
# hl20.add_constant('v_scale', 1e4)
# hl20.add_constant('alpha_scale', 30 * r2d)

# Initial Conditions
h0 = 15e3
theta0 = 0.
v0 = 1.5e3
gam0 = -5*d2r
vx0 = v0 * np.cos(gam0)
vn0 = v0 * np.sin(gam0)

hl20.add_constant('h0', h0)
hl20.add_constant('theta0', theta0)
hl20.add_constant('vx0', vx0)
hl20.add_constant('vn0', vn0)

hl20.add_constraint('initial', 't/t_scale')
hl20.add_constraint('initial', '(h - h0)/h_scale')
hl20.add_constraint('initial', '(theta - theta0)/theta_scale')
hl20.add_constraint('initial', '(vx - vx0)/v_scale')
hl20.add_constraint('initial', '(vn - vn0)/v_scale')

# Terminal conditions
hl20.add_constant('hf', 10e3)
hl20.add_constant('thetaf', 1. * d2r)
hl20.add_constant('vxf', 10.)
hl20.add_constant('vnf', 0.)

hl20.add_constraint('terminal', '(h - hf)/h_scale')
hl20.add_constraint('terminal', '(vx - vxf)/v_scale')
hl20.add_constraint('terminal', '(vn - vnf)/v_scale')

# CONSTRAINTS
# G-Load Constraint
hl20.add_constant('n2_max', 4.5**2)
hl20.add_constant('n2_min', 0.)
hl20.add_expression('g_load2', '(lift**2 + drag**2) / weight0**2')

# Control Constraint - alpha
alpha_reg_method = 'sin'
alpha_max = 30 * d2r
alpha_min = -alpha_max
eps_alpha = 1e-3
hl20.add_constant('alpha_max', alpha_max)
hl20.add_constant('alpha_min', alpha_min)
hl20.add_constant('eps_alpha', eps_alpha)
hl20.add_inequality_constraint(
    'control', 'alpha', 'alpha_min', 'alpha_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_alpha', method=alpha_reg_method)
)

# Path Constraint - Heat Rate
hl20.add_constant('k', 1.9027e-4)  # Heat rate constant for Mars [kg**0.5 / m]
hl20.add_constant('rn', 1.)  # Nose radius [m]
hl20.add_constant('heat_rate_max', 500e4)  # Max heat rate [W/m**2]
hl20.add_expression('heat_rate', 'k * (rho / rn) * v ** 3')  # Heat Rate [W/m**2]

# Path Constraint - Altitude
hl20.add_constant('h_min', 5e3)
hl20.add_constant('h_max', 120e3)
hl20.add_constant('eps_h', 1e-3)
hl20.add_inequality_constraint(
    'path', 'h', 'h_min', 'h_max', regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler(
        'eps_h / h_ref', method='utm'
    )
)

# COST FUNCTIONAL
# Maximum range
hl20.set_cost('0', '-vx / r', '0')

# # Minimum range
# hl20.set_cost('0', 'vx / r', '0')

# COMPILATION ----------------------------------------------------------------------------------------------------------
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    hl20_dual = giuseppe.problems.symbolic.SymDual(hl20, control_method='differential').compile(use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.SciPySolver(hl20_dual, verbose=2, max_nodes=100, node_buffer=15)

# SOLUTION -------------------------------------------------------------------------------------------------------------
# Generate convergent guess
alpha_max_ld = - CL0/CL1 + ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5
alpha0 = alpha_max_ld
if alpha_reg_method in ['trig', 'sin']:
    alpha_reg0 = np.arcsin(2/(alpha_max - alpha_min) * (alpha0 - 0.5*(alpha_max + alpha_min)))
elif alpha_reg_method in ['atan', 'arctan']:
    alpha_reg0 = eps_alpha * np.tan(0.5 * (2*alpha0 - alpha_max - alpha_min) * np.pi / (alpha_max - alpha_min))
else:
    alpha_reg0 = alpha0

immutable_constants = (
    'mu', 'rm', 'h_ref', 'rho0',
    'CL0', 'CL1', 'CD0', 'CD1', 'CD2', 's_ref', 'mass',
    't_scale', 'h_scale', 'theta_scale', 'v_scale',
    'alpha_max', 'alpha_min', 'eps_alpha',
    'n2_max', 'n2_min',
)

guess = giuseppe.guess_generation.auto_propagate_guess(
    hl20_dual, control=alpha_reg0, t_span=40., immutable_constants=immutable_constants
)

with open('guess_hl20.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('seed_sol_hl20.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Use continuations to achieve desired terminal conditions
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'vnf': 0.})
cont.add_linear_series(100, {'hf': 10e3, 'vxf': 1.4e3})
# cont.add_logarithmic_series(100, {'eps_alpha': 1e-6, 'eps_h': 1e-5})
cont.add_linear_series_until_failure({'vxf': -10., 'hf': 10.**2/(2*g0)})
# cont.add_logarithmic_series(100, {'eps_alpha': 1e-6})
# cont.add_linear_series_until_failure({'hf': -100.})
sol_set = cont.run_continuation()

# Save Solution
sol_set.save('sol_set_hl20.data')
