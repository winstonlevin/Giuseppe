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
rm = 3397.e3
hl20.add_constant('mu', 42828.371901e9)  # Mar's Gravitational Parameter [m**3/s**2]
hl20.add_constant('rm', rm)  # Mar's radius [m]
hl20.add_constant('h_ref', 11.1e3)  # Density reference altitude [m]
hl20.add_constant('rho0', 0.02)  # Mars sea-level density

CL0 = -0.1232
CL1 = 0.0368 / d2r
CD0 = 0.075
CD1 = -0.0029 / d2r
CD2 = 5.5556e-4 / d2r**2

hl20.add_constant('CL0', CL0)
hl20.add_constant('CL1', CL1)
hl20.add_constant('CD0', CD0)
hl20.add_constant('CD1', CD1)
hl20.add_constant('CD2', CD2)
hl20.add_constant('s_ref', 26.6)  # Reference area [m**2]
hl20.add_constant('mass', 11000.)  # Mass [kg]

# BOUNDARY CONDITIONS
# Scaling Factors
hl20.add_constant('t_scale', 1.)
hl20.add_constant('h_scale', 1e4)
hl20.add_constant('theta_scale', r2d)
hl20.add_constant('v_scale', 1e4)
hl20.add_constant('gam_scale', 5 * r2d)
hl20.add_constant('alpha_scale', 30 * r2d)

# Initial Conditions
hl20.add_constant('h0', 80e3)
hl20.add_constant('theta0', 0.)
hl20.add_constant('v0', 4e3)
hl20.add_constant('gam0', -5 * d2r)

hl20.add_constraint('initial', 't/t_scale')
hl20.add_constraint('initial', '(h - h0)/h_scale')
hl20.add_constraint('initial', '(theta - theta0)/theta_scale')
hl20.add_constraint('initial', '(v - v0)/v_scale')
hl20.add_constraint('initial', '(gam - gam0)/gam_scale')

# Terminal conditions
hl20.add_constant('hf', 10e3)
hl20.add_constant('thetaf', 1. * d2r)
hl20.add_constant('gamf', 0. * d2r)

hl20.add_constraint('terminal', '(h - hf)/h_scale')
hl20.add_constraint('terminal', '(theta - thetaf)/theta_scale')
hl20.add_constraint('terminal', '(gam - gamf)/gam_scale')

# CONSTRAINTS
# alpha limit due to G-Load
hl20.add_constant('n_max', 4.5)
hl20.add_constant('n_min', -4.5)
hl20.add_expression('alpha_n_max', '(mass * g) * n_max / (qdyn * s_ref * CL1) - CL0 / CL1')
hl20.add_expression('alpha_n_min', '(mass * g) * n_min / (qdyn * s_ref * CL1) - CL0 / CL1')

# Control Constraint - alpha
alpha_reg_method = 'sin'
alpha_max = 30 * d2r
alpha_min = -alpha_max
eps_alpha = 1e-3
hl20.add_constant('alpha_max', alpha_max)
hl20.add_constant('alpha_min', alpha_min)

# Smoothed combination of alpha limits due to alpha min/max + G-load constraint
hl20.add_constant('k_lse', 10)  # Higher -> closer to discontinuous limit
# hl20.add_expression('alpha_upper_limit_smooth', '-log(exp(k_lse * -alpha_max) + exp(k_lse * -alpha_n_max)) / k_lse')
# hl20.add_expression('alpha_lower_limit_smooth', ' log(exp(k_lse *  alpha_min) + exp(k_lse *  alpha_n_min)) / k_lse')
hl20.add_expression('alpha_upper_limit', 'minimum(alpha_max, alpha_n_max)')
hl20.add_expression('alpha_lower_limit', 'maximum(alpha_min, alpha_n_min)')
hl20.add_expression(
    'alpha_upper_limit_smooth',
    'alpha_upper_limit - ' +
    'log(exp(k_lse * -(alpha_max - alpha_upper_limit)) + exp(k_lse * -(alpha_n_max - alpha_upper_limit))) / k_lse'
)
hl20.add_expression(
    'alpha_lower_limit_smooth',
    'alpha_lower_limit + ' +
    'log(exp(k_lse * (alpha_min - alpha_lower_limit)) + exp(k_lse * (alpha_n_min - alpha_lower_limit))) / k_lse')


hl20.add_constant('eps_alpha', eps_alpha)
hl20.add_inequality_constraint(
    'control', 'alpha', 'alpha_lower_limit_smooth', 'alpha_upper_limit_smooth',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_alpha', method=alpha_reg_method)
)

# Path Constraint - Heat Rate
hl20.add_constant('k', 1.9027e-4)  # Heat rate constant for Mars [kg**0.5 / m]
hl20.add_constant('rn', 1.)  # Nose radius [m]
hl20.add_constant('heat_rate_max', 500e4)  # Max heat rate [W/m**2]
hl20.add_expression('heat_rate', 'k * (rho / rn) * v ** 3')  # Heat Rate [W/m**2]

# COST FUNCTIONAL
# Minimum terminal energy
hl20.add_constant('k_cost_v', 1)
hl20.add_constant('k_cost_alpha', 0.)
# hl20.set_cost(
#     '-v/v_scale * k_cost_v',
#     'k_cost_alpha * (alpha / alpha_scale)**2',
#     'v/v_scale * k_cost_v'
# )  # Formulation with terminal cost: min{Vf - V0}

hl20.set_cost(
    '0',
    '(-drag/mass - g * sin(gam))**2/v_scale**2 * k_cost_v + k_cost_alpha * (alpha / alpha_scale)**2',
    '0')  # Formulation with path cost: min{int(dV/dt)}

# COMPILATION ----------------------------------------------------------------------------------------------------------
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    hl20_dual = giuseppe.problems.symbolic.SymDual(hl20, control_method='differential').compile(use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.SciPySolver(hl20_dual, verbose=False, max_nodes=100, node_buffer=15)

# SOLUTION -------------------------------------------------------------------------------------------------------------
# Generate convergent guess
alpha0 = 29.5 * d2r
if alpha_reg_method in ['trig', 'sin']:
    alpha_reg0 = np.arcsin(2/(alpha_max - alpha_min) * (alpha0 - 0.5*(alpha_max + alpha_min)))
elif alpha_reg_method in ['atan', 'arctan']:
    alpha_reg0 = eps_alpha * np.tan(0.5 * (2*alpha0 - alpha_max - alpha_min) * np.pi / (alpha_max - alpha_min))
else:
    alpha_reg0 = alpha0

immutable_constants = (
    'mu', 'rm', 'h_ref', 'rho0',
    'CL0', 'CL1', 'CD0', 'CD1', 'CD2', 's_ref', 'mass',
    't_scale', 'h_scale', 'theta_scale', 'v_scale', 'gam_scale',
    'alpha_max', 'alpha_min', 'eps_alpha', 'k_lse', 'n_max', 'n_min',
    'k_cost_v', 'k_cost_alpha'
)

guess = giuseppe.guess_generation.auto_propagate_guess(
    hl20_dual, control=alpha_reg0, t_span=130., immutable_constants=immutable_constants
)

with open('guess_hl20.data', 'wb') as file:
    pickle.dump(guess, file)

seed_sol = num_solver.solve(guess)

with open('seed_sol_hl20.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Use continuations to achieve desired terminal conditions
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'gamf': 0.})
cont.add_linear_series(100, {'hf': 10e3, 'thetaf': 17 * d2r})
cont.add_logarithmic_series(100, {'eps_alpha': 1e-6})
cont.add_linear_series_until_failure({'thetaf': -0.1 * d2r})
sol_set = cont.run_continuation()

# Save Solution
sol_set.save('sol_set_hl20.data')
