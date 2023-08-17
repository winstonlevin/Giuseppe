import numpy as np
from scipy.optimize import fsolve
import pickle

import giuseppe

OPTIMIZATION = 'min_time'  # {'max_range', 'min_time', 'min_velocity', 'max_drag', 'min_heat', 'min_control'}
OPT_ERROR_MSG = 'Invalid Optimization Option!'

d2r = np.pi / 180
r2d = 180 / np.pi

# PROBLEM DEFINITION ---------------------------------------------------------------------------------------------------
hl20 = giuseppe.problems.symbolic.StrInputProb()

# Independent Variable
hl20.set_independent('t')

# Controls
hl20.add_control('alpha_rate')

# States and Dynamics
hl20.add_state('h_nd', '( v * sin(gam) ) / h_scale')  # Altitude [m], nd [-]
hl20.add_state('theta_nd', '( v * cos(gam) / r ) / theta_scale')  # Downrange angle [rad], nd [-]
hl20.add_state('v_nd', '( -drag/mass - g * sin(gam) ) / v_scale')  # Velocity [m/s], nd [-]
hl20.add_state('gam_nd', '( lift / (mass * v) + (v/r - g/v) * cos(gam) ) / gam_scale')  # Flight path ang. [rad], nd [-]
hl20.add_state('alpha_nd', 'alpha_rate / alpha_scale')  # Angle of attack [rad], nd [-]

hl20.add_expression('h', 'h_nd * h_scale')
hl20.add_expression('theta', 'theta_nd * theta_scale')
hl20.add_expression('v', 'v_nd * v_scale')
hl20.add_expression('gam', 'gam_nd * gam_scale')
hl20.add_expression('alpha', 'alpha_nd * alpha_scale')

# Expressions
hl20.add_expression('g', 'mu/r**2')  # Gravitational acceleration [m/s**2]
hl20.add_expression('g0', 'mu/rm**2')  # Gravitational acceleration [m/s**2]
hl20.add_expression('weight0', 'mass * g0')  # Gravitational acceleration [m/s**2]
hl20.add_expression('drag', 'qdyn * s_ref * CD')  # Drag [N]
hl20.add_expression('lift', 'qdyn * s_ref * CL')  # Lift [N]
hl20.add_expression('rho', 'rho0 * exp(-h / h_ref)')  # Density [kg/m**3]
hl20.add_expression('qdyn', '0.5 * rho * v ** 2')  # Dynamic Pressure [N/m**2]
hl20.add_expression('r', 'rm + h')  # Radius [m]

# # Polynomial CL/CD Model
# hl20.add_expression('CD', 'CD0 + CD1 * alpha + CD2 * alpha**2')  # Drag Coefficient [-]
# hl20.add_expression('CL', 'CL0 + CL1 * alpha')  # Lift coefficient [-]

# Trigonometric CL/CD Model
hl20.add_expression('CD', 'CD0 - CD1**2/(4*CD2) + CD2 * sin(alpha + CD1/(2*CD2))**2')  # Drag Coefficient [-]
hl20.add_expression('CL', 'CL1 * 0.5 * sin(2 * (alpha + CL0/CL1))')  # Lift coefficient [-]

# Constants
mu = 42828.371901e9
rm = 3397.e3
h_ref = 11.1e3
rho0 = 0.02
hl20.add_constant('mu', mu)  # Mar's Gravitational Parameter [m**3/s**2]
hl20.add_constant('rm', rm)  # Mar's radius [m]
hl20.add_constant('h_ref', h_ref)  # Density reference altitude [m]
hl20.add_constant('rho0', rho0)  # Mars sea-level density

CL0 = -0.1232
CL1 = 0.0368 / d2r
CD0 = 0.075
CD1 = -0.0029 / d2r
CD2 = 5.5556e-4 / d2r**2
s_ref = 26.6
mass = 11e3

hl20.add_constant('CL0', CL0)
hl20.add_constant('CL1', CL1)
hl20.add_constant('CD0', CD0)
hl20.add_constant('CD1', CD1)
hl20.add_constant('CD2', CD2)
hl20.add_constant('s_ref', s_ref)  # Reference area [m**2]
hl20.add_constant('mass', mass)  # Mass [kg]

# BOUNDARY CONDITIONS
# Scaling Factors
# h_scale = 1.
# theta_scale = 1.
# v_scale = 1.
# gam_scale = 1.
# alpha_scale = 1.

# h_scale = 1e4
# theta_scale = r2d
# v_scale = 1e4
# gam_scale = 5 * r2d
# alpha_scale = 30 * r2d

h_scale = 1e4
theta_scale = d2r
v_scale = 1e3
if OPTIMIZATION == 'min_time':
    gam_scale = 90 * d2r
else:
    gam_scale = 5 * d2r
alpha_scale = 30 * d2r

hl20.add_constant('h_scale', h_scale)
hl20.add_constant('theta_scale', theta_scale)
hl20.add_constant('v_scale', v_scale)
hl20.add_constant('gam_scale', gam_scale)
hl20.add_constant('alpha_scale', alpha_scale)

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
hl20.add_constant('vf', 10.)

hl20.add_constraint('terminal', 'h - hf')
# hl20.add_constraint('terminal', 'theta - thetaf')
hl20.add_constraint('terminal', 'gam - gamf')

# CONSTRAINTS
# G-Load Constraint (Total G's)
n2_max = 4.5**2
hl20.add_constant('n2_max', n2_max)
# hl20.add_constant('n2_min', 0.)
hl20.add_constant('n2_min', -n2_max)
hl20.add_constant('eps_n2', 1e-3)
hl20.add_expression('g_load2', '(lift**2 + drag**2) / weight0**2')  # sea-level g's of acceleration [-]
if OPTIMIZATION == 'min_time':
    hl20.add_inequality_constraint(
        'path', 'g_load2', 'n2_min', 'n2_max',
        regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler('eps_n2', method='utm')
    )

# Control Constraint - alpha rate
alpha_rate_reg_method = 'sin'
alpha_rate_max = 1. * 2 * np.pi  # 1 hz limit
alpha_rate_min = -alpha_rate_max
hl20.add_constant('alpha_rate_max', alpha_rate_max)
hl20.add_constant('alpha_rate_min', alpha_rate_min)
hl20.add_constant('eps_alpha_rate', 1e-3)
hl20.add_inequality_constraint(
    'control', 'alpha_rate', 'alpha_rate_min', 'alpha_rate_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler(
        'eps_alpha_rate', method=alpha_rate_reg_method
    )
)

# Control Constraint - alpha
# alpha_reg_method = 'sin'
# alpha_max = 30 * d2r
# alpha_min = -alpha_max
# eps_alpha = 1e-3
# hl20.add_constant('alpha_max', alpha_max)
# hl20.add_constant('alpha_min', alpha_min)
# hl20.add_constant('eps_alpha', eps_alpha)
# hl20.add_inequality_constraint(
#     'control', 'alpha', 'alpha_min', 'alpha_max',
#     regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_alpha', method=alpha_reg_method)
# )

alpha_reg_method = None
alpha_max = 90 * d2r
alpha_min = -alpha_max
eps_alpha = 0
hl20.add_constant('alpha_max', alpha_max)
hl20.add_constant('alpha_min', alpha_min)
hl20.add_constant('eps_alpha', eps_alpha)
# hl20.add_inequality_constraint(
#     'control', 'alpha', 'alpha_min', 'alpha_max',
#     regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_alpha', method=alpha_reg_method)
# )

# Path Constraint - Heat Rate
hl20.add_constant('k', 1.9027e-4)  # Heat rate constant for Mars [kg**0.5 / m]
hl20.add_constant('rn', 1.)  # Nose radius [m]
hl20.add_constant('heat_rate_max', 500e4)  # Max heat rate [W/m**2]
hl20.add_expression('heat_rate', 'k * (rho / rn) * v ** 3')  # Heat Rate [W/m**2]

# Path Constraint - Altitude
h_min = 5e3
h_max = 120e3
hl20.add_constant('h_min', h_min)
hl20.add_constant('h_max', h_max)
hl20.add_constant('eps_h', 1e-3)
if OPTIMIZATION != 'min_time':
    hl20.add_inequality_constraint(
        'path', 'h_nd', 'h_min / h_scale', 'h_max / h_scale',
        regularizer=giuseppe.problems.symbolic.regularization.PenaltyConstraintHandler(
            'eps_h', method='utm'
        )
    )

# COST FUNCTIONAL
if OPTIMIZATION == 'max_range':
    # Maximum range (path cost = -d(theta)/dt -> min J = theta0 - thetaf)
    hl20.set_cost('0', '(-v * cos(gam) / r) / theta_scale', '0')
    hl20.add_constraint('terminal', 'v - vf')  # Constrain final velocity
elif OPTIMIZATION == 'min_velocity':
    # Minimum velocity (path cost = d(V)/dt -> min J = vf - v0)
    hl20.set_cost('0', '(-drag/mass - g * sin(gam)) / v_scale', '0')
elif OPTIMIZATION == 'min_time':
    # Minimum time (path cost = 1 -> min J = tf - t0)
    hl20.set_cost('0', '1', '0')
    # hl20.add_constraint('terminal', 'v - vf')  # Constrain final velocity
elif OPTIMIZATION == 'max_drag':
    # Maximum drag (path cost = -drag -> min J = -int(drag)
    hl20.set_cost('0', '-drag / (rho0 * v_scale**2)', '0')
    hl20.add_constraint('terminal', 'v - vf')  # Constrain final velocity
elif OPTIMIZATION == 'min_heat':
    hl20.set_cost('0', 'heat_rate / (k * (rho0 / rn) * v_scale ** 3)', '0')
    hl20.add_constraint('terminal', 'v - vf')  # Constrain final velocity
elif OPTIMIZATION == 'min_control':
    hl20.set_cost('0', '(alpha / alpha_scale)**2', '0')
    hl20.add_constraint('terminal', 'v - vf')  # Constrain final velocity
else:
    raise RuntimeError(OPT_ERROR_MSG)

# COMPILATION ----------------------------------------------------------------------------------------------------------
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    hl20_dual = giuseppe.problems.symbolic.SymDual(hl20, control_method='differential').compile(use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.SciPySolver(hl20_dual, verbose=2, max_nodes=100, node_buffer=15)

# GLIDE SLOPE
# The glide slope occurs for a given energy level at the combination of h/V where:
# (1) d(gam)/dt = gam = 0
# (2) max(L/D) or min(L/D) [for max/min range]
# The glide slope will be used to run continuations
alpha_max_ld = - CL0/CL1 + ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5
alpha_min_ld = - CL0/CL1 - ((CL0**2 + CD0*CL1**2 - CD1*CL0*CL1)/(CD2*CL1**2)) ** 0.5

CL_glide = CL0 + CL1 * alpha_max_ld
CD_glide = CD0 + CD1 * alpha_max_ld + CD2 * alpha_max_ld ** 2


def glide_slope_velocity_fpa(_h):
    _rho = rho0 * np.exp(-_h / h_ref)
    _r = rm + _h
    _g = mu / _r ** 2
    _v = (mass * _g / (0.5 * _rho * s_ref * CL_glide + mass / _r)) ** 0.5
    _qdyn = 0.5 * _rho * _v**2
    _drag = _qdyn * s_ref * CD_glide
    _rSCL = _r * s_ref * CL_glide
    _gam = - np.arcsin(
        (_rho * _drag/mass + 2*_drag / _rSCL)
        / (_qdyn/h_ref + _rho * _g + 2 * mass * _g / _rSCL)
    )
    return _v, _gam


# SOLUTION -------------------------------------------------------------------------------------------------------------
# Desired initial conditions
h0_1 = 80e3
v0_1, gam0_1 = glide_slope_velocity_fpa(h0_1)

# Generate convergent guess
if alpha_rate_reg_method in ['trig', 'sin']:
    def ctrl2reg(_alpha_rate):
        return np.arcsin(2/(alpha_rate_max - alpha_rate_min) * (_alpha_rate - 0.5*(alpha_rate_max + alpha_rate_min)))
elif alpha_rate_reg_method in ['atan', 'arctan']:
    def ctrl2reg(_alpha_rate):
        return eps_alpha * np.tan(
            0.5 * (2*_alpha_rate - alpha_rate_max - alpha_rate_min) * np.pi / (alpha_rate_max - alpha_rate_min)
        )
else:
    def ctrl2reg(_alpha_rate):
        return _alpha_rate

immutable_constants = (
    'mu', 'rm', 'h_ref', 'rho0',
    'CL0', 'CL1', 'CD0', 'CD1', 'CD2', 's_ref', 'mass',
    'h_scale', 'theta_scale', 'v_scale', 'gam_scale',
    'alpha_max', 'alpha_min', 'eps_alpha',
    'alpha_rate_max', 'alpha_rate_min', 'eps_alpha_rate',
    'n2_max', 'n2_min', 'eps_n2'
)

alphaf = 29.5 * d2r
hf = 10e3
gf = mu / (rm + hf) ** 2
weightf = gf * mass
rhof = rho0 * np.exp(-hf/h_ref)
CLf = CL0 + CL1 * alphaf
vf = (2 * mass * gf / (rhof * s_ref * CLf)) ** 0.5

if OPTIMIZATION == 'min_control':
    guess = giuseppe.guess_generation.auto_propagate_guess(
        hl20_dual, control=ctrl2reg(0.), t_span=np.arange(0., 25., 5.), immutable_constants=immutable_constants,
        initial_states=np.array((h0_1/h_scale, 0./theta_scale, v0_1/v_scale, gam0_1/gam_scale, 0./alpha_scale)),
        fit_states=False, reverse=False
    )
elif OPTIMIZATION == 'min_time':
    guess = giuseppe.guess_generation.auto_propagate_guess(
        hl20_dual, control=ctrl2reg(0.), t_span=np.arange(0., 20., 5.),
        immutable_constants=immutable_constants,
        initial_states=np.array((5e3/h_scale, 0./theta_scale, 1e3/v_scale, 0./gam_scale, alphaf/alpha_scale)),
        fit_states=False, reverse=True
    )
else:
    guess = giuseppe.guess_generation.auto_propagate_guess(
        hl20_dual, control=ctrl2reg(0.), t_span=np.linspace(0., 25., 6), immutable_constants=immutable_constants,
        initial_states=np.array((hf/h_scale, 0./theta_scale, vf/v_scale, 0 * d2r/gam_scale, alphaf/alpha_scale)),
        fit_states=False, reverse=True
    )

with open('guess_hl20.data', 'wb') as file:
    pickle.dump(guess, file)

# Seed Solution from guess
seed_sol = num_solver.solve(guess)

with open('seed_sol_hl20.data', 'wb') as file:
    pickle.dump(seed_sol, file)

# Continuation Series from Glide Slope
idx_h0 = seed_sol.annotations.constants.index('h0')
idx_v0 = seed_sol.annotations.constants.index('v0')
idx_gam0 = seed_sol.annotations.constants.index('gam0')
idx_gam_scale = seed_sol.annotations.constants.index('gam_scale')
idx_eps_alpha_rate = seed_sol.annotations.constants.index('eps_alpha_rate')

k = guess.k.copy()
k[idx_gam_scale] = 90 * d2r
k[idx_eps_alpha_rate] = 1e-2
guess = giuseppe.guess_generation.auto_propagate_guess(
    hl20_dual, control=ctrl2reg(0.), t_span=np.arange(0., 20., 5.), k=k,
    immutable_constants=immutable_constants,
    initial_states=np.array((5e3 / h_scale, 0. / theta_scale, 1e3 / v_scale, 0. / k[idx_gam_scale], alphaf / alpha_scale)),
    fit_states=False, reverse=True
)
seed_sol = num_solver.solve(guess)

# Extend solution via glide slope
h0_0 = seed_sol.x[0, 0] * h_scale
v0_glide_seed, gam0_glide_seed = glide_slope_velocity_fpa(h0_0)


def glide_slope_continuation(previous_sol, frac_complete):
    _h0 = h0_0 + frac_complete * (h0_1 - h0_0)
    _v0, _gam0 = glide_slope_velocity_fpa(_h0)
    previous_sol.k[idx_h0] = _h0
    previous_sol.k[idx_v0] = _v0
    previous_sol.k[idx_gam0] = _gam0
    return previous_sol.k


# Use continuations to achieve desired terminal conditions
if OPTIMIZATION == 'max_range':
    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_linear_series(1, {'theta0': 0.})
    cont.add_logarithmic_series(100, {'eps_h': 1e-5})
    cont.add_linear_series(10, {'v0': v0_glide_seed, 'gam0': gam0_glide_seed})
    cont.add_linear_series_until_failure({'vf': -10})
    cont.add_custom_series(100, glide_slope_continuation, series_name='GlideSlope')
    cont.add_logarithmic_series(100, {'eps_h': 1e-10})
    # cont.add_logarithmic_series(100, {'eps_alpha': 1e-10, 'eps_h': 1e-10})
    sol_set = cont.run_continuation()
elif OPTIMIZATION == 'min_time':
    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_linear_series(1, {'theta0': 0.})
    cont.add_linear_series(10, {'v0': v0_glide_seed, 'gam0': gam0_glide_seed})
    cont.add_logarithmic_series(100, {'eps_n2': 1e0})
    cont.add_custom_series(100, glide_slope_continuation, series_name='GlideSlope')
    cont.add_logarithmic_series(100, {'eps_n2': 1e-10})
    sol_set = cont.run_continuation()
elif OPTIMIZATION == 'max_drag':
    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_linear_series(1, {'theta0': 0.})
    cont.add_linear_series(10, {'v0': v0_glide_seed, 'gam0': gam0_glide_seed})
    cont.add_linear_series_until_failure({'vf': -10})
    cont.add_custom_series(100, glide_slope_continuation, series_name='GlideSlope')
    cont.add_logarithmic_series(100, {'eps_alpha': 1e-10, 'eps_h': 1e-10})
    sol_set = cont.run_continuation()
elif OPTIMIZATION == 'min_heat':
    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_linear_series(1, {'theta0': 0.})
    cont.add_logarithmic_series(100, {'eps_h': 1e-5})
    cont.add_linear_series(25, {'v0': v0_glide_seed, 'gam0': gam0_glide_seed})
    # cont.add_linear_series(25, {'gam0': gam0_glide_seed})
    # cont.add_linear_series_until_failure({'vf': -10})
    # cont.add_linear_series(100, {'v0': v0_1, 'h0': h0_1, 'gam0': gam0_1})
    #
    # cont.add_custom_series(100, glide_slope_continuation, series_name='GlideSlope')
    # cont.add_logarithmic_series(100, {'eps_h': 1e-10})
    sol_set = cont.run_continuation()
elif OPTIMIZATION == 'min_control':
    cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
    cont.add_linear_series(1, {'theta0': 0.})
    cont.add_linear_series_until_failure({'vf': -10})
    cont.add_linear_series(10, {'v0': v0_glide_seed, 'gam0': gam0_glide_seed})
    cont.add_custom_series(100, glide_slope_continuation, series_name='GlideSlope')
    cont.add_logarithmic_series(100, {'eps_h': 1e-10})
    # cont.add_logarithmic_series(100, {'eps_alpha': 1e-10, 'eps_h': 1e-10})
    sol_set = cont.run_continuation()
else:
    raise RuntimeError(OPT_ERROR_MSG)

# Save Solution
sol_set.save('sol_set_hl20.data')
