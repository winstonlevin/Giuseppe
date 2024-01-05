# Re-orient a space craft using body-fI1ed torques as the control input. Dynamics use quaternions and body-fI1ed angular
# velocities assuming principal moment of inertia. The control law is explicitly derived using UTM.
# Original source: https://doi.org/10.1515/astro-2019-0011
from copy import deepcopy
import numpy as np
import pickle

import giuseppe

from scipy.integrate import solve_ivp

scorient = giuseppe.problems.input.StrInputProb()
scorient.set_independent('t')

# State Dynamics
I1 = 1.
I2 = 1.
I3 = 1.

scorient.add_constant('I1', I1)
scorient.add_constant('I2', I2)
scorient.add_constant('I3', I3)

scorient.add_expression('dq1dt', '(w1*q4 - w2*q3 + w3*q2)/2')
scorient.add_expression('dq2dt', '(w1*q3 + w2*q4 - w3*q1)/2')
scorient.add_expression('dq3dt', '(-w1*q2 + w2*q1 + w3*q4)/2')
scorient.add_expression('dq4dt', '(-w1*q1 - w2*q2 - w3*q3)/2')
scorient.add_expression('dw1dt', 'u1/I1 + ((I2 - I3)/I1)*w2*w3')
scorient.add_expression('dw2dt', 'u2/I2 + ((I3 - I1)/I2)*w1*w3')
scorient.add_expression('dw3dt', 'u3/I3 + ((I1 - I2)/I3)*w1*w2')

scorient.add_state('q1', 'dq1dt')
scorient.add_state('q2', 'dq2dt')
scorient.add_state('q3', 'dq3dt')
scorient.add_state('q4', 'dq4dt')
scorient.add_state('w1', 'dw1dt')
scorient.add_state('w2', 'dw2dt')
scorient.add_state('w3', 'dw3dt')

# Costate Dynamics
scorient.add_state('lam_q1', '(lam_q2*w3 - lam_q3*w2 + lam_q4*w1)/2')
scorient.add_state('lam_q2', '(-lam_q1*w3 + lam_q3*w1 + lam_q4*w2)/2')
scorient.add_state('lam_q3', '(lam_q1*w2 - lam_q2*w1 + lam_q4*w3)/2')
scorient.add_state('lam_q4', '(lam_q1*w1 - lam_q2*w2 - lam_q3*w3)/2')
scorient.add_state('lam_w1', '(-lam_q1*q4 - lam_q2*q3 + lam_q3*q2 + lam_q4*q1)/2 '
                             '- lam_w3*w2*(I1 - I2)/I3 - lam_w2*w3*(-I1 + I3)/I2')
scorient.add_state('lam_w2', '(lam_q1*q3 - lam_q2*q4 - lam_q3*q1 + lam_q4*q2)/2 '
                             '- lam_w3*w1*(I1 - I2)/I3 - lam_w1*w3*(I2 - I3)/I1')
scorient.add_state('lam_w3', '(-lam_q1*q2/2 + lam_q2*q1 - lam_q3*q4 + lam_q4*q3)/2 '
                             '- lam_w2*w1*(-I1 + I3)/I2 - lam_w1*w2*(I2 - I3)/I1')

# Explicit Control Law
eps_u = 1e-3
scorient.add_constant('eps_u', eps_u)

scorient.add_expression('u1', '-lam_w1/I1 / (eps_u**2 + (lam_w1/I1)**2) ** 0.5')
scorient.add_expression('Lu1', 'eps_u * (1 - eps_u / (eps_u**2 + (lam_w1/I1)**2) ** 0.5)')
scorient.add_expression('u2', '-lam_w2/I2 / (eps_u**2 + (lam_w2/I2)**2) ** 0.5')
scorient.add_expression('Lu2', 'eps_u * (1 - eps_u / (eps_u**2 + (lam_w2/I2)**2) ** 0.5)')
scorient.add_expression('u3', '-lam_w3/I3 / (eps_u**2 + (lam_w3/I3)**2) ** 0.5')
scorient.add_expression('Lu3', 'eps_u * (1 - eps_u / (eps_u**2 + (lam_w3/I3)**2) ** 0.5)')

# Augmented Path Cost
scorient.add_expression(
    'H',
    '1 + Lu1 + Lu2 + Lu3 '
    '+ lam_q1 * dq1dt + lam_q2 * dq2dt + lam_q3 * dq3dt + lam_q4 * dq4dt '
    '+ lam_w1 * dw1dt + lam_w2 * dw2dt + lam_w3 * dw3dt'
)
scorient.set_cost('0', 'H', '0')

# State Boundary Conditions
q1_0 = 0.0
q2_0 = 0.0
q3_0 = 0.0
q4_0 = 1.0
w1_0 = 0.0
w2_0 = 0.0
w3_0 = 0.0
scorient.add_constant('q1_0', q1_0)
scorient.add_constant('q2_0', q2_0)
scorient.add_constant('q3_0', q3_0)
scorient.add_constant('q4_0', q4_0)
scorient.add_constant('w1_0', w1_0)
scorient.add_constant('w2_0', w2_0)
scorient.add_constant('w3_0', w3_0)

q1_f = 0.0
q2_f = 0.0
q3_f = 1.0
q4_f = 0.0
w1_f = 0.0
w2_f = 0.0
w3_f = 0.0
scorient.add_constant('q1_f', q1_f)
scorient.add_constant('q2_f', q2_f)
scorient.add_constant('q3_f', q3_f)
scorient.add_constant('q4_f', q4_f)
scorient.add_constant('w1_f', w1_f)
scorient.add_constant('w2_f', w2_f)
scorient.add_constant('w3_f', w3_f)

scorient.add_constraint('initial', 't')
scorient.add_constraint('initial', 'q1 - q1_0')
scorient.add_constraint('initial', 'q2 - q2_0')
scorient.add_constraint('initial', 'q3 - q3_0')
scorient.add_constraint('initial', 'q4 - q4_0')
scorient.add_constraint('initial', 'w1 - w1_0')
scorient.add_constraint('initial', 'w2 - w2_0')
scorient.add_constraint('initial', 'w3 - w3_0')

scorient.add_constraint('terminal', 'q1 - q1_f')
scorient.add_constraint('terminal', 'q2 - q2_f')
scorient.add_constraint('terminal', 'q3 - q3_f')
scorient.add_constraint('terminal', 'q4 - q4_f')
scorient.add_constraint('terminal', 'w1 - w1_f')
scorient.add_constraint('terminal', 'w2 - w2_f')
scorient.add_constraint('terminal', 'w3 - w3_f')

# Costate Boundary Conditions
scorient.add_constraint('initial', 'H')  # Equivalent to Hf = 0 b/c H(t) = 0. Does not require propagation to compute.

# Compilation
with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_scorient = giuseppe.problems.symbolic.SymBVP(scorient).compile(use_jit_compile=False)
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_scorient, verbose=2, max_nodes=100, node_buffer=10)

# idx_eps_u = comp_scorient.annotations.constants.index('eps_u')
# idx_I1 = comp_scorient.annotations.constants.index('I1')
# idx_I2 = comp_scorient.annotations.constants.index('I2')
# idx_I3 = comp_scorient.annotations.constants.index('I3')


def hamiltonian(_x, _lam):
    q1 = _x[0]
    q2 = _x[1]
    q3 = _x[2]
    q4 = _x[3]
    w1 = _x[4]
    w2 = _x[5]
    w3 = _x[6]

    lam_q1 = _lam[0]
    lam_q2 = _lam[1]
    lam_q3 = _lam[2]
    lam_q4 = _lam[3]
    lam_w1 = _lam[4]
    lam_w2 = _lam[5]
    lam_w3 = _lam[6]

    # Control Law
    u1 = -lam_w1/I1 / (eps_u**2 + (lam_w1/I1)**2) ** 0.5
    Lu1 = eps_u - eps_u**2 / (eps_u**2 + (lam_w1/I1)**2) ** 0.5
    u2 = -lam_w2/I2 / (eps_u**2 + (lam_w2/I2)**2) ** 0.5
    Lu2 = eps_u - eps_u**2 / (eps_u**2 + (lam_w2/I2)**2) ** 0.5
    u3 = -lam_w3/I3 / (eps_u**2 + (lam_w3/I3)**2) ** 0.5
    Lu3 = eps_u - eps_u**2 / (eps_u**2 + (lam_w3/I3)**2) ** 0.5

    # Dynamics
    dq1dt = (w1*q4 - w2*q3 + w3*q2)/2
    dq2dt = (w1*q3 + w2*q4 - w3*q1)/2
    dq3dt = (-w1*q2 + w2*q1 + w3*q4)/2
    dq4dt = (-w1*q1 - w2*q2 - w3*q3)/2
    dw1dt = u1/I1 + ((I2 - I3)/I1)*w2*w3
    dw2dt = u2/I2 + ((I3 - I1)/I2)*w1*w3
    dw3dt = u3/I3 + ((I1 - I2)/I3)*w1*w2

    # dlam_q1 = (lam_q2*w3 - lam_q3*w2 + lam_q4*w1)/2
    # dlam_q2 = (-lam_q1*w3 + lam_q3*w1 + lam_q4*w2)/2
    # dlam_q3 = (lam_q1*w2 - lam_q2*w1 + lam_q4*w3)/2
    # dlam_q4 = (lam_q1*w1 - lam_q2*w2 - lam_q3*w3)/2
    # dlam_q4 = (lam_q1*w1 - lam_q2*w2 - lam_q3*w3)/2
    # dlam_w1 = (-lam_q1*q4 - lam_q2*q3 + lam_q3*q2 + lam_q4*q1)/2 - lam_w3*w2*(I1 - I2)/I3 - lam_w2*w3*(-I1 + I3)/I2
    # dlam_w2 = (lam_q1*q3 - lam_q2*q4 - lam_q3*q1 + lam_q4*q2)/2 - lam_w3*w1*(I1 - I2)/I3 - lam_w1*w3*(I2 - I3)/I1
    # dlam_w3 = (-lam_q1*q2/2 + lam_q2*q1 - lam_q3*q4 + lam_q4*q3)/2 - lam_w2*w1*(-I1 + I3)/I2 - lam_w1*w2*(I2 - I3)/I1

    # Augmented Path Cost
    H = 1 + Lu1 + Lu2 + Lu3 \
        + lam_q1 * dq1dt + lam_q2 * dq2dt + lam_q3 * dq3dt + lam_q4 * dq4dt \
        + lam_w1 * dw1dt + lam_w2 * dw2dt + lam_w3 * dw3dt
    return H


immutable_constants = (
    'I1', 'I2', 'I3', 'eps_u',
    'q1_0', 'q2_0', 'q3_0', 'q4_0', 'w1_0', 'w2_0', 'w3_0'
)
t_span = np.linspace(0., 1., 25)
initial_states = np.array((q1_0, q2_0, q3_0, q4_0, w1_0, w2_0, w3_0))
initial_costates = np.array((0., 0., -1., 1., 0., 0., -I3 * np.sqrt(1 + 2*eps_u)))
xlam0 = np.concatenate((initial_states, initial_costates))

guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_scorient, t_span=t_span,
    initial_states=np.concatenate((initial_states, initial_costates)),
    fit_states=False,
    match_constants=True, immutable_constants=immutable_constants
)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

idx_q1_f = comp_scorient.annotations.constants.index('q1_f')
idx_q2_f = comp_scorient.annotations.constants.index('q2_f')
idx_q3_f = comp_scorient.annotations.constants.index('q3_f')
idx_q4_f = comp_scorient.annotations.constants.index('q4_f')
q3_f_seed = seed_sol.k[idx_q3_f]
sign_q4 = np.sign(q4_0 + q4_f)


def q3_continuation(previous_sol, frac_complete):
    _constants = previous_sol.k.copy()
    _q1_f = _constants[idx_q1_f]
    _q2_f = _constants[idx_q2_f]
    _q3_f = q3_f_seed + frac_complete * (q3_f - q3_f_seed)
    _q4_f = sign_q4 * np.sqrt(1 - _q1_f**2 - _q2_f**2 - _q3_f**2)  # Enforce ||q|| = 1

    _constants[idx_q3_f] = _q3_f
    _constants[idx_q4_f] = _q4_f
    return _constants


# Calculate Jacobian Matrix
dynamics_sym = comp_scorient.source_bvp.dynamics
states_sym = comp_scorient.source_bvp.states
time_sym = comp_scorient.source_bvp.independent
# time_states_sym = giuseppe.utils.typing.SymMatrix([time_sym, states_sym])
# time_state_jac_sym = dynamics_sym.jacobian(time_states_sym)
state_jac_sym = dynamics_sym.jacobian(states_sym)
compute_dynamics = comp_scorient.compute_dynamics
# compute_time_state_jac = giuseppe.utils.compilation.lambdify(
#     comp_scorient.sym_args, time_state_jac_sym, use_jit_compile=comp_scorient.use_jit_compile
# )
compute_state_jac = giuseppe.utils.compilation.lambdify(
    comp_scorient.sym_args, state_jac_sym, use_jit_compile=comp_scorient.use_jit_compile
)
n_x = states_sym.shape[0]

dtf_dtfxf = np.vstack((1., np.zeros((n_x, 1))))
# TODO - dH/d(t,x), etc.


def eom_state_stm(_t, _x_stm, _p, _k):
    _x = _x_stm[:n_x]
    _stm_flat = _x_stm[n_x:]
    _stm = _stm_flat.reshape((n_x, n_x))

    _dx_dt = compute_dynamics(_t, _x, _p, _k)
    _dstm_dt = compute_state_jac(_t, _x, _p, _k) @ _stm

    _dx_stm_dt = np.concatenate((_dx_dt, _dstm_dt.flatten()))
    return _dx_stm_dt


# Continue boundary conditions using a shooting method
def shooting_jac(_seed_sol):
    _seed_sol = deepcopy(_seed_sol)
    if not _seed_sol.converged:
        _seed_sol = num_solver.solve(_seed_sol)

    # Initial state/STM
    _x0 = _seed_sol.x[:, 0]
    _stm0 = np.eye(_x0.shape[0])
    _t0 = _seed_sol.t[0]
    _tf = _seed_sol.t[-1]
    _x_stm0 = np.concatenate((_x0, _stm0.flatten()))

    # Solve for terminal state/STM
    ivp_sol = solve_ivp(
        fun=lambda t, x_stm: eom_state_stm(t, x_stm, seed_sol.p, seed_sol.k),
        t_span=(_t0, _tf), y0=_x_stm0
    )
    _x_stm_f = ivp_sol.y
    _x_f = _x_stm_f[:n_x, -1]
    _stm_f = _x_stm_f[n_x:, -1].reshape((n_x, n_x))

    # Build Jacobians for correcting parameter estimates
    _dtf_dtfxf = np.vstack((1., np.zeros((n_x, 1))))

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(100, {'w3_f': 0.})
# cont.add_custom_series(100, q3_continuation, 'q3/q4')

sol_set = cont.run_continuation()

