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
I2 = 3.
I3 = 3.

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
idx_w1_f = comp_scorient.annotations.constants.index('w1_f')
idx_w2_f = comp_scorient.annotations.constants.index('w2_f')
idx_w3_f = comp_scorient.annotations.constants.index('w3_f')
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


# Derive Expressions for Shooting
dynamics_sym = comp_scorient.source_bvp.dynamics
states_sym = comp_scorient.source_bvp.states
# time_sym = comp_scorient.source_bvp.independent
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
n_x2 = int(n_x/2)
zero_1n = np.zeros((1, n_x))

hamiltonian_sym = comp_scorient.source_bvp.boundary_conditions.initial[-1, :]
ham_x_sym = hamiltonian_sym.jacobian(states_sym)
compute_ham = giuseppe.utils.compilation.lambdify(
    comp_scorient.sym_args, hamiltonian_sym, use_jit_compile=comp_scorient.use_jit_compile
)
compute_ham_x = giuseppe.utils.compilation.lambdify(
    comp_scorient.sym_args, ham_x_sym, use_jit_compile=comp_scorient.use_jit_compile
)


def eom_state_stm(_tau, _x_stm, _p, _k, _t0, _tf):
    _dt_dtau = _tf - _t0
    _t = _t0 + _dt_dtau * _tau
    _x = _x_stm[:n_x]
    _stm_flat = _x_stm[n_x:]
    _stm = _stm_flat.reshape((1+n_x, 1+n_x))

    _dx_dt = compute_dynamics(_t, _x, _p, _k)
    _dx_dtau = _dx_dt * _dt_dtau
    _f_jac = compute_state_jac(_t, _x, _p, _k)
    _dstm_dt = np.hstack((
        np.concatenate((np.array((_dt_dtau,)), _dx_dt)).reshape((-1, 1)),  # Jac for t
        np.vstack((zero_1n, _dt_dtau * _f_jac))
    )) @ _stm

    _dx_stm_dt = np.concatenate((_dx_dt, _dstm_dt.flatten()))

    return _dx_stm_dt


# Continue boundary conditions using a shooting method
def shooting_jac(_seed_sol):
    _seed_sol = deepcopy(_seed_sol)
    if not _seed_sol.converged:
        _seed_sol = num_solver.solve(_seed_sol)

    # Desired final states
    _x_f_cmd = _seed_sol.k[((idx_q1_f, idx_q2_f, idx_q3_f, idx_q4_f, idx_w1_f, idx_w2_f, idx_w3_f),)]

    # Initial state/STM
    _x0 = _seed_sol.x[:, 0]
    _stm0 = np.eye(1+_x0.shape[0])
    _t0 = _seed_sol.t[0]
    _tf = _seed_sol.t[-1]
    _x_stm0 = np.concatenate((_x0, _stm0.flatten()))

    # Solve for terminal state/STM
    ivp_sol = solve_ivp(
        fun=lambda tau, x_stm: eom_state_stm(tau, x_stm, seed_sol.p, seed_sol.k, _t0, _tf),
        t_span=(0., 1.), y0=_x_stm0
    )
    _x_stm_f = ivp_sol.y
    _x_f = _x_stm_f[1:n_x, -1]
    _stm_f = _x_stm_f[n_x:, -1].reshape((1+n_x, 1+n_x))
    _dxf_dtf = _stm_f[1:1+n_x2, 0:1]
    _dxf_dlam0 = _stm_f[1:1+n_x2, 1+n_x2:]

    # Build Jacobians for correcting parameter estimates
    # Jac for H(0) = 0
    _jac0 = np.hstack((np.zeros((1, 1)), compute_ham_x(_t0, _x0, seed_sol.p, seed_sol.k)[0:1, n_x2:]))

    # Jac for q(tf) = qf, w(tf) = wf
    _jacf = np.hstack((_dxf_dtf, _dxf_dlam0))

    # Jac for H(f) = 0
    _dhamf_dtfx0 = np.hstack((np.zeros((1, 1)), compute_ham_x(_tf, _x_f, seed_sol.p, seed_sol.k)[0:1, :])) @ _stm_f
    _dhamf_dtf = _dhamf_dtfx0[:, 0:1]
    _dhamf_dlam0 = _dhamf_dtfx0[:, 1+n_x2:]
    _jacf_aug = np.hstack((_dhamf_dtf, _dhamf_dlam0))

    _jac = np.vstack((_jac0, _jacf))
    # _jac = np.vstack((_jac0, _jacf, _jacf_aug))

    # Residual
    _ham0 = compute_ham(_t0, _x0, seed_sol.p, seed_sol.k)
    _x_err = _x_f[:n_x2] - _x_f_cmd
    _hamf = compute_ham(_tf, _x_f, seed_sol.p, seed_sol.k)
    _res = np.vstack((_ham0, _x_err.reshape((-1, 1))))
    # _res = np.vstack((_ham0, _x_err.reshape((-1, 1)), _hamf))
    _pinv_sol = np.linalg.lstsq(_jac, _res, rcond=None)
    _full_step = _pinv_sol[0].flatten()

    # Back-tracking linsearch
    _z0 = np.concatenate((np.array((_tf,)), _x0[n_x2:]))
    _steps_max = 10
    _alpha = 1.

    for _idx in _steps_max:
        _z0_new = _z0 - _alpha * _full_step
        _tf_new = _z0_new[0]
        _lam0_new = _z0_new[1:]
        _x0_new = np.concatenate((_x0[:n_x2], _lam0_new))

        _ivp_sol = solve_ivp(
            fun=lambda _t, _x: compute_dynamics(_t, _x, seed_sol.p, seed_sol.k),
            t_span=(_t0, _tf_new), y0=_x0_new
        )
        _x_f_new = _ivp_sol.y[:, -1]

        _ham0_new = compute_ham(_t0, _x0_new, seed_sol.p, seed_sol.k)
        _x_err_new = _x_f_new[:n_x2] - _x_f_cmd
        _hamf_new = compute_ham(_tf, _x_f, seed_sol.p, seed_sol.k)
        # _res_new = np.vstack((_ham0_new, _x_err_new.reshape((-1, 1))))
        _res_new = np.vstack((_ham0_new, _x_err_new.reshape((-1, 1)), _hamf_new)).flatten()

