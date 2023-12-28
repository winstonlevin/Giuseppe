import numpy as np
import giuseppe
import pickle

scorient = giuseppe.problems.input.StrInputProb()

scorient.set_independent('t')

scorient.add_state('q1', '(w1*q4 - w2*q3 + w3*q2)/2')
scorient.add_state('q2', '(w1*q3 + w2*q4 - w3*q1)/2')
scorient.add_state('q3', '(-w1*q2 + w2*q1 + w3*q4)/2')
scorient.add_state('q4', '(-w1*q1 - w2*q2 - w3*q3)/2')
scorient.add_state('w1', 'u1/Ix + ((Iy - Iz)/Ix)*w2*w3')
scorient.add_state('w2', 'u2/Iy + ((Iz - Ix)/Iy)*w1*w3')
scorient.add_state('w3', 'u3/Iz + ((Ix - Iy)/Iz)*w1*w2')

scorient.add_control('u1')
scorient.add_control('u2')
scorient.add_control('u3')

# Case 17 of the paper
Ix = 1
Iy = 100
Iz = 100
scorient.add_constant('Ix', 1)
scorient.add_constant('Iy', 1)
scorient.add_constant('Iz', 1)

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

u_min = -1.0
u_max = 1.0
eps_u = 1e-3
scorient.add_constant('eps_u', eps_u)
scorient.add_constant('u_min', u_min)
scorient.add_constant('u_max', u_max)

# scorient.add_constant('cost_off', 1e3)
# scorient.set_cost('0', '1 + cost_off * (u1**2 + u2**2)', '0')  # Minimum time problem
scorient.set_cost('0', '1', '0')  # Minimum time problem

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

reg_meth = 'sin'
scorient.add_inequality_constraint(
        'control', 'u1', lower_limit='u_min', upper_limit='u_max',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method=reg_meth))

scorient.add_inequality_constraint(
        'control', 'u2', lower_limit='u_min', upper_limit='u_max',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method=reg_meth))

scorient.add_inequality_constraint(
        'control', 'u3', lower_limit='u_min', upper_limit='u_max',
        regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method=reg_meth))


with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_scorient = giuseppe.problems.symbolic.SymDual(
        scorient, control_method='algebraic'
    ).compile(use_jit_compile=False)
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_scorient, verbose=0, max_nodes=0, node_buffer=10)

if reg_meth in ['trig', 'sin']:
    def ctrl2reg(_alpha):
        return np.arcsin(2/(u_max - u_min) * (_alpha - 0.5*(u_max + u_min)))
elif reg_meth in ['atan', 'arctan']:
    def ctrl2reg(_alpha):
        return eps_u * np.tan(0.5 * (2*_alpha - u_max - u_min) * np.pi / (u_max - u_min))
else:
    def ctrl2reg(_alpha):
        return _alpha

t_span = np.linspace(0., 0.1, 6)
t_sw = 0.5 * t_span[-1]
uf = 0.5


def ctrl_law(_t, _x, _p, _k):
    _u3 = 2 * uf * (_t - t_sw)
    return ctrl2reg(np.array((0., 0., _u3)))


guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_scorient,
    t_span=t_span,
    initial_costates=np.array((0., 0., 1., 1., 0., 0., 1.)) + 1e-6,
    fit_adjoints=False
)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

print(seed_sol.converged)

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

# cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
#
# cont.add_linear_series(100, {'q1_f': q1_f, 'q2_f': q2_f, 'q3_f': q3_f, 'q4_f': q4_f})
# cont.add_linear_series(100, {'w1_f': w1_f, 'w2_f': w2_f, 'w3_f': w3_f})
# cont.add_logarithmic_series(200, {'eps_u': 1e-6})
#
# sol_set = cont.run_continuation()
#
# sol_set.save('sol_set.data')
