from copy import deepcopy

import numpy as np
import giuseppe
import pickle

ocp = giuseppe.problems.input.StrInputProb()

ocp.set_independent('t')

# Equations of Motion
ocp.add_state('om1', 'a * om30 * om2 + u1')
ocp.add_state('om2', '-a * om30 * om1 + u2')
ocp.add_state('x1', 'om30 * x2 + om2 * x1 * x2 + 0.5 * om1 * (1 + x1**2 - x2**2)')
ocp.add_state('x2', 'om30 * x1 + om1 * x1 * x2 + 0.5 * om2 * (1 + x2**2 - x1**2)')

ocp.add_constant('a', 0.)
ocp.add_constant('om30', 0.)

# Control
ocp.add_control('u1')
ocp.add_control('u2')

u_min = -1.
u_max = 1.
ocp.add_constant('u_min', u_min)
ocp.add_constant('u_max', u_max)
ocp.add_constant('eps_u', 1e-3)

ocp.add_inequality_constraint(
    'control', 'u1', lower_limit='u_min', upper_limit='u_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin')
)

ocp.add_inequality_constraint(
    'control', 'u2', lower_limit='u_min', upper_limit='u_max',
    regularizer=giuseppe.problems.symbolic.regularization.ControlConstraintHandler('eps_u', method='sin')
)

# Boundary Conditions
ocp.add_constant('om10', 0.)
ocp.add_constant('om20', 0.)
ocp.add_constant('x10', 0.)
ocp.add_constant('x20', 0.)

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'om1 - om10')
ocp.add_constraint('initial', 'om2 - om20')
ocp.add_constraint('initial', 'x1 - x10')
ocp.add_constraint('initial', 'x2 - x20')

ocp.add_constant('om1f', 0.)
ocp.add_constant('om2f', 0.)

ocp.add_constraint('terminal', 'om1 - om1f')
ocp.add_constraint('terminal', 'om2 - om2f')

# Cost (Minimum Time)
ocp.set_cost('0', '0', 't')

# Variant with constrained final state
ocp_xcon = deepcopy(ocp)

ocp_xcon.add_constant('x1f', 0.)
ocp_xcon.add_constant('x2f', 0.)
ocp_xcon.add_constraint('terminal', 'x1 - x1f')
ocp_xcon.add_constraint('terminal', 'x2 - x2f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_ocp = giuseppe.problems.symbolic.SymDual(ocp, control_method='differential').compile()
    comp_ocp_xcon = giuseppe.problems.symbolic.SymDual(ocp_xcon, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_ocp, verbose=False, max_nodes=100, node_buffer=10)
    num_solver_xcon = giuseppe.numeric_solvers.SciPySolver(comp_ocp_xcon, verbose=2, max_nodes=100, node_buffer=10)


def ctrl2reg(u: np.array) -> np.array:
    return np.arcsin((2*u - u_min - u_max) / (u_max - u_min))


def reg2ctrl(u_reg: np.array) -> np.array:
    return 0.5 * ((u_max - u_min) * np.sin(u_reg) + u_max + u_min)


guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_ocp,
    control=ctrl2reg(np.array((1.0, 0.0))),
    t_span=0.1
)

with open('guess.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

if seed_sol.converged:
    print(f'Seed sol IS converged!')
else:
    print(f'Seed sol is NOT converged!')

with open('seed_sol.data', 'wb') as f:
    pickle.dump(seed_sol, f)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(25, {'om1f': 1., 'om2f': 2})
cont.add_linear_series(25, {'om30': -0.3})
cont.add_logarithmic_series(25, {'eps_u': 1e-6})

sol_set_case2 = cont.run_continuation()

sol_set_case2.save('sol_set_case2.data')

cont = giuseppe.continuation.ContinuationHandler(num_solver, deepcopy(seed_sol))
# cont.add_linear_series(30, {'x10': 0.1, 'x20': -0.1, 'om10': -0.45, 'om20': -1.10, 'om1f': 0., 'om2f': 0., 'a': 0.5})
cont.add_linear_series(30, {'x10': 0.1, 'x20': -0.1, 'om10': -0.45, 'om1f': 0.5, 'om2f': 0., 'a': 0.5})
sol_set_case1_xfree = cont.run_continuation()

# guess_xcon = giuseppe.guess_generation.auto_propagate_guess(
#     comp_ocp_xcon,
#     control=ctrl2reg(np.array((0.25, 0.0))),
#     t_span=0.1, reverse=True
# )
#
# with open('guess_xcon.data', 'wb') as f:
#     pickle.dump(guess_xcon, f)
#
# seed_sol_xcon = num_solver_xcon.solve(guess_xcon)
#
# if seed_sol_xcon.converged:
#     print(f'Seed sol IS converged!')
# else:
#     print(f'Seed sol is NOT converged!')
#
# with open('seed_sol.data', 'wb') as f:
#     pickle.dump(seed_sol_xcon, f)

guess_xcon = deepcopy(sol_set_case1_xfree.solutions[-1])
guess_xcon.k = np.append(guess_xcon.k, guess_xcon.x[2:, -1])  # Add x1f, x2f to constants
guess_xcon.nuf = np.append(guess_xcon.nuf, (0., 0.))  # Add adjoints for x1f, x2f

seed_sol_xcon = num_solver_xcon.solve(guess_xcon)

cont = giuseppe.continuation.ContinuationHandler(num_solver_xcon, seed_sol_xcon)
cont.add_linear_series(100, {'eps_u': 1e-1})
cont.add_linear_series(100, {'x2f': 0., 'om20': 0.})
cont.add_linear_series(100, {'x1f': 0., 'om20': -1.1})
cont.add_logarithmic_series(100, {'eps_u': 1e-6})

sol_set_case1 = cont.run_continuation()

sol_set_case1.save('sol_set_case1.data')
