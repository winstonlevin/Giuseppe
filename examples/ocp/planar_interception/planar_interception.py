import os

import numpy as np

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

intercept = giuseppe.problems.input.StrInputProb()

intercept.set_independent('t')

intercept.add_state('x', 'v*cos(psi)')
intercept.add_state('y', 'v*sin(psi)')
intercept.add_state('psi', 'sin(u)')

intercept.add_control('u')

intercept.add_constant('v', 1.)

intercept.add_constant('x_0', 0.)
intercept.add_constant('y_0', 0.)
intercept.add_constant('psi_0', 0.)

intercept.add_constant('x_f', 0.)
intercept.add_constant('y_f', 6.)
intercept.add_constant('psi_f', -0.5*np.pi)

intercept.add_constant('k', 1.)
intercept.add_constant('eps_u', 1E-1)
intercept.set_cost('0', '1 + 0.5*k*sin(u)**2 - eps_u*cos(u)', '0')

intercept.add_constraint('initial', 't')
intercept.add_constraint('initial', 'x - x_0')
intercept.add_constraint('initial', 'y - y_0')
intercept.add_constraint('initial', 'psi - psi_0')

intercept.add_constraint('terminal', 'x - x_f')
intercept.add_constraint('terminal', 'y - y_f')
intercept.add_constraint('terminal', 'psi - psi_f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_dual = giuseppe.problems.symbolic.SymDual(intercept, control_method='differential').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_dual, verbose=2, max_nodes=500, node_buffer=10)

guess = giuseppe.guess_generation.auto_propagate_guess(comp_dual, control=0., t_span=1.0)
seed_sol = num_solver.solve(guess)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'x_f': 6., 'y_f': 0.})
cont.add_linear_series(1, {'psi_0': 0.5*np.pi, 'psi_f': -0.5*np.pi})
sol_set = cont.run_continuation()

cont_k1 = giuseppe.continuation.ContinuationHandler(num_solver, sol_set[-1])
cont_k1.add_logarithmic_series(10, {'eps_u': 1E-6})
sol_set_k1 = cont_k1.run_continuation()
sol_set_k1.save('sol_set.data')

cont_klow = giuseppe.continuation.ContinuationHandler(num_solver, sol_set[-1])
cont_klow.add_logarithmic_series(10, {'eps_u': 1E-6, 'k': 1E-3})
sol_set_klow = cont_klow.run_continuation()
sol_set_klow.save('sol_set_klow.data')

cont_switch = giuseppe.continuation.ContinuationHandler(num_solver, sol_set[-1])
cont_switch.add_linear_series(1, {'psi_f': 0.5*np.pi})
cont_switch.add_logarithmic_series(10, {'eps_u': 1E-6})
sol_set_switch = cont_switch.run_continuation()
sol_set_switch.save('sol_set_switch.data')
