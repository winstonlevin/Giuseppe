import os

import numpy as np

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

intercept = giuseppe.problems.input.StrInputProb()

intercept.set_independent('t')

intercept.add_state('x', 'v*cos(psi)')
intercept.add_state('y', 'v*sin(psi)')
intercept.add_state('psi', 'u_sat')

# The Saturation function comes from the formula:
# Sat(x | xL, xU) = 0.5*(xL + xU + |x - xL| - |x - xU|)
#
# Which is equivalent to:
#
#                   { xL  if x < xL
# Sat(x | xL, xU) = { xU  if x > xU
#                   { x   otherwise
#
# The derivative of this function is undefined at x = xL or x = xU, so the "smooth" saturation function is:
#
# |y| ~= (y**2 + eps**2)**0.5
#
# Whose derivatives are well-defined everywhere.
intercept.add_control('u')
intercept.add_constant('eps_u', 1E-3)
intercept.add_expression('u_sat', '0.5*(( (u+1)**2 + eps_u**2 )**0.5 - ( (u-1)**2 + eps_u**2 )**0.5)')

intercept.add_constant('v', 1)

intercept.add_constant('x_0', 0.)
intercept.add_constant('y_0', 0.)
intercept.add_constant('psi_0', 0.)

intercept.add_constant('x_f', 0.)
intercept.add_constant('y_f', 6.)
intercept.add_constant('psi_f', -0.5*np.pi)

intercept.add_constant('k', 1.)
intercept.set_cost('0', 'k + 0.5*u**2', '0')

intercept.add_constraint('initial', 't')
intercept.add_constraint('initial', 'x - x_0')
intercept.add_constraint('initial', 'y - y_0')
intercept.add_constraint('initial', 'psi - psi_0')

intercept.add_constraint('terminal', 'x - x_f')
intercept.add_constraint('terminal', 'y - y_f')
intercept.add_constraint('terminal', 'psi - psi_f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    intercept = giuseppe.problems.symbolic.SymDual(intercept, control_method='algebraic').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(intercept, verbose=2, max_nodes=100, node_buffer=10)

guess = giuseppe.guess_generation.auto_propagate_guess(intercept, control=0., t_span=1.0)
seed_sol = num_solver.solve(guess)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'x_f': 6., 'y_f': 0.})
cont.add_linear_series(1, {'psi_0': 0.5*np.pi, 'psi_f': -0.5*np.pi})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
