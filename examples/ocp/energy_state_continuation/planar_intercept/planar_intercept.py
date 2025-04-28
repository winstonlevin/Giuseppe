import os
from copy import copy
import pickle

import numpy as np
import casadi as ca

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set directory to current location

# -------------------------------------------------------------------------------------------------------------------- #
# ANALYTICAL VALUES                                                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
# Boundary conditions
x0 = np.zeros(shape=(3,), dtype=float)
x0[2] = np.pi/2
xf = np.zeros(shape=(3,), dtype=float)
xf[0] = 10.
xf[1] = -2.
xf[2] = np.pi/3

# Outer solution
dp = xf[:2] - x0[:2]
p_hat = dp / np.linalg.norm(dp)

psi_outer = np.arctan2(dp[1], dp[0])
u_outer = np.array((0.,))
lam_outer = np.array((-p_hat[0], -p_hat[1], 0.))
tf_outer = np.dot(dp, dp)**0.5

# -------------------------------------------------------------------------------------------------------------------- #
# GIUSEPPE SOLUTION                                                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
intercept = giuseppe.problems.input.StrInputProb()

intercept.set_independent('t')

intercept.add_state('x', 'cos(psi)')
intercept.add_state('y', 'sin(psi)')
intercept.add_state('psi', 'u')

intercept.add_control('u')


intercept.add_constant('x_0', x0[0])
intercept.add_constant('y_0', x0[1])
intercept.add_constant('psi_0', psi_outer)


intercept.add_constant('x_f', xf[0])
intercept.add_constant('y_f', xf[1])
intercept.add_constant('psi_f', psi_outer)

k0 = 1.
kf = 1E-2
intercept.add_constant('k', k0)
intercept.set_cost('0', '1 + k/2*(u*u)', '0')

intercept.add_constraint('initial', 't')
intercept.add_constraint('initial', 'x - x_0')
intercept.add_constraint('initial', 'y - y_0')
intercept.add_constraint('initial', 'psi - psi_0')

intercept.add_constraint('terminal', 'x - x_f')
intercept.add_constraint('terminal', 'y - y_f')
intercept.add_constraint('terminal', 'psi - psi_f')

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_dual = giuseppe.problems.symbolic.SymDual(intercept, control_method='algebraic').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_dual, verbose=2)

guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_dual, control=0., t_span=1.0, immutable_constants=('k', 'x_0', 'y_0')
)
seed_sol = num_solver.solve(guess)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'x_0': x0[0], 'y_0': x0[1], 'x_f': xf[0], 'y_f': xf[1]})
cont.add_linear_series(1, {'psi_0': x0[2], 'psi_f': xf[2]})
cont.add_logarithmic_series(5, {'k': kf})
sol_set = cont.run_continuation()
sol_set.save('sol_set.data')
