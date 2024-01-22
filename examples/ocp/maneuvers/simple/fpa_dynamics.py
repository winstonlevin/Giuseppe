# Solve for minimum time from x0 to xf using FPA as control and assuming constant energy.
import numpy as np
import pickle

import giuseppe

input = giuseppe.problems.input.StrInputProb()

input.set_independent('t')

input.add_state('h', 'V * sin(gam)')
input.add_state('x', 'V * cos(gam)')
input.add_state('gam', 'u')

input.add_control('u')

E0 = 0.5*100**2
g = 9.8
input.add_expression('V', '(2*(E - g*h))**0.5')
input.add_constant('E', E0)
input.add_constant('g', g)

# Boundary conditions
h0 = 0
input.add_constraint('initial', 't')
input.add_constraint('initial', 'x')
input.add_constraint('initial', 'h - h0')
input.add_constraint('initial', 'gam')
input.add_constant('h0', h0)

hf = 0
xf = 1
input.add_constraint('terminal', 'x - xf')
input.add_constraint('terminal', 'h - hf')
input.add_constraint('terminal', 'gam')
input.add_constant('xf', xf)
input.add_constant('hf', hf)

# Set Cost
input.set_cost('0', '1 + 0.5*eps*u**2', '0')  # Minimum time
eps = 1e-2
input.add_constant('eps', eps)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp = giuseppe.problems.symbolic.SymDual(
        input, control_method='algebraic'
    ).compile(use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp, verbose=0, max_nodes=1000, node_buffer=10)

# Generate convergent solution
guess = giuseppe.guess_generation.auto_propagate_guess(
    comp,
    control=0.,
    # t_span=1.
    t_span=np.linspace(0., 1., 5),
)

with open('guess_fpa_dyn.data', 'wb') as f:
    pickle.dump(guess, f)

seed_sol = num_solver.solve(guess)

with open('seed_sol_fpa_dyn.data', 'wb') as f:
    pickle.dump(seed_sol, f)

# Continuations
cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_logarithmic_series(100, {'xf': 1e8})
sol_set = cont.run_continuation()
sol_set.save('sol_set_fpa_dyn.data')
# TODO - write plotting script
