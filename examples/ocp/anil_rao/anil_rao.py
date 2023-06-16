import giuseppe

arao = giuseppe.problems.input.StrInputProb()

# Dynamics
arao.set_independent('t')
arao.add_state('y', '-y**3 + u')
arao.add_control('u')

# Cost
arao.set_cost(initial='0', path='y**2 + u**2', terminal='0')

# Boundary Conditions
arao.add_constraint('initial', 't')
arao.add_constraint('initial', 'y - y0')
arao.add_constant('y0', 1.)
arao.add_constraint('terminal', 'y - yf')
arao.add_constant('yf', 1.5)
arao.add_constraint('terminal', 't - tf')
arao.add_constant('tf', 1e4)

with giuseppe.utils.Timer(prefix='Compilation Time:'):
    comp_arao = giuseppe.problems.symbolic.SymDual(arao, control_method='algebraic').compile()
    num_solver = giuseppe.numeric_solvers.SciPySolver(comp_arao, verbose=2, max_nodes=100, node_buffer=10)

guess = giuseppe.guess_generation.auto_propagate_guess(
    comp_arao,
    control=0.,
    t_span=1.0)

seed_sol = num_solver.solve(guess)

cont = giuseppe.continuation.ContinuationHandler(num_solver, seed_sol)
cont.add_linear_series(1, {'yf': 1.5})
cont.add_linear_series(100, {'tf': 1e4})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')
