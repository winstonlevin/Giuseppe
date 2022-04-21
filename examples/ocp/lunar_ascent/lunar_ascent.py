import pickle

import numpy as np

import giuseppe

G = 5.3
NM2FT = 6076.1
T_GUESS = 10
R_M = 938 * NM2FT

lunar = giuseppe.io.InputOCP()

lunar.set_independent('t')

lunar.add_state('h', 'h_dot')
lunar.add_state('θ', 'θ_dot')
lunar.add_state('h_dot', 'a * sin(β) - g')
lunar.add_state('θ_dot', '(a * cos(β) + h_dot * θ_dot) / (r_m + h)')

lunar.add_control('β')

lunar.add_constant('a', 3 * G)
lunar.add_constant('g', G)
lunar.add_constant('r_m', R_M)

lunar.add_constant('h_0', 0)
lunar.add_constant('θ_0', 0)
lunar.add_constant('h_dot_0', 0)
lunar.add_constant('θ_dot_0', 0)

lunar.add_constant('h_f', G * T_GUESS ** 2)
lunar.add_constant('h_dot_f', G * T_GUESS)
lunar.add_constant('θ_dot_f', G * T_GUESS / R_M)

lunar.set_cost('0', '0', 't')

lunar.add_constraint('initial', 't')
lunar.add_constraint('initial', 'h - h_0')
lunar.add_constraint('initial', 'θ - θ_0')
lunar.add_constraint('initial', 'h_dot - h_dot_0')
lunar.add_constraint('initial', 'θ_dot - θ_dot_0')

lunar.add_constraint('terminal', 'h - h_f')
lunar.add_constraint('terminal', 'h_dot - h_dot_f')
lunar.add_constraint('terminal', 'θ_dot - θ_dot_f')

with giuseppe.utils.Timer(prefix='Complilation Time:'):
    sym_ocp = giuseppe.problems.SymOCP(lunar)
    sym_dual = giuseppe.problems.SymDual(sym_ocp)
    sym_bvp = giuseppe.problems.SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    comp_dual_ocp = giuseppe.problems.CompDualOCP(sym_bvp, use_jit_compile=True)
    num_solver = giuseppe.numeric_solvers.ScipySolveBVP(comp_dual_ocp, use_jit_compile=True, verbose=True)

guess = giuseppe.guess_generators.auto_propagate_guess(
        comp_dual_ocp, control=45/180*3.14159, t_span=T_GUESS, abs_tol=1e-6)
seed_sol = num_solver.solve(guess.k, guess)
sol_set = giuseppe.continuation.SolutionSet(sym_bvp, seed_sol)
cont = giuseppe.continuation.ContinuationHandler(sol_set)
cont.add_linear_series(15, {'h_f': 50_000, 'h_dot_f': 0, 'θ_dot_f': 5_780 / (R_M + 50_000)}, bisection=10)

with giuseppe.utils.Timer(prefix='Continuation Time:'):
    for series in cont.continuation_series:
        for k, last_sol in series:
            sol_i = num_solver.solve(k, last_sol)
            sol_set.append(sol_i)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
