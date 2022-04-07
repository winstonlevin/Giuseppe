import pickle

import numpy as np

import giuseppe
from giuseppe.io import InputOCP
from giuseppe.problems.dual import SymDual, SymDualOCP, CompDualOCP, DualSol
from giuseppe.problems.ocp import SymOCP
from giuseppe.numeric_solvers.bvp import ScipySolveBVP
from giuseppe.continuation import ContinuationHandler, SolutionSet
from giuseppe.utils import Timer

giuseppe.utils.complilation.JIT_COMPILE = True

ocp = InputOCP()

ocp.set_independent('t')

ocp.add_state('x', 'v*cos(theta)')
ocp.add_state('y', 'v*sin(theta)')
ocp.add_state('v', '-g*sin(theta)')

ocp.add_control('theta')

ocp.add_constant('g', 32.2)

ocp.add_constant('x_0', 0)
ocp.add_constant('y_0', 0)
ocp.add_constant('v_0', 1)

ocp.add_constant('x_f', 1)
ocp.add_constant('y_f', -1)

ocp.set_cost('0', '1', '0')

ocp.add_constraint('initial', 't')
ocp.add_constraint('initial', 'x - x_0')
ocp.add_constraint('initial', 'y - y_0')
ocp.add_constraint('initial', 'v - v_0')

ocp.add_constraint('terminal', 'x - x_f')
ocp.add_constraint('terminal', 'y - y_f')

with Timer(prefix='Complilation Time:'):
    sym_ocp = SymOCP(ocp)
    sym_dual = SymDual(sym_ocp)
    sym_bvp = SymDualOCP(sym_ocp, sym_dual, control_method='algebraic')
    comp_dual_ocp = CompDualOCP(sym_bvp, use_jit_compile=False)
    num_solver = ScipySolveBVP(comp_dual_ocp, use_jit_compile=False)

n = 2
t = np.linspace(0, 0.25, n)
x = np.linspace(np.array([0., 0., 1.]), np.array([1., 1., 8.]), n).T
lam = np.linspace(np.array([-0.1, -0.1, -0.1]), np.array([-0.1, -0.1, -0.1]), n).T
nu0 = np.array([-0.1, -0.1, -0.1, -0.1])
nuf = np.array([-0.1, -0.1])
k = sym_ocp.default_values

guess = num_solver.solve(k, DualSol(t=t, x=x, lam=lam, nu0=nu0, nuf=nuf, k=k))
sol_set = SolutionSet(sym_bvp, guess)
cont = ContinuationHandler(sol_set)
cont.add_linear_series(5, {'x_f': 30, 'y_f': -30}, bisection=True)

with Timer(prefix='Continuation Time:'):
    for series in cont.continuation_series:
        for k, guess in series:
            sol_i = num_solver.solve(k, guess)
            sol_set.append(sol_i)

with open('sol_set.data', 'wb') as file:
    pickle.dump(sol_set, file)
