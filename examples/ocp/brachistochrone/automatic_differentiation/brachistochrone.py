import numpy as np
import casadi as ca

from giuseppe.continuation import ContinuationHandler
from giuseppe.guess_generation import auto_guess
from giuseppe.numeric_solvers import SciPySolver
from giuseppe.problems.input import ADiffInputProb
from giuseppe.problems.automatic_differentiation import ADiffDual
from giuseppe.utils import Timer

input_ocp = ADiffInputProb()

# Independent Variable
t = ca.SX.sym('t', 1)
input_ocp.set_independent(t)

# Control
theta = ca.SX.sym('θ', 1)
input_ocp.add_control(theta)

# Known Constant Parameters
g = ca.SX.sym('g', 1)
input_ocp.add_constant(g, 32.2)

# States
x = ca.SX.sym('x', 1)
y = ca.SX.sym('y', 1)
v = ca.SX.sym('v', 1)

input_ocp.add_state(x, v * ca.cos(theta))
input_ocp.add_state(y, v * ca.sin(theta))
input_ocp.add_state(v, -g * ca.sin(theta))

# Boundary Conditions
x_0 = ca.SX.sym('x_0', 1)
y_0 = ca.SX.sym('y_0', 1)
v_0 = ca.SX.sym('v_0', 1)
input_ocp.add_constant(x_0, 0)
input_ocp.add_constant(y_0, 0)
input_ocp.add_constant(v_0, 1)

x_f = ca.SX.sym('x_f', 1)
y_f = ca.SX.sym('y_f', 1)
input_ocp.add_constant(x_f, 1)
input_ocp.add_constant(y_f, -1)

input_ocp.set_cost(0, 0, t)

input_ocp.add_constraint('initial', t)
input_ocp.add_constraint('initial', x - x_0)
input_ocp.add_constraint('initial', y - y_0)
input_ocp.add_constraint('initial', v - v_0)

input_ocp.add_constraint('terminal', x - x_f)
input_ocp.add_constraint('terminal', y - y_f)

with Timer(prefix='Compilation Time:'):
    adiff_dual = ADiffDual(input_ocp)
    solver = SciPySolver(adiff_dual)

guess = auto_guess(adiff_dual, u=-15 / 180 * 3.14159)

cont = ContinuationHandler(solver, guess)
cont.add_linear_series(5, {'x_f': 30, 'y_f': -30})
sol_set = cont.run_continuation()

sol_set.save('sol_set.data')

