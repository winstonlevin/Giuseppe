import os

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
xf[2] = -np.pi/2

# Outer solution
dp = xf[:2] - x0[:2]
p_hat = dp / np.linalg.norm(dp)

psi_outer = np.arctan2(dp[1], dp[0])
u_outer = np.array((0.,))
lam_outer = np.array((-p_hat[0], -p_hat[1], 0.))

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

k = 3
intercept.add_constant('k', k)
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
sol_set = cont.run_continuation()
sol_set.save('sol_set.data')

# -------------------------------------------------------------------------------------------------------------------- #
# CUSTOM SOLUTION                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
# Derivation of necessary conditions
x_sym = ca.SX.sym('x', 3)
lam_sym = ca.SX.sym('lam', 3)
u_sym = ca.SX.sym('u', 1)
y_sym = ca.vcat((x_sym, lam_sym, u_sym))
nx = x_sym.shape[0]
nlam = nx
nu = u_sym.shape[0]

fx_sym = ca.vcat((ca.cos(x_sym[2]), ca.sin(x_sym[2]), u_sym))
path_cost_sym = 1 + k/2 * (u_sym*u_sym)
ham_sym = path_cost_sym + ca.dot(lam_sym, fx_sym)
hu_sym = ca.jacobian(ham_sym, u_sym)
flam_sym = ca.jacobian(ham_sym, x_sym).T

h_fun_ca = ca.Function('H', (y_sym,), (ham_sym,), ('y',), ('H',))
hu_fun_ca = ca.Function('Hu', (y_sym,), (hu_sym,), ('y',), ('Hu',))
fxlam_fun_ca = ca.Function('fxlam', (y_sym,), (ca.vcat((fx_sym, flam_sym)),), ('y',), ('fxlam',))

# Solution on same mesh as SciPy solution
sol = sol_set[-1]

X_sym = ca.SX.sym('X', sol.x.shape)
L_sym = ca.SX.sym('Lam', sol.lam.shape)
U_sym = ca.SX.sym('U', sol.u.shape)
Y_sym = ca.vcat((X_sym, L_sym, U_sym))
tf_sym = ca.SX.sym('tf')


def mesh_error(_y0, _y1, _h):
    # Trapezoidal Rule
    _fxlam0 = fxlam_fun_ca(_y0)
    _fxlam1 = fxlam_fun_ca(_y0)
    int_error = (_y1[:-nu] - _y0[:-nu]) - _h * (_fxlam0 + _fxlam1)

    # Control Law
    _hu0 = hu_fun_ca(_y0)

    return ca.vcat((int_error, _hu0))


# TODO - generate residual and solve
