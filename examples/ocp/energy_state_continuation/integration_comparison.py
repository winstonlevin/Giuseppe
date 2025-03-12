import numpy as np
import casadi as ca
from scipy import optimize, interpolate
from matplotlib import pyplot as plt

import giuseppe


# -------------------------------------------------------------------------------------------------------------------- #
# INTEGRATION METHODS                                                                                                  #
# -------------------------------------------------------------------------------------------------------------------- #
def generate_pseudospectral_residual(dynamic_fun: ca.Function, n: int, col_method: str = 'lgl'):
    """
    Generates residual associated with n collocation points, returning a Tuple of:
    (residual_function, terminal_state_function)
    The inputs to the residual function are the n collocation states and final time.
    """
    y0 = ca.SX.sym('y0')
    ycol = ca.SX.sym('y', n)  # n collocation points
    tf = ca.SX.sym('tf')

    # Generate nondimensional collocation points
    _collocation_method = col_method.lower()
    if _collocation_method == 'lgl':
        # Legendre-Gauss-Lobatto Quadrature
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lgl(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        int_vec = col_weights @ diff_mat
        col_points = col_points[1:]
        diff_mat = diff_mat[1:, :]  # Initial diff is for y0 -> no residual
        y_vals = ca.vcat((y0, ycol))
    elif _collocation_method == 'lg':
        # Legendre-Gauss Quadrature
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lg(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points[1:], compute_interp_matrix=False, compute_diff_matrix=True
        )  # Initial diff is for y0 -> no residual
        col_points = col_points[1:]
        int_vec = col_weights @ diff_mat
        y_vals = ca.vcat((y0, ycol))
    elif _collocation_method == 'lgr':
        # Legendre-Gauss-Radau Quadrature
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lgr(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        int_vec = col_weights @ diff_mat
        col_points = col_points[1:]
        diff_mat = diff_mat[1:, :]  # Initial diff is for y0 -> no residual
        y_vals = ca.vcat((y0, ycol))
    elif _collocation_method == 'lgrr':
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lgr(n)
        col_points = -col_points[::-1]
        col_weights = col_weights[::-1]
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        int_vec = col_weights @ diff_mat
        y_vals = ycol
    elif _collocation_method == 'cg':
        (col_points, col_weights) = giuseppe.utils.pseudospectral.cg(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        int_vec = col_weights @ diff_mat
        col_points = col_points[1:]
        diff_mat = diff_mat[1:, :]  # Initial diff is for y0 -> no residual
        y_vals = ca.vcat((y0, ycol))
    else:
        raise ValueError(
            f'col_method="{col_method}" is not implemented! Valid options are:\n'
            f'\t["lgl", "lg", "lgr", "lgrr", "cg"]'
        )

    # Residual function
    res_sym = diff_mat @ y_vals - tf/2*dynamic_fun(ycol)
    res_fun = ca.Function('rcol', (ycol, y0, tf), (res_sym,), ('yCol', 'y0', 'tf'), ('rcol',))

    yf_sym = ca.dot(int_vec, y_vals)
    yf_fun = ca.Function('yf', (ycol, y0, tf), (yf_sym,), ('yCol', 'y0', 'tf'), ('rcol',))
    tcol_fun = ca.Function('tcol', (tf,), ((col_points + 1)*tf/2,), ('tf',), ('tcol',))

    return res_fun, yf_fun, tcol_fun


# -------------------------------------------------------------------------------------------------------------------- #
# NUMERICAL EXAMPLES                                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
# Problem definition ------------------------------------------------------------------------------------------------- #
# "Well-scaled" time and state: 0 <= time <= 1, -1 <= state <= 1
tf = 10.
y0 = -1.
yf = 1

t_sym = ca.SX.sym('t')
y_sym = ca.SX.sym('y')
function_type = 'smooth'

if function_type == 'smooth':
    exp_tf = np.exp(tf)
    c1 = (yf - y0*exp_tf) / (1 - exp_tf)
    c0 = (y0 - c1)/y0
    equations_of_motion = ca.Function('f', (y_sym,), (y_sym - c1,), ('y',), ('f',))
    true_state_equation = ca.Function('Phi', (t_sym,), (c0*y0*ca.exp(t_sym) + c1,), ('t',), ('y',))
else:
    raise ValueError(
        f'function_type "{function_type}" is not implemented! Valid options are:\n'
        f'["smooth"]'
    )

# Integration scheme ------------------------------------------------------------------------------------------------- #
num_col = 8
collocation_method = 'lg'
integration_scheme = 'pseudospectral'

if integration_scheme == 'pseudospectral':
    def generator(_num_col):
        return generate_pseudospectral_residual(
            equations_of_motion, n=_num_col, col_method=collocation_method
        )
else:
    raise ValueError(
        f'integration_scheme="{function_type}" is not implemented! Valid options are:\n'
        f'["pseudospectral"]'
    )

# True value
res_1, yf_1, tcol_fun = generator(num_col)
tcol_1 = tcol_fun(tf).full()[:, 0]
ycol_1 = true_state_equation(tcol_1)

# Root
ycol_sol = optimize.root(lambda _ycol: res_1(_ycol, y0, tf).full()[:, 0], ycol_1)
ycol_1_hat = ycol_sol.x

# -------------------------------------------------------------------------------------------------------------------- #
# PLOTTING                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #
t_vals = np.linspace(0, tf, 1_000)
y_vals = true_state_equation(t_vals).full()[:, 0]

# Plot state vs. time
fig_y, ax_y = plt.subplots()
ax_y.grid(zorder=-1)
ax_y.plot(t_vals, y_vals, label='True')
ax_y.scatter(tcol_1, ycol_1_hat, label='Pseudospectral')
ax_y.set_xlabel(r'$t$')
ax_y.set_ylabel(r'$y(t)$')
