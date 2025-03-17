from typing import Optional, Callable
import math

import numpy as np
from scipy.sparse.linalg import splu
import casadi as ca
from scipy import optimize, interpolate
from matplotlib import pyplot as plt, widgets

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
    y_vals = ca.vcat((y0, ycol))

    # Generate nondimensional collocation points
    _collocation_method = col_method.lower()
    if _collocation_method == 'lgl':
        # Legendre-Gauss-Lobatto Quadrature
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lgl(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        yf = y0 + ca.dot(np.dot(col_weights, diff_mat), y_vals)
        ycol_dot = diff_mat[1:, :] @ y_vals
        col_points = col_points[1:]
    elif _collocation_method == 'lg':
        # Legendre-Gauss Quadrature
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lg(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points[1:], compute_interp_matrix=False, compute_diff_matrix=True
        )  # Initial diff is for y0 -> no residual
        yf = y0 + ca.dot(np.dot(col_weights, diff_mat), y_vals)
        ycol_dot = diff_mat @ y_vals
        col_points = col_points[1:]
    elif _collocation_method == 'lgr':
        # Legendre-Gauss-Radau Quadrature
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lgr(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        yf = y0 + ca.dot(np.dot(col_weights, diff_mat), y_vals)
        ycol_dot = diff_mat[1:, :] @ y_vals
        col_points = col_points[1:]
    elif _collocation_method == 'flgr':
        (col_points, col_weights) = giuseppe.utils.pseudospectral.lgr(n)
        col_points = -col_points[::-1]
        col_weights = col_weights[::-1]
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            np.concatenate(((-1.,), col_points)), col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        diff0 = diff_mat[:, 0]
        diff_mat = diff_mat[:, 1:]
        ycol_dot = diff0*y0 + diff_mat @ ycol
        yf = y0 + ca.dot(col_weights, ycol_dot)
    elif _collocation_method == 'cg':
        (col_points, col_weights) = giuseppe.utils.pseudospectral.cg(n+1)
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        yf = ca.dot(np.dot(col_weights, diff_mat), ca.vcat((y0, ycol)))
        ycol_dot = diff_mat[1:, :] @ ca.vcat((y0, ycol))
        col_points = col_points[1:]
    elif _collocation_method == 'unif':
        col_points = np.linspace(-1., 1., n+1)
        col_weights = giuseppe.utils.pseudospectral.integration_matrix(col_points, np.ones(shape=(1,))).ravel()
        _, diff_mat = giuseppe.utils.pseudospectral.lagrange_matrices(
            col_points, col_points, compute_interp_matrix=False, compute_diff_matrix=True
        )
        yf = y0 + ca.dot(np.dot(col_weights, diff_mat), y_vals)
        ycol_dot = diff_mat[1:, :] @ y_vals
        col_points = col_points[1:]
    else:
        raise ValueError(
            f'col_method="{col_method}" is not implemented! Valid options are:\n'
            f'\t["lgl", "lg", "lgr", "flgr", "cg", "unif"]'
        )

    # Residual, final value, and collocation time functions
    res_fun = ca.Function('rcol', (ycol, y0, tf), (ycol_dot - tf/2*dynamic_fun(ycol),), ('yCol', 'y0', 'tf'), ('rcol',))
    yf_fun = ca.Function('yf', (ycol, y0, tf), (yf,), ('yCol', 'y0', 'tf'), ('rcol',))
    tcol_fun = ca.Function('tcol', (tf,), ((col_points + 1)*tf/2,), ('tf',), ('tcol',))
    ycol_fun = ca.Function('ycol', (y0, ycol, tf), (ycol,), ('y0', 'ycol', 'tf'), ('ycol',))

    return res_fun, yf_fun, tcol_fun, ycol_fun


def generate_collocation_derivative_residual(dynamic_fun: ca.Function, n: int, col_method: str = 'lgl'):
    """
    Generates residual associated with n collocation points, returning a Tuple of:
    (residual_function, terminal_state_function)
    The inputs to the residual function are the n collocation derivatives and final time.
    """
    y0 = ca.SX.sym('y0')
    ydotcol = ca.SX.sym('ydot', n)  # n collocation points
    tf = ca.SX.sym('tf')

    # Generate nondimensional collocation points
    _collocation_method = col_method.lower()
    if _collocation_method == 'lgl':
        # Legendre-Gauss-Lobatto Quadrature
        col_points, _ = giuseppe.utils.pseudospectral.lgl(n)
        int_mat = giuseppe.utils.pseudospectral.integration_matrix(col_points)
        ycol = y0 + tf/2*(int_mat @ ydotcol)
        yf = ycol[-1]
    elif _collocation_method == 'lg':
        # Legendre-Gauss Quadrature
        col_points, _ = giuseppe.utils.pseudospectral.lg(n+1)
        col_points = col_points[1:]
        int_mat = giuseppe.utils.pseudospectral.integration_matrix(col_points, np.append(col_points, 1.))
        ycol = y0 + tf/2*(int_mat[:-1, :] @ ydotcol)
        yf = y0 + tf/2*((int_mat[-1:, :] @ ydotcol))
    elif _collocation_method == 'lgr':
        # Legendre-Gauss-Radau Quadrature
        col_points, _ = giuseppe.utils.pseudospectral.lgr(n)
        int_mat = giuseppe.utils.pseudospectral.integration_matrix(col_points, np.append(col_points, 1.))
        ycol = y0 + int_mat[:-1, :] @ ydotcol
        yf = y0 + tf/2*((int_mat[-1:, :] @ ydotcol))
    elif _collocation_method == 'flgr':
        col_points, _ = giuseppe.utils.pseudospectral.lgr(n)
        col_points = -col_points[::-1]
        int_mat = giuseppe.utils.pseudospectral.integration_matrix(col_points)
        ycol = y0 + tf/2*(int_mat @ ydotcol)
        yf = ycol[-1]
    elif _collocation_method == 'cg':
        col_points, _ = giuseppe.utils.pseudospectral.cg(n+2)
        col_points = col_points[1:-1]
        int_mat = giuseppe.utils.pseudospectral.integration_matrix(col_points, np.append(col_points, 1.))
        ycol = y0 + tf/2*(int_mat[:-1, :] @ ydotcol)
        yf = y0 + tf/2*(int_mat[-1:, :] @ ydotcol)
    elif _collocation_method == 'unif':
        col_points = np.linspace(-1., 1., n)
        int_mat = giuseppe.utils.pseudospectral.integration_matrix(col_points)
        ycol = y0 + tf/2*(int_mat @ ydotcol)
        yf = ycol[-1]
    else:
        raise ValueError(
            f'col_method="{col_method}" is not implemented! Valid options are:\n'
            f'\t["lgl", "lg", "lgr", "flgr", "cg", "unif"]'
        )

    # Residual, final value, and collocation time functions
    res_fun = ca.Function('rcol', (ydotcol, y0, tf), (ydotcol - dynamic_fun(ycol),), ('ydotCol', 'y0', 'tf'), ('rcol',))
    yf_fun = ca.Function('yf', (ydotcol, y0, tf), (yf,), ('ydotCol', 'y0', 'tf'), ('rcol',))
    tcol_fun = ca.Function('tcol', (tf,), ((col_points + 1)*tf/2,), ('tf',), ('tcol',))
    ycol_fun = ca.Function('ycol', (y0, ydotcol, tf), (ycol,), ('y0', 'ydotcol', 'tf'), ('ycol',))

    return res_fun, yf_fun, tcol_fun, ycol_fun


def generate_simpson_residual(dynamic_fun: ca.Function, n: int, col_method: str = 'lgl'):
    """
    Implicitly integrate between start/end point using the Simpson scheme
    """
    y0 = ca.SX.sym('y0')
    ycol = ca.SX.sym('ycol', n)
    yvals = ca.vcat((y0, ycol))
    tf = ca.SX.sym('tf')

    # Generate nondimensional fixed mesh
    _collocation_method = col_method.lower()
    if _collocation_method == 'lgl':
        # Legendre-Gauss-Lobatto Quadrature
        col_points, _ = giuseppe.utils.pseudospectral.lgl(n + 1)
    elif _collocation_method == 'lg':
        # Legendre-Gauss Quadrature
        col_points, _ = giuseppe.utils.pseudospectral.lg(n)
        col_points = np.append(col_points, 1.)
    elif _collocation_method == 'lgr':
        # Legendre-Gauss-Radau Quadrature
        col_points, _ = giuseppe.utils.pseudospectral.lgr(n)
        col_points = np.append(col_points, 1.)
    elif _collocation_method == 'flgr':
        col_points, _ = giuseppe.utils.pseudospectral.lgr(n)
        col_points = np.concatenate(((-1.,), -col_points[::-1]))
    elif _collocation_method == 'cg':
        col_points, _ = giuseppe.utils.pseudospectral.cg(n + 1)
    elif _collocation_method == 'unif':
        col_points = np.linspace(-1., 1., n+1)
    else:
        raise ValueError(
            f'col_method="{col_method}" is not implemented! Valid options are:\n'
            f'\t["lgl", "lg", "lgr", "flgr", "cg", "unif"]'
        )

    # Convert to changes in time
    dt = np.diff(col_points) * tf/2

    # Generate "middle" point
    fvals = dynamic_fun(yvals)
    y_middle = (yvals[:-1] + yvals[1:])/2 + dt/8 * (fvals[:-1] - fvals[1:])
    f_middle = dynamic_fun(y_middle)
    col_res = (yvals[1:] - yvals[:-1]) - dt/6*(fvals[:-1] + 4*f_middle + fvals[1:])
    yf = yvals[-1]

    # Residual, final value, and collocation time functions
    res_fun = ca.Function('rcol', (ycol, y0, tf), (col_res,), ('yCol', 'y0', 'tf'), ('rcol',))
    yf_fun = ca.Function('yf', (ycol, y0, tf), (yf,), ('yCol', 'y0', 'tf'), ('rcol',))
    tcol_fun = ca.Function('tcol', (tf,), ((col_points[1:] + 1)*tf/2,), ('tf',), ('tcol',))
    ycol_fun = ca.Function('ycol', (y0, ycol, tf), (ycol,), ('y0', 'ycol', 'tf'), ('ycol',))

    return res_fun, yf_fun, tcol_fun, ycol_fun


def orthonormal_sampler(n: int, n_samples: Optional[int] = None, rng_seed=None):
    """
    Generate m sets of n orthogonal vectors of length n, randomly generated using the rng_seed. Each set of n
    orthogonal values is taken from the QR decomposition of the matrix H whose elements are normally distributed.
    This algorithm is adapted from:
    https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy

    Parameters
    ----------
    n, int, length of each sample vector
    n_samples, int, default=n, number of sample vectors to be generated. Each set of n is orthogonal.
    rng_seed, int or None, default=None, seed used in NumPy's i.i.d. standard normal generator

    Returns
    -------
    list of samples
    """
    _generator = np.random.default_rng(seed=rng_seed)
    _q_matrices = []
    _samples = []
    _sign_mult = -1
    m = np.ceil(n_samples / n).astype(int)
    for _multiplicity_index in range(m):
        _normal_matrix = _generator.random(size=(n, n))
        _q, _r = np.linalg.qr(_normal_matrix, mode='complete')
        _q = _q @ np.diag(np.sign(np.diag(_r)))

        # Multiply every other matrix by -1. For whatever reason, I am not seeming to get typical sign flips for my
        # data. Multiplying by -1 should not affect distribution of rotation angles, since they are still randomly
        # between [-pi, pi].
        _q *= _sign_mult
        _sign_mult *= -1

        _samples.extend(_q)
    return _samples[:n_samples]


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
cols_try = np.arange(2, 15+1, 1)
collocation_method = 'unif'
# integration_scheme = 'pseudospectral'
# integration_scheme = 'collocation'
integration_scheme = 'simpson'
fixed_final_time = True  # True -> estimate y(tf). False -> estimate tf(yf)
use_log_tf = True  # True -> replace tf with log(tf) in unknown vector
rng_seed = 10

if integration_scheme == 'pseudospectral':
    def generator(_num_col):
        return generate_pseudospectral_residual(
            equations_of_motion, n=_num_col, col_method=collocation_method
        )
    method_str = 'Pseudospectral (' + collocation_method.upper() + ')'
elif integration_scheme == 'collocation':
    def generator(_num_col):
        return generate_collocation_derivative_residual(equations_of_motion, n=_num_col, col_method=collocation_method)
    method_str = 'Collocation (' + collocation_method.upper() + ')'
elif integration_scheme == 'simpson':
    def generator(_num_col):
        return generate_simpson_residual(equations_of_motion, n=_num_col, col_method=collocation_method)
    method_str = 'Simpson (' + collocation_method.upper() + ')'
else:
    raise ValueError(
        f'integration_scheme="{function_type}" is not implemented! Valid options are:\n'
        f'["pseudospectral"]'
    )

sol_dicts: list[dict] = []
cols_to_dict = {}
r_max = 100  # Upper bound to give up search for radius of convergence
tf_min = 1/r_max  # Lower bound on final time (in random sample generation)
for idx, num_col in enumerate(cols_try):
    cols_to_dict[num_col] = idx  # Get dict value (to get sol idx from number of collocation points)

    # True value
    rcol_fun, yf_fun, tcol_fun, ycol_fun = generator(num_col)
    tcol = tcol_fun(tf).full()[:, 0]
    ycol = true_state_equation(tcol).full()[:, 0]
    ydotcol = equations_of_motion(ycol).full()[:, 0]
    ycol_sym = ca.SX.sym('ycol', num_col)

    if fixed_final_time:
        # Root (fixed terminal time)
        z_sym = ycol_sym
        if integration_scheme == 'collocation':
            z_true = ydotcol
        else:
            z_true = ycol
        yf_sym = yf_fun(ycol_sym, y0, tf)
        res_sym = rcol_fun(ycol_sym, y0, tf)
        solution_fun = ca.Function('s', (z_sym,), (tf, yf_sym, ycol_fun(y0, ycol_sym, tf)))
    else:
        # Root (free terminal time, fixed terminal state)
        if use_log_tf:
            log_tf_sym = ca.SX.sym('log_tf')
            z_sym = ca.vcat((log_tf_sym, ycol_sym))
            tf_sym = np.exp(log_tf_sym)
        else:
            tf_sym = ca.SX.sym('tf')
            z_sym = ca.vcat((tf_sym, ycol_sym))

        z_true = np.concatenate(((tf,), ycol))
        yf_sym = yf_fun(ycol_sym, y0, tf_sym)
        res_sym = ca.vcat((yf_sym - yf, rcol_fun(ycol_sym, y0, tf_sym)))
        solution_fun = ca.Function('s', (z_sym,), (tf_sym, yf_sym, ycol_fun(y0, ycol_sym, tf_sym)))

    jac_sym = ca.jacobian(res_sym, z_sym)
    res_fun = ca.Function('r', (z_sym,), (res_sym,))

    def res_fun_wrapped(_z):
        return res_fun(_z).full()[:, 0]

    jac_fun = ca.Function('J', (z_sym,), (jac_sym,), ('z',), ('J',))

    sol_root = optimize.root(res_fun_wrapped, z_true, jac=jac_fun, method='hybr')
    tf_hat, yf_hat, ycol_hat = solution_fun(sol_root.x)
    tf_hat = float(tf_hat)  # Convert to non-CasADi type
    yf_hat = float(yf_hat)
    ycol_hat = ycol_hat.full().ravel()
    tcol_hat = tcol_fun(tf_hat).full().ravel()

    # Errors
    root_error = sol_root.x - z_true
    norm_root_err = np.dot(root_error, root_error)**0.5

    ef = yf_hat - yf
    ecol = ycol_hat - ycol
    etcol = tcol_hat - tcol
    etf = tf_hat - tf

    err_root_sol = np.dot(sol_root.fun, sol_root.fun)
    success = err_root_sol < 1E-3

    sol_dicts.append({
        'n': num_col,
        'tcol': np.append(tcol, tf),
        'ycol': np.append(ycol, yf),
        'tcol_hat': np.append(tcol_hat, tf_hat),
        'ycol_hat': np.append(ycol_hat, yf_hat),
        'etcol': np.append(etcol, etf),
        'ecol': np.append(ecol, ef),
        'success': success,
    })

# -------------------------------------------------------------------------------------------------------------------- #
# PLOTTING                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #
t_vals = np.linspace(0, tf, 1_000)
y_vals = true_state_equation(t_vals).full()[:, 0]
y_min = y_vals.min(initial=np.inf)
y_max = y_vals.max(initial=-np.inf)
plot_buffer = 0.2

# State vs. time plot ------------------------------------------------------------------------------------------------ #
fig_y, ax_y = plt.subplots()
ax_y.grid(zorder=-1)
ax_y.plot(t_vals, y_vals, label='True')
ycol_plot = ax_y.plot([], [], 'o', label='Pseudospectral')[0]
ax_y.set_xlabel(r'$t$')
ax_y.set_ylabel(r'$y(t)$')
ax_y.set_xlim((-plot_buffer, tf + plot_buffer))


def set_plot(_n_col):
    _n_col = np.round(_n_col).astype(int)
    _sol_dict = sol_dicts[cols_to_dict[_n_col]]
    ycol_plot.set_data(_sol_dict['tcol_hat'], _sol_dict['ycol_hat'])
    if _sol_dict["success"]:
        convergence_str = "SUCCESS"
    else:
        convergence_str = "FAILURE"
    ax_y.set_title(
        method_str + f' [{_sol_dict["n"]} col. pts, {convergence_str}, ef={_sol_dict["ecol"][-1]:.2e}]'
    )
    ax_y.set_ylim((
        min(_sol_dict['ycol_hat'].min(), y_min) - plot_buffer,
        max(_sol_dict['ycol_hat'].max(), y_max) + plot_buffer
    ))


fig_y.subplots_adjust(bottom=0.25)
ax_ncol = fig_y.add_axes([0.25, 0.1, 0.65, 0.03])
slider_ncol = widgets.Slider(
    ax=ax_ncol,
    label='Num. Col. Pts.',
    valmin=cols_try[0],
    valmax=cols_try[-1],
    valinit=cols_try[-1],
    valstep=1,
)
slider_ncol.on_changed(set_plot)
set_plot(cols_try[-1])

# Radius of convergence plot ----------------------------------------------------------------------------------------- #
err_lab = 'yf Err.' if fixed_final_time else 'tf Err.'
ncol_vals = []
ef_vals = []
for _sol_dict in sol_dicts:
    ncol_vals.append(_sol_dict['n'])
    ef_vals.append(_sol_dict['ecol'][-1] if fixed_final_time else _sol_dict['etcol'][-1])

ncol_vals = np.array(ncol_vals)
ef_vals = np.array(ef_vals)

idces = np.argsort(ncol_vals)
fig_col, ax_ef = plt.subplots()
ax_ef.grid(zorder=-1)
ax_ef.plot(ncol_vals[idces], ef_vals[idces], 'o')
ax_ef.set_xlabel('Num. Col. Pts.')
ax_ef.set_xticks(np.unique(np.round(np.linspace(ncol_vals[0], ncol_vals[-1], 10)).astype('int')))
ax_ef.set_ylabel(err_lab)
ax_ef.set_title(method_str)

fig_col.tight_layout()
