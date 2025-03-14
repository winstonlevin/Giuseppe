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
    else:
        raise ValueError(
            f'col_method="{col_method}" is not implemented! Valid options are:\n'
            f'\t["lgl", "lg", "lgr", "flgr", "cg"]'
        )

    # Residual, final value, and collocation time functions
    res_fun = ca.Function('rcol', (ycol, y0, tf), (ycol_dot - tf/2*dynamic_fun(ycol),), ('yCol', 'y0', 'tf'), ('rcol',))
    yf_fun = ca.Function('yf', (ycol, y0, tf), (yf,), ('yCol', 'y0', 'tf'), ('rcol',))
    tcol_fun = ca.Function('tcol', (tf,), ((col_points + 1)*tf/2,), ('tf',), ('tcol',))

    return res_fun, yf_fun, tcol_fun


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


def determine_linearization_radius(
        jac_fun: ca.Function, res_fun: ca.Function, z0: np.ndarray,
        r_upper: float = 10., confidence: float = 0.95,
        max_iter: int = 100, tol: float = 1E-3, n_samples: int = 100, preprocess: Optional[Callable] = None
):
    """
    Determined the radius of convergence around the root z* via a binary search. The radius is
    determined by:
        1. Generate co-varied points about the optimal solution at distance R away from optimal solution
        2. The perturbed step vs the predicted step is:
                  z - z0   = R*dzhat ~= J^-1 (F(z) - F0)
                ||z - z0|| = R       ~= ||J^-1 (F(z) - F0)|| = Rhat
        3. The ``radius of convergence'' is defined as the location where:
                |E(Rhat) - R| == (1 - confidence)*R
           The expectation E(*) is calculated from the average  of random orthogonal samples
    NOTE: since a binary search is used, it is implicitly assumed that increasing R will lower the likelihood of
    convergence.

    Parameters
    ----------
    jac_fun, ca.Function, function of z to calculate Jacobian
    res_fun, ca.Function, function of z to calculate residual
    z0, np.ndarray, value about which to find the radius of convergence
    r_upper, float, default=10., maximum value checked for radius of convergence
    confidence, float, default=0.95, (1-confidence) is ratio of error norm to residual norm. Should be in range:
                                     0 < confidence < 1
    max_iter, int, default=100, maximum number of binary search iterations to check convergence
    tol, float, default=1E-3, tolerance for radius of convergence
    n_samples, int, default=100, number of randomly generated sample unit vectors at which to check error
    preprocess, Callable or None, default=None, if given, puts generated samples through preprocessor before checking
                                                the error

    Returns
    -------
    float, the radius of convergence value where, on average, ||e(z)|| == (1 - confidence) ||F(z)||
    """
    # Generate Jacobian (once)
    jac = jac_fun(z0).full()
    try:
        jac_inv = np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        # Jacobian is not invertible -> no radius of convergence
        return 0.

    res0 = res_fun(z0).full().ravel()
    convergent_error_fraction = 1. - confidence

    # Generate the unit vectors for the samples
    sample_unit_vectors = orthonormal_sampler(n=len(z0), n_samples=n_samples, rng_seed=rng_seed)

    if preprocess is None:
        def preprocess(_z, _dz):
            return _z + _dz

    def _estimate_radius(_z):
        """Rhat = ||J^-1 (F(z) - F0)||"""
        return np.linalg.norm(jac_inv.dot(res_fun(_z).full().ravel() - res0))

    def _mean_radius_error(_r):
        return np.sum(
            [_estimate_radius(preprocess(z0, _r * _sample)) for _sample in sample_unit_vectors]
        ) / n_samples - _r
        # for _idx_sample, _sample in enumerate(sample_unit_vectors):
        #     if _convergence_number(preprocess(z0 + _r * _sample)) > 0:
        #         # Move failed sample to front to speed up next check
        #         sample_unit_vectors.insert(0, sample_unit_vectors.pop(_idx_sample))
        #         return False
        # return True

    # Save list of prior values
    # |E(Rhat) - R| == (1 - confidence)*R
    error = _mean_radius_error(r_upper)
    if error > convergent_error_fraction * r_upper:
        return r_upper  # Uppermost value is inside radius of convergence

    r_lower = tol
    error = _mean_radius_error(r_lower)
    if error > convergent_error_fraction * r_lower:
        return 0.  # Even at tolerance, outside radius of convergence

    # Conduct binary search to determine where r stops converging
    r = (r_lower + r_upper) / 2
    for iteration in range(max_iter):
        # Update R value
        error = _mean_radius_error(r)
        if error > convergent_error_fraction * r:
            # This value is outside the radius of convergence
            r_upper = r
        else:
            # This value is inside the radius of convergence
            r_lower = r
        r = (r_lower + r_upper) / 2

        # Termination criteria
        if abs(r_lower - r_upper) < tol or abs(error) < tol*r:
            break

    return r

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
cols_try = np.arange(2, 20+1, 1)
collocation_method = 'lg'
integration_scheme = 'pseudospectral'
fixed_final_time = False  # True -> estimate y(tf). False -> estimate tf(yf)
use_log_tf = True  # True -> replace tf with log(tf) in unknown vector
rng_seed = 10

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

sol_dicts: list[dict] = []
cols_to_dict = {}
r_max = 100  # Upper bound to give up search for radius of convergence
tf_min = 1/r_max  # Lower bound on final time (in random sample generation)
for idx, num_col in enumerate(cols_try):
    cols_to_dict[num_col] = idx  # Get dict value (to get sol idx from number of collocation points)

    # True value
    rcol_fun, yf_fun, tcol_fun = generator(num_col)
    tcol = tcol_fun(tf).full()[:, 0]
    ycol = true_state_equation(tcol).full()[:, 0]
    ycol_sym = ca.SX.sym('ycol', num_col)

    if fixed_final_time:
        # Root (fixed terminal time)
        z_sym = ycol_sym
        z_true = ycol
        yf_sym = yf_fun(ycol_sym, y0, tf)
        res_sym = rcol_fun(ycol_sym, y0, tf)
        solution_fun = ca.Function('s', (z_sym,), (tf, yf_sym, ycol_sym))
        _preprocess_guess = None
    else:
        # Root (free terminal time, fixed terminal state)
        if use_log_tf:
            log_tf_sym = ca.SX.sym('log_tf')
            z_sym = ca.vcat((log_tf_sym, ycol_sym))
            tf_sym = np.exp(log_tf_sym)

            def _preprocess_guess(_z0, _dz):
                _z = _z0 + _dz
                _z[0] = np.exp(_z0[0]) + _dz[0]  # Convert logt0 + dt to t0 + dt
                np.maximum(_z[0], tf_min, out=_z[0:1])
                _z[0] = np.log(_z[0], out=_z[0:1])
                return _z

        else:
            tf_sym = ca.SX.sym('tf')
            z_sym = ca.vcat((tf_sym, ycol_sym))

            def _preprocess_guess(_z0, _dz):
                _z = _z0 + _dz
                np.maximum(_z[0], tf_min, out=_z[0:1])
                return _z

        z_true = np.concatenate(((tf,), ycol))
        yf_sym = yf_fun(ycol_sym, y0, tf_sym)
        res_sym = ca.vcat((yf_sym - yf, rcol_fun(ycol_sym, y0, tf_sym)))
        solution_fun = ca.Function('s', (z_sym,), (tf_sym, yf_sym, ycol_sym))

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

    # Determine radius of convergence numerically
    err_root_sol = np.dot(sol_root.fun, sol_root.fun)
    success = err_root_sol < 1E-3
    if success:
        rlin = determine_linearization_radius(jac_fun, res_fun, sol_root.x, preprocess=_preprocess_guess)
    else:
        rlin = 0.

    sol_dicts.append({
        'n': num_col,
        'tcol': np.append(tcol, tf),
        'ycol': np.append(ycol, yf),
        'tcol_hat': np.append(tcol_hat, tf_hat),
        'ycol_hat': np.append(ycol_hat, yf_hat),
        'etcol': np.append(etcol, etf),
        'ecol': np.append(ecol, ef),
        'success': success,
        'rlin': rlin
    })

# # Experiment with radius of converged, TODO - remove
# tolerance_for_near = 1E-3 + np.dot(ycol_sol.fun, ycol_sol.fun)
#
#
# def _rfp(_z0):
#     _sol = optimize.root(lambda _ycol: res_fun(_ycol, y0, tf).full()[:, 0], _z0, jac=jac_fun, method='hybr')
#     _err = _sol.x - ycol_sol.x
#     return np.dot(_err, _err) < tolerance_for_near
#
#
# interp = determine_radius_of_convergence(_rfp, ycol_sol.x, return_continuous=True, r_max=1_000)
#
# fig_rcov, ax_rcov = plt.subplots()
# ax_rcov.grid(zorder=-1)
# r_vals = np.linspace(interp.x[0], interp.x[-1], 1000)
# ax_rcov.plot(r_vals, 100*interp(r_vals))
# ax_rcov.plot(interp.x, 100*interp(interp.x), 'o')
# ax_rcov.set_xlabel('r')
# fig_rcov.tight_layout()

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
        f'PS Approximation [{_sol_dict["n"]} col. pts, {convergence_str}, ef={_sol_dict["ecol"][-1]:.2e}]'
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
rconv_vals = []
ncol_vals = []
ef_vals = []
for _sol_dict in sol_dicts:
    rconv_vals.append(_sol_dict['rconv'])
    ncol_vals.append(_sol_dict['n'])
    ef_vals.append(_sol_dict['ecol'][-1] if fixed_final_time else _sol_dict['etcol'][-1])
rconv_vals = np.array(rconv_vals)
ncol_vals = np.array(ncol_vals)
ef_vals = np.array(ef_vals)

idces = np.argsort(ncol_vals)
fig_col, axes_col = plt.subplots(nrows=2, sharex=True)

ax_ef = axes_col[0]
ax_ef.grid(zorder=-1)
ax_ef.plot(ncol_vals[idces], ef_vals[idces], 'o')
# ax_ef.set_xlabel('Num. Col. Pts.')
ax_ef.set_ylabel(err_lab)

ax_rconv = axes_col[1]
ax_rconv.grid(zorder=-1)
ax_rconv.plot(ncol_vals[idces], rconv_vals[idces], 'o')
ax_rconv.set_xlabel('Num. Col. Pts.')
ax_rconv.set_ylabel('Radius of Convergence')
ax_rconv.set_xticks(np.unique(np.round(np.linspace(ncol_vals[0], ncol_vals[-1], 5))))
fig_col.tight_layout()
