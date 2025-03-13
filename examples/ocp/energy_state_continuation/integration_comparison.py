import numpy as np
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


def orthonormal_sampler(n: int, m: int, rng_seed=None):
    """
    Generate m sets of n orthogonal vectors of length n, randomly generated using the _rng_seed. Each set of n
    orthogonal values is taken from the QR decomposition of the matrix H whose elements are normally distributed.
    This algorithm is adapted from:
    https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy

    Parameters
    ----------
    n, int, length of each sample vector
    m, int, number of sets of n orthogonal vectors to be generated
    rng_seed, int or None, default=None, seed used in NumPy's i.i.d. standard normal generator

    Returns
    -------
    list of samples
    """
    _generator = np.random.default_rng(seed=rng_seed)
    _q_matrices = []
    _samples = []

    for _multiplicity_index in range(m):
        _normal_matrix = _generator.random(size=(n, n))
        _q, _r = np.linalg.qr(_normal_matrix, mode='complete')
        _q = _q @ np.diag(np.sign(np.diag(_r)))
        _samples.extend(_q)

    return _samples


def determine_radius_of_convergence(
        root_finding_problem, sol_converged, multiplicity: int = 2, confidence: int = 0.99,
        r_max: float = 100., return_continuous=False, rng_seed=None, max_iter: int=1_000, tol: float = 1E-3
):
    """
    Determined the radius of convergence around the root z* via a binary search. The validity of a given radius is
    determined by:
        1. Generate co-varied points about the optimal solution at distance R away from optimal solution
        2. Try to solve the root-finding problem with each guess. If the fraction of passed results are at least equal
        to ``_confidence'', then the result is considered a pass.
    NOTE: since a binary search is used, it is implicitly assumed that increasing R will lower the likelihood of
    convergence.

    Parameters
    ----------
    root_finding_problem, Callable, function taking in the initial guess and outputting a boolean flag of success.
    sol_converged, (n,) np.ndarray, root of the root-finding problem about which to find the radius of convergence.
    multiplicity, int, default=2, multiplicity*len(sol_converged) permutations are used to perturb samples.
    confidence, float, default=0.99, fraction of values required to pass to be considered inside radius of convergence.
                (ignored if return-continuous=True)
    return_continuous, bool, default=False, whether to return a single r value or an interpolant of confidence vs. r
    rng_seed, int or None, default=None, seed to be used in sampling perturbations to the guess vector
    max_iter, int, default=1_000, maximum number of search steps to find radius of convergence
    tol, float, default=1E-3, tolerance for binary search (break when upper/lower are within this value of each other)

    Returns
    -------
    if _return_continuous, returns an interpolant of the fraction of successful results vs. radius of convergence.
    Otherwise, returns radius of convergence as a float.
    """
    # Generate the unit vectors for the samples
    sample_unit_vectors = orthonormal_sampler(n=len(sol_converged), m=multiplicity, rng_seed=rng_seed)

    def _find_num_converged(_r):
        return np.sum([root_finding_problem(sol_converged + _r * _sample) for _sample in sample_unit_vectors])

    # Save list of prior values
    r_values = []
    n_pass_values = []

    # Generate upper bound on radius of convergence
    if return_continuous:
        n_pass_min = 1
    else:
        n_pass_min = np.ceil(confidence * len(sample_unit_vectors)).astype(int)

    r_upper = 1.
    iterations_to_find_max = 0
    upper_bound_active = False
    for iterations_to_find_max in range(max_iter):
        r_values.append(r_upper)
        n_pass_values.append(_find_num_converged(r_upper))
        # max_iter -= 1
        if n_pass_values[-1] < n_pass_min:
            # An upper bound on the failure rate is found
            break
        else:
            # Continue increasing radius of convergence until breaking value is found
            r_upper *= 2.

            # Maximum bound on radius of convergence was discovered. If this max bound converges, return the maximum
            if r_upper >= r_max:
                r_upper = r_max
                if upper_bound_active:
                    # Ensure we only check the upper bound once
                    break
                upper_bound_active = True

    if n_pass_values[-1] < n_pass_min:
        # Conduct binary search to determine where n_pass_min occurs
        r_lower = 0.
        for iteration in range(max_iter - iterations_to_find_max):
            r_values.append(0.5 * (r_lower + r_upper))
            n_pass_values.append(_find_num_converged(r_values[-1]))

            if n_pass_values[-1] < n_pass_min:
                # This value is outside the radius of convergence
                r_upper = r_values[-1]
            else:
                # This value is inside the radius of convergence
                r_lower = r_values[-1]

            if abs(r_lower - r_upper) < tol:
                break
    else:
        r_lower = r_values[-1]

    if return_continuous:
        r_values = np.array(r_values)
        pass_frac = np.array(n_pass_values) / len(sample_unit_vectors)
        idces = np.argsort(r_values)
        return interpolate.PchipInterpolator(r_values[idces], pass_frac[idces])
    else:
        # Return conservative bound
        return r_lower

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
    else:
        # Root (free terminal time, fixed terminal state)
        tf_sym = ca.SX.sym('tf')
        z_sym = ca.vcat((tf_sym, ycol_sym))
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
    ycol_hat = ycol_hat.full()[:, 0]
    tcol_hat = tcol_fun(tf_hat).full()[:, 0]

    # Errors
    root_error = sol_root.x - z_true
    norm_root_err = np.dot(root_error, root_error)**0.5

    ef = yf_hat - yf
    ecol = ycol_hat - ycol
    etcol = tcol_hat - tcol
    etf = tf_hat - tf

    # Determine radius of convergence numerically
    err_root_ycol_sol = np.dot(sol_root.fun, sol_root.fun)
    success = err_root_ycol_sol < 1E-3
    if success:
        tolerance_for_near = 1E-3 + err_root_ycol_sol

        def _rfp(_z0):
            _sol = optimize.root(res_fun_wrapped, _z0, jac=jac_fun, method='hybr')
            _err = _sol.x - sol_root.x
            return np.dot(_err, _err) < tolerance_for_near

        r_max = 100
        rconv = determine_radius_of_convergence(_rfp, sol_root.x, r_max=r_max)
        if rconv == r_max:
            # We hit numeric limit -> simply set to infinity
            rconv = np.inf
    else:
        rconv = 0.

    sol_dicts.append({
        'n': num_col,
        'tcol': np.append(tcol, tf),
        'ycol': np.append(ycol, yf),
        'tcol_hat': np.append(tcol_hat, tf_hat),
        'ycol_hat': np.append(ycol_hat, yf_hat),
        'etcol': np.append(etcol, etf),
        'ecol': np.append(ecol, ef),
        'success': success,
        'rconv': rconv
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
fig_col, axes_col = plt.subplots(nrows=2)

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
