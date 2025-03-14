from typing import Optional
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


def full_newton_solver(
        res_fun: ca.Function, jac_fun: ca.Function,
        z: np.array, tol: float = 1E-3, max_iter_without_improvement=4, max_jev: int = 100,
        stall_fraction: float = 0.95
):
    """

    Parameters
    ----------
    res_fun, residual function
    jac_fun, jacobian function
    z, initial guess for root
    tol, tolerance to break before maximum iterations are exceeded.
    max_iter_without_improvement, int, default=4, maximum iteration without desired cost reduction before giving up.
    max_jev, int, default=100, maximum iterations before failing.
    stall_fraction, float, default=0.95, equal to 1 - required_accuracy. When cost/cost_old below this ratio, considered
    an iteration without improvement.

    Returns
    -------
    dict with fields:
        'z': np.ndarray, root
        'res': np.ndarray, residual, equal to res_fun(z)
        'njev': int, number of jacobian evaluations and inversions performed
        'success': bool, True if residual is below tolerance
        'message': str, message associated with success
    """
    _z = z.copy().ravel()  # _z will be modified in-place -> make copy
    res = res_fun(z).full().ravel()
    cost_old = np.abs(res).max(initial=0.)
    success = False
    message = 'Failure: Exceeded maximum number of Jacobian evaluations.'
    njev = 0
    niter_without_improvement = 0
    for njev in range(1, max_jev+1):
        # Generate step
        try:
            lu = splu(jac_fun(z).sparse())
        except RuntimeError:
            message = 'Failure: Jacobian cannot be inverted.'
            break

        _z -= lu.solve(res)
        res = res_fun(_z).full().ravel()
        cost = np.abs(res).max(initial=0.)
        if cost < tol:
            success = True
            message = 'Success: maximum residual is below tolerance.'
            break
        elif cost_old*stall_fraction < cost:
            niter_without_improvement += 1
            if niter_without_improvement == max_iter_without_improvement:
                message = 'Failure: Exceeded maximum number of iterations without improvement!'
                break
        else:
            # Reset iterations without improvment
            niter_without_improvement = 0
            cost_old = cost

    return {'z': _z, 'res': res, 'njev': njev, 'success': success, 'message': message}


def damped_newton_solver(
        res_fun: ca.Function, jac_fun: ca.Function, z: np.array,
        alpha_min: float = 1E-6, alpha_max: float = 1., acc_min: float = 0.75,
        fac_decrease: float = 0.5, fac_increase: float = 8.,
        max_jev: int = 100, max_fev: int = 100, tol: float = 1E-3,
        max_jev_steps: int = 4
):
    z = z.reshape((-1, 1))
    nfev = 0
    njev = 0
    jev_steps = 0
    res = res_fun(z).full()
    nfev += 1
    cost = np.abs(res).max(initial=0.)
    alpha = alpha_max
    success = True
    message = 'Success: maximum residual is below tolerance.'
    recompute_jac = True

    while cost > tol:
        if recompute_jac:
            # Compute step direction (Newton step)
            if njev == max_jev:
                success = False
                message = 'Failure: exceeded maximum number of Jacobian evaluations.'
                break

            jac = jac_fun(z).sparse()
            njev += 1
            recompute_jac = False
            try:
                lu = splu(jac)
            except RuntimeError:
                success = False
                message = 'Failure: Jacobian cannot be inverted.'
                break
            step = lu.solve(-res)
            jev_steps = 1

        # Evaluate damped step
        z_new = z + alpha * step
        if nfev == max_fev:
            success = False
            message = 'Failure: exceeded maximum number of residual evaluations.'
            break
        res_new = res_fun(z_new).full()
        nfev += 1
        cost_new = np.abs(res_new).max(initial=0.)

        # Accuracy = (1 - cost_new/cost_old)/alpha < Acc_min                --> Fail. Equivalently:
        #            (cost_old - cost_new)/alpha   < acc_min*cost_old
        #             cost_old - cost_new          < alpha*acc_min*cost_old
        #             cost_old*(1 - alpha*acc_min) < cost_new
        if cost*(1 - alpha*acc_min) < cost_new:  # accuracy:
            # Reject step
            if jev_steps == 1:
                # This step is a new Jacobian, decrease alpha
                alpha *= fac_decrease
                if alpha < alpha_min:
                    success = False
                    message = 'Failure: damping parameter below minimum value.'
                    break
            else:
                # This step is a re-used Jacobian, re-calculate Jacobian
                recompute_jac = True

        else:
            # Accept step
            z = z_new
            res = res_new
            cost = cost_new

            if alpha == alpha_max and jev_steps < max_jev_steps:
                # Keep same Jac for up to "max_jev_steps" steps instead of recomputing inverse
                jev_steps += 1
                step = lu.solve(-res)
            else:
                alpha *= fac_increase
                alpha = alpha_max if alpha > alpha_max else alpha
                recompute_jac = True

    return {'z': z, 'res': res, 'njev': njev, 'nfev': nfev, 'message': message, 'success': success}


def determine_radius_of_convergence(
        root_finding_problem, sol_converged, n_samples: Optional[int] = None,
        r_upper: float = 100., rng_seed=None, max_iter: int = 1_000, tol: float = 1E-3,
):
    """
    Determined the radius of convergence around the root z* via a binary search. The validity of a given radius is
    determined by:
        1. Generate co-varied points about the optimal solution at distance R away from optimal solution
        2. If any fail, this is outside the radius of convergence
    NOTE: since a binary search is used, it is implicitly assumed that increasing R will lower the likelihood of
    convergence.

    Parameters
    ----------
    root_finding_problem, Callable, function taking in the initial guess and outputting a boolean flag of success.
    sol_converged, (n,) np.ndarray, root of the root-finding problem about which to find the radius of convergence.
    n_samples, int, default=n, Number of perturb samples.
    r_max, float, default=100.,
    rng_seed, int or None, default=None, seed to be used in sampling perturbations to the guess vector
    max_iter, int, default=1_000, maximum number of search steps to find radius of convergence
    tol, float, default=1E-3, tolerance for binary search (break when upper/lower are within this value of each other)

    Returns
    -------
    if _return_continuous, returns an interpolant of the fraction of successful results vs. radius of convergence.
    Otherwise, returns radius of convergence as a float.
    """
    # Generate the unit vectors for the samples
    sample_unit_vectors = orthonormal_sampler(n=len(sol_converged), n_samples=n_samples, rng_seed=rng_seed)

    def _check_if_convergent(_r):
        for _idx_sample, _sample in enumerate(sample_unit_vectors):
            if not root_finding_problem(sol_converged + _r * _sample):
                # Move failed sample to front to speed up next check
                sample_unit_vectors.insert(0, sample_unit_vectors.pop(_idx_sample))
                return False
        return True

    # Save list of prior values
    r_upper_convergent = _check_if_convergent(r_upper)
    if r_upper_convergent:
        return r_upper

    # Conduct binary search to determine where r stops converging
    # (It occurs at some value between 0 and r_upper)
    # Since it is much faster to check failure (only one needs to fail rather than all needing to pass),
    # the gain is biased toward r_upper by setting:
    # r = 1/3 r_lower + 2/3 r_upper
    r_lower = 0.
    for iteration in range(max_iter):
        r = (r_lower + r_upper*2)/3
        if _check_if_convergent(r):
            # This value is inside the radius of convergence
            r_lower = r
        else:
            # This value is outside the radius of convergence
            r_upper = r

        if abs(r_lower - r_upper) < tol:
            break

    return (r_lower + r_upper)/2

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

        def _preprocess_guess(_z):
            pass

        def _postprocess_guess(_z):
            pass

    else:
        # Root (free terminal time, fixed terminal state)
        if use_log_tf:
            log_tf_sym = ca.SX.sym('log_tf')
            z_sym = ca.vcat((log_tf_sym, ycol_sym))
            tf_sym = np.exp(log_tf_sym)

            def _preprocess_guess(_z):
                np.maximum(_z[0], tf_min, out=_z[0:1])
                _z[0] = np.log(_z[0], out=_z[0:1])

            def _postprocess_guess(_z):
                np.exp(_z[0], out=_z[0:1])

        else:
            tf_sym = ca.SX.sym('tf')
            z_sym = ca.vcat((tf_sym, ycol_sym))

            def _preprocess_guess(_z):
                if _z[0] < tf_min:
                    _z[0] = tf_min

            def _postprocess_guess(_z):
                pass

        z_true = np.concatenate(((tf,), ycol))
        yf_sym = yf_fun(ycol_sym, y0, tf_sym)
        res_sym = ca.vcat((yf_sym - yf, rcol_fun(ycol_sym, y0, tf_sym)))
        solution_fun = ca.Function('s', (z_sym,), (tf_sym, yf_sym, ycol_sym))

    jac_sym = ca.jacobian(res_sym, z_sym)
    res_fun = ca.Function('r', (z_sym,), (res_sym,))

    # def res_fun_wrapped(_z):
    #     return res_fun(_z).full()[:, 0]

    jac_fun = ca.Function('J', (z_sym,), (jac_sym,), ('z',), ('J',))

    sol_root = full_newton_solver(res_fun=res_fun, jac_fun=jac_fun, z=z_true, tol=1E-4)
    # sol_root = optimize.root(res_fun_wrapped, z_true, jac=jac_fun, method='hybr')
    # tf_hat, yf_hat, ycol_hat = solution_fun(sol_root.x)
    tf_hat, yf_hat, ycol_hat = solution_fun(sol_root['z'])
    tf_hat = float(tf_hat)  # Convert to non-CasADi type
    yf_hat = float(yf_hat)
    ycol_hat = ycol_hat.full()[:, 0]
    tcol_hat = tcol_fun(tf_hat).full()[:, 0]

    # Errors
    root_error = sol_root['z'][:, 0] - z_true
    # root_error = sol_root.x - z_true
    norm_root_err = np.dot(root_error, root_error)**0.5

    ef = yf_hat - yf
    ecol = ycol_hat - ycol
    etcol = tcol_hat - tcol
    etf = tf_hat - tf

    # Determine radius of convergence numerically
    # err_root_ycol_sol = np.dot(sol_root.fun, sol_root.fun)
    # success = err_root_ycol_sol < 1E-3
    success = np.abs(sol_root['res']).max(initial=0.) < 1E-3
    if success:
        err_root_ycol_sol = sol_root['res'].T.dot(sol_root['res'])
        tolerance_for_near = 1E-3 + err_root_ycol_sol
        _postprocess_guess(sol_root['z'])
        # _postprocess_guess(sol_root.x)

        def _rfp(_z0):
            _preprocess_guess(_z0)
            _sol = full_newton_solver(res_fun=res_fun, jac_fun=jac_fun, z=_z0, tol=1E-4)
            _postprocess_guess(_sol['z'])
            _err = _sol['z'] - sol_root['z']
            # _sol = optimize.root(res_fun_wrapped, _z0, jac=jac_fun, method='hybr')
            # _postprocess_guess(_sol.x)
            # _err = _sol.x - sol_root.x
            return _err.T.dot(_err) < tolerance_for_near

        rconv = determine_radius_of_convergence(
            _rfp, sol_root['z'][:, 0], r_upper=r_max, n_samples=50, rng_seed=rng_seed, tol=1E-2
        )
        # rconv = determine_radius_of_convergence(
        #     _rfp, sol_root.x, r_max=r_max, n_samples=50, rng_seed=rng_seed, confidence=0.95
        # )
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
