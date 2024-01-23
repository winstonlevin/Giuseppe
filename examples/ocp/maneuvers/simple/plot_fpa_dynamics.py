import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']
DATA = 2
PLOT_AUX = True

# Load data
if DATA == 0:
    with open('guess_fpa_dyn.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan
        sol.h_u = np.nan * sol.u.copy()
        sol.eig_h_uu = np.nan * sol.u.copy()
elif DATA == 1:
    with open('seed_sol_fpa_dyn.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set_fpa_dyn.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

# Create Dicts
k_dict = {}
x_dict = {}
lam_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val

for key, x_val, lam_val in zip(sol.annotations.states, list(sol.x), list(sol.lam)):
    x_dict[key] = x_val
    lam_dict[key] = lam_val

for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val

# Process data
r2d = 180/np.pi

g = k_dict['g']
E = k_dict['E']
u_max = k_dict['u_max']
u_min = k_dict['u_min']
eps = k_dict['eps']

h = x_dict['h']
x = x_dict['x']
gam = x_dict['gam']
lam_h = lam_dict['h']
lam_x = lam_dict['x']
lam_gam = lam_dict['gam']
u = u_dict['u']

v = (2*(E - g * x_dict['h']))**0.5
mu_u = 0.5 * (u_min + u_max)
r_u = 0.5 * (u_max - u_min)
dgam_dt = mu_u + r_u * np.sin(u)
ham = 1. + eps * (1 - np.cos(u)) + lam_h * v * np.sin(gam) + lam_x * v * np.cos(gam) + lam_gam * dgam_dt

hc = k_dict['h0']
vc = (2*(E - g * hc))**0.5
x_dist = np.abs(x[-1] - x[0])

if PLOT_AUX:
    # Newton Search for tha : Vc/Vmax = cos(tha) -> Vmax = Vc / cos(tha)
    v_ratio_giu = vc * np.mean(np.abs(lam_dict['x']))
    const = g/vc**2 * x_dist

    def newton_step_fun(_z):
        # z is ratio of initial to maximum velocity.
        _acos_z = np.arccos(_z)
        _z2 = _z**2
        _sqrt = (1 - _z2)**0.5
        _full_step = 0.5 * (_z * _acos_z + _z2 * _sqrt - const*_z**3)*_sqrt/(_z + _acos_z*_sqrt)
        _f_pseudo = _acos_z + _sqrt * _z - const * _z ** 2
        return _full_step, _f_pseudo

    def newton_fun(_z):
        _acos_z = np.arccos(_z)
        _z2 = _z**2
        _sqrt = (1 - _z2)**0.5
        _f = _acos_z/_z2 + _sqrt/_z - const
        _g = -2 * (_z + _acos_z*_sqrt)/(_z**3 * _sqrt)
        return _f, _g

    def saturate_full_step(_z, _z_min, _z_max, _full_step):
        if _full_step < 0:
            # Decreasing z -> ensure z >= z_min
            _step = -min(-_full_step, _z - _z_min)
        else:
            # Increasing z -> ensure z <= z_max
            _step = max(_full_step, _z_max - _z)
        return _step

    max_iter = 5
    tol = 1e-6
    max_reduction = (0.1*tol) ** (1 / max_iter)
    success = False

    v_ratio_max = 1. - 0.5 * tol
    v_ratio_min = 0.5 * tol
    v_ratio = max(min(v_ratio_max, (np.pi / (2 * const)) ** 0.5), v_ratio_min)
    full_step, res = newton_step_fun(v_ratio)
    full_step = saturate_full_step(v_ratio, v_ratio_min, v_ratio_max, full_step)
    fun_evals = 1
    cost = abs(res)
    sign_step = np.sign(full_step)
    alp = 1.
    for _ in range(100):
        if cost < tol:
            success = True
            break

        backtrack_success = False
        for backtrack_idx in range(max_iter):
            v_ratio_new = v_ratio + alp*full_step
            step_new, res_new = newton_step_fun(v_ratio_new)
            fun_evals += 1
            cost_new = abs(res_new)
            sign_step_new = np.sign(step_new)

            if sign_step == sign_step_new or cost_new < 0.15 * cost:
                v_ratio = v_ratio_new
                full_step = saturate_full_step(v_ratio, v_ratio_min, v_ratio_max, step_new)
                cost = cost_new
                sign_step = sign_step_new
                backtrack_success = True
                alp = min(1., 2*alp)  # Increase step size toward original
                break
            else:
                # Adjust step size to reduce error with <10% overshoot
                error_adjust = min(max(cost/cost_new, 0.1), 1.)
                alp *= 0.5 ** (1. + backtrack_idx) * error_adjust

        if not backtrack_success:
            break

    if success:
        print(f'Found V0/Vmax after {fun_evals} iterations.')
    else:
        print(f'Failure to find V0/Vmax after {fun_evals} iterations!')

    # Numerical values
    lam_h2 = lam_dict['h']**2
    r_gam = (lam_h2 + lam_dict['x']**2)**0.5

    # Analytic values
    v_max_analytic = vc / v_ratio
    t_s_analytic = v_max_analytic/g * np.arccos(v_ratio)
    tf_analytic = 2 * t_s_analytic
    t_analytic = sol.t
    v_analytic = v_max_analytic * np.cos(g/v_max_analytic * (t_analytic - t_s_analytic))
    lam_h2_analytic = 1/v_analytic**2 - 1/v_max_analytic**2
    lam_x2_analytic = (v_ratio/vc)**2
    r_gam_analytic = 1 / v_analytic
    x_s_analytic = 0. + np.sign(k_dict['xf']) * (0.5 * v_max_analytic * t_s_analytic + v_max_analytic**2/(4*g) * np.sin(2*g/v_max_analytic * t_s_analytic))
    x_analytic = x_s_analytic \
        + 0.5 * v_max_analytic * (t_analytic - t_s_analytic) \
        + v_max_analytic**2/(4*g) * np.sin(2*g/v_max_analytic * (t_analytic- t_s_analytic))
    gam_analytic = np.arctan2(np.sign(t_analytic - t_s_analytic) * np.maximum((v_max_analytic/v_analytic)**2 - 1, 0.)**0.5, np.sign(x_analytic[-1] - x_analytic[0]))
    u_analytic = None
else:
    lam_h2 = None
    lam_h2_analytic = None
    r_gam = None
    r_gam_analytic = None
    gam_analytic = None
    v_analytic = None
    x_analytic = None
    u_analytic = None

# PLOTTING -------------------------------------------------------------------------------------------------------------
t_label = 'Time [s]'

# PLOT STATES
ylabs = (r'$h$ [m]', r'$V$ [m/s]', r'$x$ [m]', r'$\gamma$ [deg]', r'$u$ [-]', r'$\dot{\gamma}$ [deg/s]')
ymult = np.array((1., 1., 1., r2d, 1., r2d))
ydata = (h, v, x, gam, u, dgam_dt)
yaux = (None, v_analytic, x_analytic, gam_analytic, u_analytic, None)
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(ydata):
    axes_states.append(fig_states.add_subplot(3, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

    if yaux[idx] is not None:
        ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--')

fig_states.tight_layout()

# PLOT COSTATES
ylabs = (
    r'$\lambda_{h}$ [s/m]', r'$\lambda_{x}$ [s/m]', r'$\lambda_{\gamma}$ [s]',
    r'$H_u$ [1/rad]', r'$H_{uu}$ [1/rad$^2$]', r'$H$ [-]'
)
ymult = np.array((1., 1., 1., 1., 1., 1.))
ydata = (lam_h, lam_x, lam_gam, sol.h_u.flatten(), sol.eig_h_uu.flatten(), ham)
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(ydata):
    axes_costates.append(fig_costates.add_subplot(3, 2, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate * ymult[idx])

fig_costates.tight_layout()

if PLOT_AUX:
    ydata = (lam_h2, r_gam,)
    yaux = (lam_h2_analytic, r_gam_analytic,)
    yerr = (lam_h2 - lam_h2_analytic, r_gam - r_gam_analytic,)
    ylabs_names = (r'$\lambda_h^2$', r'$R_{\gamma}$')
    ylabs_units = (r'[s$^2$/m$^s$]', r'[s/m]')
    n_aux = len(ydata)
    fig_aux = plt.figure()
    axes_aux = []

    for idx, y in enumerate(ydata):
        axes_aux.append(fig_aux.add_subplot(n_aux, 2, 2*idx + 1))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs_names[idx] + ' ' + ylabs_units[idx])
        ax.plot(sol.t, y * ymult[idx])
        ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--')

        axes_aux.append(fig_aux.add_subplot(n_aux, 2, 2 * idx + 2))
        ax = axes_aux[-1]
        ax.grid()
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabs_names[idx] + ' err. ' + ylabs_units[idx])
        ax.plot(sol.t, yerr[idx] * ymult[idx])

    fig_aux.tight_layout()

plt.show()
