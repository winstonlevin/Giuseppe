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
    with open('guess.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan
        sol.h_u = np.nan * sol.u.copy()
        sol.eig_h_uu = np.nan * sol.u.copy()
elif DATA == 1:
    with open('seed_sol.data', 'rb') as f:
        sol = pickle.load(f)
else:
    with open('sol_set.data', 'rb') as f:
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
v = (2*(k_dict['E'] - k_dict['g'] * x_dict['h']))**0.5
ham = 1. + lam_dict['h'] * v * np.sin(u_dict['gam']) + lam_dict['x'] * v * np.cos(u_dict['gam'])
g = k_dict['g']
vc = (2*k_dict['E'] - 2*k_dict['g'] * k_dict['h0'])**0.5
x = x_dict['x']
x_dist = np.abs(x[-1] - x[0])

if PLOT_AUX:
    # Newton Search for tha : Vc/Vmax = cos(tha) -> Vmax = Vc / cos(tha)
    v_ratio_giu = vc * np.mean(np.abs(lam_dict['x']))
    const = g/vc**2 * x_dist
    tha_guess0 = np.arctan(const)

    # v_ratio_min = -0.5*const + (0.25*const**2 + 1)**0.5
    # v_ratio_max = min(1., (np.pi/(2*const))**0.5)
    # const_min = np.arccos(v_ratio_max) / v_ratio_max**2 + v_ratio_max * (1 - v_ratio_max**2)**0.5
    # const_max = np.arccos(v_ratio_min) / v_ratio_min**2 + v_ratio_min * (1 - v_ratio_min**2)**0.5
    #
    # for idx in range(1000):
    #     v_ratio_guess = 0.5 * (v_ratio_min + v_ratio_max)
    #     const_guess = np.arccos(v_ratio_guess) / v_ratio_guess**2 + v_ratio_guess * (1 - v_ratio_guess**2)**0.5
    #     if const_guess > const:
    #         # V0/Vmax too low (i.e. req. lower Vmax)
    #         v_ratio_min = v_ratio_guess
    #     else:
    #         # V0/Vmax too high (i.e. req. higher Vmax)
    #         v_ratio_max = v_ratio_guess
    #
    #     if abs(const - const_guess) < 1e-8:
    #         break

    def newton_fun(_tha):
        _c_tha2 = np.cos(_tha)**2
        _s_2tha = np.sin(2*_tha)
        _f = const * _c_tha2 - _tha - 0.5 * _s_2tha
        _g = 2 * _c_tha2 - const * _s_2tha - 1
        return _f, _g

    max_iter = 5
    tol = 1e-6
    max_reduction = tol ** (1 / max_iter)
    tha_guess = tha_guess0
    success = False
    fun_evals = 0
    for _ in range(100):
        res, grad = newton_fun(tha_guess0)
        fun_evals += 1
        cost = abs(res)
        sign_res = np.sign(res)

        alp = 1.
        cost_new = np.inf
        for __ in range(max_iter):
            tha_guess = tha_guess0 - alp*res/grad
            res_new, grad_new = newton_fun(tha_guess)
            fun_evals += 1
            cost_new = abs(res_new)
            sign_res_new = np.sign(res_new)

            if (cost_new < cost and sign_res == sign_res_new) or cost_new < 0.1 * cost:
                break
            else:
                # Adjust step size to reduce error with <10% overshoot
                alp *= np.maximum(0.75 * min(np.abs(res/res_new), 1.), max_reduction)

        if cost_new < cost:
            tha_guess0 = tha_guess
            if cost_new < tol:
                success = True
                break
        else:
            break

    # TODO - use tha_guess to calculate info.

    lam_h2 = lam_dict['h']**2
    lam_h2_analytic = lam_dict['h'][0]**2 - 1/v[0]**2 + 1/v**2
    r_gam = (lam_dict['h']**2 + lam_dict['x']**2)**0.5
    r_gam_analytic = 1 / v
    v_max_idx = np.argmax(v)
    v_max_analytic = np.abs(1 / np.mean(lam_dict['x']))
    t_s_analytic = 0.5*(sol.t[0] + sol.t[-1])
    x_s_analytic = 0.5*(x_dict['x'][-1] + x_dict['x'][0])
    v_analytic = v_max_analytic * np.cos(k_dict['g']/v_max_analytic * (sol.t - t_s_analytic))
    x_analytic = x_s_analytic \
        + 0.5 * v_max_analytic * (sol.t - t_s_analytic) \
        + v_max_analytic**2/(4*k_dict['g']) * np.sin(2*k_dict['g']/v_max_analytic * (sol.t - t_s_analytic))
    gam_analytic = np.arctan2(np.sign(sol.t - t_s_analytic) * ((v_max_analytic/v)**2 - 1)**0.5, np.sign(x_dict['x'][-1] - x_dict['x'][0]))

    z_analytic = vc / v_max_analytic
    res_v_max_analytic = k_dict['g']/vc**2 * np.abs(x_dict['x'][-1] - x_dict['x'][0]) * z_analytic**2 - (
        np.arccos(z_analytic) + z_analytic*(1-z_analytic**2)**0.5
    )
    res_tmp = k_dict['g']/v_max_analytic**2 * np.abs(x_dict['x'][-1] - x_dict['x'][0]) - (
        np.arccos(v_analytic[0]/v_max_analytic)
        + v_analytic[0]/v_max_analytic * (1 - (v_analytic[0]/v_max_analytic)**2)**0.5
    )
else:
    lam_h2 = None
    lam_h2_analytic = None
    r_gam = None
    r_gam_analytic = None
    gam_analytic = np.arctan2(-lam_dict['h'], -lam_dict['x'])

# PLOTTING -------------------------------------------------------------------------------------------------------------
t_label = 'Time [s]'

# PLOT STATES/CONTROLS
ylabs = (r'$h$ [m]', r'$V$ [m/s]', r'$x$ [m]', r'$\gamma$ [deg]')
ymult = np.array((1., 1., 1., r2d))
ydata = (x_dict['h'], v, x_dict['x'], u_dict['gam'])
yaux = (None, v_analytic, x_analytic, gam_analytic)
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(ydata):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state * ymult[idx])

    if yaux[idx] is not None:
        ax.plot(sol.t, yaux[idx] * ymult[idx], 'k--')

fig_states.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{h}$ [s/m]', r'$\lambda_{x}$ [s/m]', r'$H_u$ [1/rad]', r'$H_{uu}$ [1/rad$^2$]', r'$H$ [-]')
ymult = np.array((1., 1., 1., 1., 1.))
ydata = (lam_dict['h'], lam_dict['x'], sol.h_u.flatten(), sol.eig_h_uu.flatten(), ham)
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
