from typing import Callable, Tuple

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as sp

INTERPOLATE_U = False

# ---- UNPACK DATA -----------------------------------------------------------------------------------------------------
mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

with open('sol_set_hl20.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]
    sols = [sol]

# Create Dicts
k_dict = {}
p_dict = {}
x_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val
for key, val in zip(sol.annotations.parameters, sol.p):
    p_dict[key] = val
for key, val in zip(sol.annotations.states, list(sol.x)):
    x_dict[key] = val
for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val


hf = 5e3

# CONTROL LIMITS
n_max = 4.5
n_min = -4.5

# STATE LIMITS
gam_max = 85 * np.pi / 180
gam_min = -gam_max


# ---- DYNAMICS & CONTROL LAWS -----------------------------------------------------------------------------------------
def saturate(_val, _val_min, _val_max):
    return max(_val_min, min(_val_max, _val))


def interp_control_law(_t: float, _u_interp: sp.interpolate.PchipInterpolator):
    return np.array((_u_interp(_t),))


def lift_control_law(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict, _max: bool = True) -> np.array:
    _h = _x[0]
    _v = _x[2]

    _r = _h + _k_dict['rm']
    _g = _k_dict['mu'] / _r**2
    _g0 = _k_dict['mu'] / _k_dict['rm'] ** 2
    _qdyn = 0.5 * _k_dict['rho0'] * np.exp(-_h/_k_dict['h_ref']) * _v ** 2

    if _max:
        _cl_max = _k_dict['CL1'] * 0.5
        _lift_max = _qdyn * _k_dict['s_ref'] * _cl_max
        _lift_n_max = _k_dict['mass'] * _g0 * k_dict['n_max']
        _lift = min(_lift_max, _lift_n_max)
    else:
        _cl_min = -_k_dict['CL1'] * 0.5
        _lift_min = _qdyn * _k_dict['s_ref'] * _cl_min
        _lift_n_min = _k_dict['mass'] * _g0 * k_dict['n_min']
        _lift = max(_lift_min, _lift_n_min)

    _sin = 2 * _lift / (_qdyn * _k_dict['s_ref'] * _k_dict['CL1'])
    _sin_sat = saturate(_sin, _val_min=-1., _val_max=1.)
    _asin = np.arcsin(_sin_sat)
    _alpha = 0.5 * _asin - _k_dict['CL0'] / _k_dict['CL1']

    return np.array((_alpha,))


def eom(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _theta = _x[1]
    _v = _x[2]
    _gam = _x[3]

    _alpha = _u[0]

    _r = _h + _k_dict['rm']
    _g = _k_dict['mu'] / _r**2
    _g0 = _k_dict['mu'] / _k_dict['rm'] ** 2
    _qdyn = 0.5 * _k_dict['rho0'] * np.exp(-_h/_k_dict['h_ref']) * _v ** 2
    _cl = _k_dict['CL1'] * 0.5 * np.sin(2 * (_alpha + _k_dict['CL0']/_k_dict['CL1']))
    _cd = _k_dict['CD0'] - _k_dict['CD1']**2/(4*_k_dict['CD2']) + \
          _k_dict['CD2'] * np.sin(_alpha + _k_dict['CD1']/(2*_k_dict['CD2']))**2
    _lift = _qdyn * _k_dict['s_ref'] * _cl
    _drag = _qdyn * _k_dict['s_ref'] * _cd

    _dh = _v * np.sin(_gam)
    _dtheta = _v * np.cos(_gam) / _r
    _dv = -_drag / _k_dict['mass']  - _g * np.sin(_gam)
    _dgam = _lift / (_k_dict['mass'] * _v) + (_v/_r - _g/_v) * np.cos(_gam)

    return np.array((_dh, _dtheta, _dv, _dgam))


def estimate_hf(_t: float, _x: np.array, _p_dict: dict, _k_dict: dict, _n_steps: int = 3):
    _h0 = _x[0]
    _theta0 = _x[1]
    _v0 = _x[2]
    _gam0 = _x[3]

    def dhv_dgam(__gam, __hv):
        __x = np.array((__hv[0], _theta0, __hv[1], __gam))
        __u = lift_control_law(_t, __x, _p_dict, _k_dict)
        dx_dt = eom(_t, __x, __u, _p_dict, _k_dict)
        return np.array((dx_dt[0] / dx_dt[3], dx_dt[2] / dx_dt[3]))

    # RK-4 Estimate of hf, Vf
    _step = (0. - _gam0) / _n_steps
    _x0 = np.array((_h0, _v0))
    _x = _x0.copy()
    _gam = _gam0.copy()
    for _ in range(_n_steps):
        _k1 = dhv_dgam(_gam, _x)
        _k2 = dhv_dgam(_gam + _step/2., _x + _k1 * _step/2.)
        _k3 = dhv_dgam(_gam + _step/2., _x + _k2 * _step/2.)
        _k4 = dhv_dgam(_gam + _step, _x + _k3 * _step)
        _x += (_k1/6. + _k2/3. + _k3/3. + _k4/6.) * _step
        _gam += _step

    return _x[0]


# This event estimates when a max-lift pull up will achieve the terminal altitude. When this occurs, stop integrating
# with min(Lift) and start pulling up with max(Lift)
def generate_pull_up_event(_p_dict, _k_dict, _n_steps: int = 3) -> Callable:
    def _begin_pull_up_event(_t: float, _x: np.array) -> float:
        _h = _x[0]
        _gam = _x[3]
        hf_cmd = _k_dict['hf']

        if _gam > 0:
            hf_pull_up = _h
        else:
            hf_pull_up = estimate_hf(_t, _x, _p_dict, _k_dict, _n_steps=_n_steps)

        return hf_pull_up - hf_cmd

    _begin_pull_up_event.terminal = True
    _begin_pull_up_event.direction = 0
    return _begin_pull_up_event


# This event stops the pull up when the terminal FPA (0 deg) is achieved.
def cruise_fpa_event(_t: float, _x: np.array) -> float:
    return _x[3]


cruise_fpa_event.terminal = True
cruise_fpa_event.direction = 0


# ---- RUN SIM ---------------------------------------------------------------------------------------------------------
rtol = 1e-6
atol = 1e-8

x_scale = np.array((k_dict['h_scale'], k_dict['theta_scale'], k_dict['v_scale'], k_dict['gam_scale']))

n_sols = len(sols)
ivp_sols_dict = [{}] * n_sols
h0_arr = np.empty((n_sols,))
v0_arr = np.empty((n_sols,))
opt_arr = np.empty((n_sols,))

for idx, sol in enumerate(sols):
    for key, val in zip(sol.annotations.states, list(sol.x)):
        x_dict[key] = val

    # IVP Parameters
    t_max = 2 * sol.t[-1]
    x0 = sol.x[:, 0] * x_scale

    if INTERPOLATE_U:
        t_span = np.array((sol.t[0], sol.t[-1]))
        u_interp = sp.interpolate.PchipInterpolator(sol.t, sol.u[0, :])
        ivp_sol = sp.integrate.solve_ivp(
            fun=lambda t, x: eom(t, x, interp_control_law(t, u_interp), p_dict, k_dict),
            t_span=t_span,
            y0=x0,
            rtol=rtol, atol=atol
        )

        ivp_sols_dict[idx] = {
            't': ivp_sol.t,
            'x': ivp_sol.y,
        }
    else:
        t_span = np.array((sol.t[0], t_max))

        # Integrate pull down maneuver
        begin_pull_up_event = generate_pull_up_event(p_dict, k_dict, 25)

        ivp_sol1 = sp.integrate.solve_ivp(
            fun=lambda t, x: eom(t, x, lift_control_law(t, x, p_dict, k_dict, _max=False), p_dict, k_dict),
            t_span=t_span,
            y0=x0,
            events=begin_pull_up_event,
            rtol=rtol, atol=atol
        )
        _t_1 = ivp_sol1.t
        _x_1 = ivp_sol1.y

        if ivp_sol1.t[-1] < t_span[-1]:
            ivp_sol2 = sp.integrate.solve_ivp(
                fun=lambda t, x: eom(t, x, lift_control_law(t, x, p_dict, k_dict, _max=True), p_dict, k_dict),
                t_span=np.array((ivp_sol1.t[-1], t_span[-1])),
                y0=ivp_sol1.y[:, -1],
                events=cruise_fpa_event,
                rtol=rtol, atol=atol
            )

            _t_2 = ivp_sol2.t
            _x_2 = ivp_sol2.y

            ivp_sols_dict[idx] = {
                't': np.concatenate((_t_1, _t_2)),
                'x': np.hstack((_x_1, _x_2)),
            }
        else:
            ivp_sols_dict[idx] = {
                't': _t_1,
                'x': _x_1,
            }


# ---- PLOTTING --------------------------------------------------------------------------------------------------------
gradient = mpl.colormaps['viridis'].colors

if len(sols) == 1:
    grad_idcs = np.array((0,), dtype=np.int32)
else:
    grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(sols))))


def cols_gradient(n):
    return gradient[grad_idcs[n]]


t_label = r'$t$ [s]'
r2d = 180 / np.pi

# PLOT STATES
ylabs = (r'$h$ [km]', r'$\theta$ [deg]', r'$V$ [km/s]', r'$\gamma$ [deg]')
ymult = np.array((1e-3, r2d, 1e-3, r2d))
fig_states = plt.figure()
axes_states = []

for idx, lab in enumerate(ylabs):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabs[idx])

    for jdx, (ivp_sol_dict, sol) in enumerate(zip(ivp_sols_dict, sols)):
        ax.plot(sol.t, sol.x[idx, :] * x_scale[idx] * ymult[idx], 'k--')
        ax.plot(ivp_sol_dict['t'], ivp_sol_dict['x'][idx, :] * ymult[idx], color=cols_gradient(jdx))

fig_states.tight_layout()

plt.show()
